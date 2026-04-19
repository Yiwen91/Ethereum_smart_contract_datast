#!/usr/bin/env python3
"""
Hybrid CodeBERT + AST/CFG GNN model for function-level multilabel classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from models_gnn import _ASTCFGGraphBuilder, _GCNLayer


@dataclass
class HybridTrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


class _HybridFunctionDataset(Dataset):
    def __init__(self, records: list[dict], labels: np.ndarray):
        self.records = records
        self.labels = labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return {
            "record": self.records[idx],
            "labels": self.labels[idx],
        }


class _HybridCodeGraphClassifier(nn.Module):
    def __init__(
        self,
        *,
        text_encoder: nn.Module,
        graph_layers: list[nn.Module],
        encoder_hidden_size: int,
        graph_hidden_dim: int,
        fusion_dim: int,
        attention_heads: int,
        num_labels: int,
        dropout: float,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.graph_layers = nn.ModuleList(graph_layers)
        self.graph_norm = nn.LayerNorm(graph_hidden_dim * 2)
        self.text_proj = nn.Linear(encoder_hidden_size, fusion_dim)
        self.graph_proj = nn.Linear(graph_hidden_dim * 2, fusion_dim)
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 5),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 5, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, num_labels),
        )
        self.dropout = dropout
        self.num_labels = num_labels

    def _encode_graph(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.graph_layers:
            h = layer(h, adj)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        mask_expanded = mask.unsqueeze(-1)
        pooled_mean = (h * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        masked_h = h.masked_fill(mask_expanded == 0, float("-inf"))
        pooled_max = masked_h.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.graph_norm(torch.cat([pooled_mean, pooled_max], dim=-1))

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        text_embedding = self._encode_text(input_ids=input_ids, attention_mask=attention_mask)
        graph_embedding = self._encode_graph(x=x, adj=adj, mask=mask)

        text_feature = self.text_proj(text_embedding)
        graph_feature = self.graph_proj(graph_embedding)

        modalities = torch.stack([text_feature, graph_feature], dim=1)
        attended_modalities, _ = self.modality_attention(modalities, modalities, modalities)
        attended_summary = attended_modalities.mean(dim=1)

        gate = self.gate(torch.cat([text_feature, graph_feature], dim=-1))
        gated_fusion = gate * text_feature + (1.0 - gate) * graph_feature
        difference = torch.abs(text_feature - graph_feature)

        fused = torch.cat(
            [text_feature, graph_feature, attended_summary, gated_fusion, difference],
            dim=-1,
        )
        return self.classifier(fused)


class HybridCodeBERTGNNMultilabelBaseline:
    def __init__(
        self,
        *,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 256,
        max_nodes: int = 128,
        feature_dim: int = 256,
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 2,
        fusion_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.2,
        train_batch_size: int = 4,
        eval_batch_size: int = 8,
        transformer_learning_rate: float = 2e-5,
        head_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        epochs: int = 3,
        max_pos_weight: float = 8.0,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_num_layers = graph_num_layers
        self.fusion_dim = fusion_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.transformer_learning_rate = transformer_learning_rate
        self.head_learning_rate = head_learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_pos_weight = max_pos_weight
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: _HybridCodeGraphClassifier | None = None
        self.history = HybridTrainingHistory(train_losses=[], val_losses=[])
        self.graph_builder = _ASTCFGGraphBuilder(max_nodes=max_nodes, feature_dim=feature_dim)
        self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, num_labels: int):
        text_encoder = AutoModel.from_pretrained(self.model_name)
        encoder_hidden_size = int(text_encoder.config.hidden_size)
        graph_layers: list[nn.Module] = []
        input_dim = self.feature_dim
        for _ in range(self.graph_num_layers):
            graph_layers.append(_GCNLayer(input_dim, self.graph_hidden_dim))
            input_dim = self.graph_hidden_dim
        self.model = _HybridCodeGraphClassifier(
            text_encoder=text_encoder,
            graph_layers=graph_layers,
            encoder_hidden_size=encoder_hidden_size,
            graph_hidden_dim=self.graph_hidden_dim,
            fusion_dim=self.fusion_dim,
            attention_heads=self.attention_heads,
            num_labels=num_labels,
            dropout=self.dropout,
        ).to(self.device)

    def _collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        texts = [item["record"].get("function_code", "") for item in batch]
        labels = np.asarray([item["labels"] for item in batch], dtype=np.float32)
        features = np.zeros((batch_size, self.max_nodes, self.feature_dim), dtype=np.float32)
        adjacency = np.zeros((batch_size, self.max_nodes, self.max_nodes), dtype=np.float32)
        masks = np.zeros((batch_size, self.max_nodes), dtype=np.float32)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        for idx, item in enumerate(batch):
            x, adj, mask = self.graph_builder.build_graph(item["record"])
            features[idx] = x
            adjacency[idx] = adj
            masks[idx] = mask

        adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
        degree = adjacency_tensor.sum(dim=-1)
        degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1.0), -0.5)
        normalized_adjacency = adjacency_tensor * degree_inv_sqrt.unsqueeze(-1) * degree_inv_sqrt.unsqueeze(-2)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "x": torch.tensor(features, dtype=torch.float32),
            "adj": normalized_adjacency,
            "mask": torch.tensor(masks, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    def _compute_pos_weight(self, labels: np.ndarray) -> torch.Tensor:
        positives = labels.sum(axis=0)
        negatives = labels.shape[0] - positives
        weights = np.ones_like(positives, dtype=np.float32)
        mask = positives > 0
        weights[mask] = np.sqrt(negatives[mask] / positives[mask])
        weights = np.clip(weights, 1.0, self.max_pos_weight)
        return torch.tensor(weights, dtype=torch.float32)

    def _build_optimizer(self) -> AdamW:
        assert self.model is not None
        head_parameters: list[nn.Parameter] = []
        encoder_parameter_ids = {id(param) for param in self.model.text_encoder.parameters()}
        for param in self.model.parameters():
            if id(param) not in encoder_parameter_ids:
                head_parameters.append(param)
        return AdamW(
            [
                {
                    "params": list(self.model.text_encoder.parameters()),
                    "lr": self.transformer_learning_rate,
                },
                {
                    "params": head_parameters,
                    "lr": self.head_learning_rate,
                },
            ],
            weight_decay=self.weight_decay,
        )

    def fit(
        self,
        train_records: list[dict],
        train_labels: np.ndarray,
        val_records: list[dict] | None = None,
        val_labels: np.ndarray | None = None,
    ):
        self._build_model(train_labels.shape[1])
        train_dataset = _HybridFunctionDataset(train_records, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

        pos_weight = self._compute_pos_weight(train_labels).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = self._build_optimizer()
        print(f"[train] Using capped sqrt pos_weight: {pos_weight.detach().cpu().tolist()}")

        val_loader = None
        if val_records is not None and val_labels is not None and len(val_records) > 0:
            val_dataset = _HybridFunctionDataset(val_records, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )

        best_state = None
        best_val_loss = None
        self.history = HybridTrainingHistory(train_losses=[], val_losses=[])

        for epoch in range(self.epochs):
            assert self.model is not None
            self.model.train()
            running_loss = 0.0
            total_examples = 0
            for batch in train_loader:
                labels = batch.pop("labels").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad()
                logits = self.model(**batch)
                loss = loss_fn(logits, labels)
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                total_examples += batch_size

            train_loss = running_loss / max(total_examples, 1)
            self.history.train_losses.append(train_loss)

            if val_loader is None:
                print(f"[train] Epoch {epoch + 1}/{self.epochs} train_loss={train_loss:.4f}")
                continue

            val_loss = self.evaluate_loss(val_loader, loss_fn)
            self.history.val_losses.append(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

            print(
                f"[train] Epoch {epoch + 1}/{self.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def evaluate_loss(self, dataloader: DataLoader, loss_fn: nn.Module) -> float:
        assert self.model is not None
        self.model.eval()
        running_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch)
                loss = loss_fn(logits, labels)
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                total_examples += batch_size
        return running_loss / max(total_examples, 1)

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        assert self.model is not None
        dataset = _HybridFunctionDataset(
            records,
            np.zeros((len(records), self.model.num_labels), dtype=np.float32),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                batch.pop("labels")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch)
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        if not outputs:
            return np.zeros((0, self.model.num_labels), dtype=np.float32)
        return np.vstack(outputs).astype(np.float32)

    def save_model(self, output_dir: str):
        assert self.model is not None
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.text_encoder.save_pretrained(output_path / "text_encoder")
        self.tokenizer.save_pretrained(output_path / "text_encoder")
        torch.save(self.model.state_dict(), output_path / "hybrid_state.pt")
        metadata = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "max_nodes": self.max_nodes,
            "feature_dim": self.feature_dim,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_num_layers": self.graph_num_layers,
            "fusion_dim": self.fusion_dim,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "num_labels": self.model.num_labels,
        }
        (output_path / "hybrid_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
