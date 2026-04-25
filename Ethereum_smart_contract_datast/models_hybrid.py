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

from experiment_utils import VULN_TYPES, apply_thresholds, choose_thresholds, compute_multilabel_metrics
from models_gnn import _ASTCFGGraphBuilder, _GCNLayer


@dataclass
class HybridTrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    val_micro_f1: list[float]
    val_weighted_f1: list[float]
    best_epoch: int | None = None


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
        graph_residual_scale: float,
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
        self.graph_delta = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.graph_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.graph_residual_scale = graph_residual_scale
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, num_labels),
        )
        self.dropout = dropout
        self.num_labels = num_labels
        self._initialize_fusion_biases()

    def _initialize_fusion_biases(self):
        gate_output = self.graph_gate[-2]
        if isinstance(gate_output, nn.Linear) and gate_output.bias is not None:
            nn.init.constant_(gate_output.bias, -2.0)

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
        attended_modalities, _ = self.modality_attention(
            text_feature.unsqueeze(1),
            modalities,
            modalities,
        )
        attended_summary = attended_modalities.squeeze(1)
        difference = torch.abs(text_feature - graph_feature)
        interaction = text_feature * graph_feature
        fusion_inputs = torch.cat([text_feature, graph_feature, difference, interaction], dim=-1)
        gate = self.graph_gate(fusion_inputs)
        graph_delta = self.graph_delta(fusion_inputs)
        gated_fusion = text_feature + self.graph_residual_scale * gate * graph_delta

        fused = torch.cat(
            [text_feature, attended_summary, gated_fusion, difference],
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
        graph_residual_scale: float = 0.2,
        dropout: float = 0.2,
        train_batch_size: int = 4,
        eval_batch_size: int = 8,
        transformer_learning_rate: float = 2e-5,
        head_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        epochs: int = 3,
        max_pos_weight: float = 8.0,
        grad_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        encoder_warmup_epochs: int = 0,
        checkpoint_metric: str = "micro_f1",
        selection_candidate_thresholds: list[float] | None = None,
        selection_default_threshold: float = 0.5,
        selection_threshold_min_support: int = 5,
        selection_threshold_min_precision: float = 0.15,
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
        self.graph_residual_scale = graph_residual_scale
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.transformer_learning_rate = transformer_learning_rate
        self.head_learning_rate = head_learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_pos_weight = max_pos_weight
        self.grad_clip_norm = grad_clip_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.encoder_warmup_epochs = max(0, encoder_warmup_epochs)
        self.checkpoint_metric = checkpoint_metric
        self.selection_candidate_thresholds = selection_candidate_thresholds
        self.selection_default_threshold = selection_default_threshold
        self.selection_threshold_min_support = selection_threshold_min_support
        self.selection_threshold_min_precision = selection_threshold_min_precision
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: _HybridCodeGraphClassifier | None = None
        self.history = HybridTrainingHistory(
            train_losses=[],
            val_losses=[],
            val_micro_f1=[],
            val_weighted_f1=[],
        )
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
            graph_residual_scale=self.graph_residual_scale,
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

    def _set_encoder_trainable(self, trainable: bool):
        assert self.model is not None
        for param in self.model.text_encoder.parameters():
            param.requires_grad = trainable

    def _checkpoint_score(self, metrics: dict, val_loss: float) -> tuple[float, float, float]:
        primary = float(metrics.get(self.checkpoint_metric, 0.0))
        weighted = float(metrics.get("weighted_f1", 0.0))
        return (primary, weighted, -float(val_loss))

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
        best_score = None
        self.history = HybridTrainingHistory(
            train_losses=[],
            val_losses=[],
            val_micro_f1=[],
            val_weighted_f1=[],
        )

        for epoch in range(self.epochs):
            assert self.model is not None
            encoder_trainable = epoch >= self.encoder_warmup_epochs
            self._set_encoder_trainable(encoder_trainable)
            if epoch == 0 and self.encoder_warmup_epochs > 0:
                print(
                    f"[train] Freezing text encoder for {self.encoder_warmup_epochs} "
                    f"epoch(s) of warmup."
                )
            self.model.train()
            running_loss = 0.0
            total_examples = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                labels = batch.pop("labels").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch)
                raw_loss = loss_fn(logits, labels)
                loss = raw_loss / self.gradient_accumulation_steps
                loss.backward()
                should_step = (
                    step % self.gradient_accumulation_steps == 0
                    or step == len(train_loader)
                )
                if should_step:
                    if self.grad_clip_norm and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_size = labels.size(0)
                running_loss += raw_loss.item() * batch_size
                total_examples += batch_size

            train_loss = running_loss / max(total_examples, 1)
            self.history.train_losses.append(train_loss)

            if val_loader is None:
                print(f"[train] Epoch {epoch + 1}/{self.epochs} train_loss={train_loss:.4f}")
                continue

            val_loss, val_prob = self.evaluate_loss_and_probabilities(val_loader, loss_fn)
            self.history.val_losses.append(val_loss)
            thresholds = choose_thresholds(
                val_labels,
                val_prob,
                label_order=VULN_TYPES,
                candidate_thresholds=self.selection_candidate_thresholds,
                default_threshold=self.selection_default_threshold,
                min_support=self.selection_threshold_min_support,
                min_precision=self.selection_threshold_min_precision,
            )
            val_pred = apply_thresholds(val_prob, thresholds, label_order=VULN_TYPES)
            val_metrics = compute_multilabel_metrics(
                val_labels,
                val_pred,
                y_prob=val_prob,
                label_order=VULN_TYPES,
            )
            self.history.val_micro_f1.append(val_metrics["micro_f1"])
            self.history.val_weighted_f1.append(val_metrics["weighted_f1"])
            current_score = self._checkpoint_score(val_metrics, val_loss)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                self.history.best_epoch = epoch + 1

            print(
                f"[train] Epoch {epoch + 1}/{self.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def evaluate_loss(self, dataloader: DataLoader, loss_fn: nn.Module) -> float:
        loss, _ = self.evaluate_loss_and_probabilities(dataloader, loss_fn)
        return loss

    def evaluate_loss_and_probabilities(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> tuple[float, np.ndarray]:
        assert self.model is not None
        self.model.eval()
        running_loss = 0.0
        total_examples = 0
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch)
                loss = loss_fn(logits, labels)
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                total_examples += batch_size
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        if outputs:
            y_prob = np.vstack(outputs).astype(np.float32)
        else:
            y_prob = np.zeros((0, self.model.num_labels), dtype=np.float32)
        return running_loss / max(total_examples, 1), y_prob

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
            "graph_residual_scale": self.graph_residual_scale,
            "dropout": self.dropout,
            "transformer_learning_rate": self.transformer_learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "max_pos_weight": self.max_pos_weight,
            "grad_clip_norm": self.grad_clip_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "encoder_warmup_epochs": self.encoder_warmup_epochs,
            "checkpoint_metric": self.checkpoint_metric,
            "selection_candidate_thresholds": self.selection_candidate_thresholds,
            "selection_default_threshold": self.selection_default_threshold,
            "selection_threshold_min_support": self.selection_threshold_min_support,
            "selection_threshold_min_precision": self.selection_threshold_min_precision,
            "num_labels": self.model.num_labels,
        }
        (output_path / "hybrid_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
