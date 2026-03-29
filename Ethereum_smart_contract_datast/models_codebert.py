#!/usr/bin/env python3
"""
CodeBERT baseline for function-level multilabel classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _FunctionTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx],
        }


@dataclass
class CodeBERTTrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


class CodeBERTMultilabelBaseline:
    def __init__(
        self,
        *,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 256,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 2,
        max_pos_weight: float = 8.0,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_pos_weight = max_pos_weight
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.history = CodeBERTTrainingHistory(train_losses=[], val_losses=[])
        self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, num_labels: int):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        ).to(self.device)

    def _collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = np.asarray([item["labels"] for item in batch], dtype=np.float32)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encoded

    def _compute_pos_weight(self, labels: np.ndarray) -> torch.Tensor:
        positives = labels.sum(axis=0)
        negatives = labels.shape[0] - positives
        weights = np.ones_like(positives, dtype=np.float32)
        mask = positives > 0
        weights[mask] = np.sqrt(negatives[mask] / positives[mask])
        weights = np.clip(weights, 1.0, self.max_pos_weight)
        return torch.tensor(weights, dtype=torch.float32)

    def fit(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str] | None = None,
        val_labels: np.ndarray | None = None,
    ):
        self._build_model(train_labels.shape[1])
        train_dataset = _FunctionTextDataset(train_texts, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

        pos_weight = self._compute_pos_weight(train_labels).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print(f"[train] Using capped sqrt pos_weight: {pos_weight.detach().cpu().tolist()}")

        val_loader = None
        if val_texts is not None and val_labels is not None and len(val_texts) > 0:
            val_dataset = _FunctionTextDataset(val_texts, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )

        best_state = None
        best_val_loss = None
        self.history = CodeBERTTrainingHistory(train_losses=[], val_losses=[])

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_examples = 0
            for batch in train_loader:
                labels = batch.pop("labels").to(self.device)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = loss_fn(outputs.logits, labels)
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
                outputs = self.model(**batch)
                loss = loss_fn(outputs.logits, labels)
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                total_examples += batch_size
        return running_loss / max(total_examples, 1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None
        dataset = _FunctionTextDataset(texts, np.zeros((len(texts), self.model.config.num_labels), dtype=np.float32))
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
                logits = self.model(**batch).logits
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        if not outputs:
            return np.zeros((0, self.model.config.num_labels), dtype=np.float32)
        return np.vstack(outputs).astype(np.float32)

    def save_model(self, output_dir: str):
        assert self.model is not None
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

