#!/usr/bin/env python3
"""
Lightweight GNN baseline for function-level multilabel classification.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,.:=+\-*/%<>]"
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_REPEAT_EDGES = 64


def _split_function_lines(function_code: str, max_nodes: int) -> list[str]:
    lines = [line.strip() for line in function_code.splitlines() if line.strip()]
    if not lines:
        lines = [function_code.strip() or "empty_function"]
    return lines[:max_nodes]


def _tokens_for_line(line: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(line)
    return tokens if tokens else ["<empty>"]


def _hash_feature_index(token: str, feature_dim: int) -> int:
    return hash(token) % feature_dim


def _build_line_graph(
    function_code: str,
    *,
    max_nodes: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = _split_function_lines(function_code, max_nodes=max_nodes)
    node_count = len(lines)
    x = np.zeros((max_nodes, feature_dim), dtype=np.float32)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros(max_nodes, dtype=np.float32)
    identifier_to_nodes: dict[str, list[int]] = {}

    for idx, line in enumerate(lines):
        mask[idx] = 1.0
        tokens = _tokens_for_line(line)
        for token in tokens:
            x[idx, _hash_feature_index(token, feature_dim)] += 1.0
            if _IDENTIFIER_PATTERN.match(token):
                identifier_to_nodes.setdefault(token, []).append(idx)

        # Encode a few structural cues directly into the node features.
        if idx == 0:
            x[idx, _hash_feature_index("<first_line>", feature_dim)] += 1.0
        if idx == node_count - 1:
            x[idx, _hash_feature_index("<last_line>", feature_dim)] += 1.0
        if "if" in tokens or "require" in tokens or "assert" in tokens:
            x[idx, _hash_feature_index("<branch>", feature_dim)] += 1.0
        if "for" in tokens or "while" in tokens:
            x[idx, _hash_feature_index("<loop>", feature_dim)] += 1.0
        if "return" in tokens:
            x[idx, _hash_feature_index("<return>", feature_dim)] += 1.0
        if "call" in tokens or "delegatecall" in tokens or "send" in tokens:
            x[idx, _hash_feature_index("<external_call>", feature_dim)] += 1.0

        if idx < node_count - 1:
            adjacency[idx, idx + 1] = 1.0
            adjacency[idx + 1, idx] = 1.0

    added_repeat_edges = 0
    for nodes in identifier_to_nodes.values():
        if len(nodes) < 2:
            continue
        for left, right in zip(nodes[:-1], nodes[1:]):
            if left == right:
                continue
            adjacency[left, right] = 1.0
            adjacency[right, left] = 1.0
            added_repeat_edges += 1
            if added_repeat_edges >= _MAX_REPEAT_EDGES:
                break
        if added_repeat_edges >= _MAX_REPEAT_EDGES:
            break

    row_sums = x.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(-1) > 0
    x[nonzero_rows] /= row_sums[nonzero_rows]
    return x, adjacency, mask


class _FunctionGraphDataset(Dataset):
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
class GNNTrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        return torch.bmm(normalized_adjacency, support)


class FunctionGNNMultilabelBaseline:
    def __init__(
        self,
        *,
        max_nodes: int = 48,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 3,
        max_pos_weight: float = 8.0,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
        seed: int = 42,
    ):
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_pos_weight = max_pos_weight
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: nn.Module | None = None
        self.history = GNNTrainingHistory(train_losses=[], val_losses=[])
        self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, num_labels: int):
        layers: list[nn.Module] = []
        input_dim = self.feature_dim
        for _ in range(self.num_layers):
            layers.append(_GCNLayer(input_dim, self.hidden_dim))
            input_dim = self.hidden_dim
        self.model = _FunctionGNNClassifier(
            layers=layers,
            hidden_dim=self.hidden_dim,
            num_labels=num_labels,
            dropout=self.dropout,
        ).to(self.device)

    def _collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        features = np.zeros((batch_size, self.max_nodes, self.feature_dim), dtype=np.float32)
        adjacency = np.zeros((batch_size, self.max_nodes, self.max_nodes), dtype=np.float32)
        masks = np.zeros((batch_size, self.max_nodes), dtype=np.float32)
        labels = np.asarray([item["labels"] for item in batch], dtype=np.float32)

        for idx, item in enumerate(batch):
            x, adj, mask = _build_line_graph(
                item["text"],
                max_nodes=self.max_nodes,
                feature_dim=self.feature_dim,
            )
            features[idx] = x
            adjacency[idx] = adj
            masks[idx] = mask

        for idx in range(batch_size):
            active_nodes = int(masks[idx].sum())
            if active_nodes == 0:
                continue
            adjacency[idx, :active_nodes, :active_nodes] += np.eye(active_nodes, dtype=np.float32)

        adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
        degree = adjacency_tensor.sum(dim=-1)
        degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1.0), -0.5)
        normalized_adjacency = (
            adjacency_tensor
            * degree_inv_sqrt.unsqueeze(-1)
            * degree_inv_sqrt.unsqueeze(-2)
        )

        return {
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

    def fit(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str] | None = None,
        val_labels: np.ndarray | None = None,
    ):
        self._build_model(train_labels.shape[1])
        train_dataset = _FunctionGraphDataset(train_texts, train_labels)
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
            val_dataset = _FunctionGraphDataset(val_texts, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )

        best_state = None
        best_val_loss = None
        self.history = GNNTrainingHistory(train_losses=[], val_losses=[])

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

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None
        dataset = _FunctionGraphDataset(
            texts,
            np.zeros((len(texts), self.model.num_labels), dtype=np.float32),
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


class _FunctionGNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        layers: list[nn.Module],
        hidden_dim: int,
        num_labels: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.num_labels = num_labels
        self.output = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        mask_expanded = mask.unsqueeze(-1)
        pooled_mean = (h * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        masked_h = h.masked_fill(mask_expanded == 0, float("-inf"))
        pooled_max = masked_h.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        graph_embedding = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.output(graph_embedding)


@dataclass
class _SourceSpan:
    start_offset: int
    length: int
    start_line: int
    end_line: int


@dataclass
class _ContractArtifacts:
    source: str
    line_offsets: list[int]
    ast_root: dict | None
    slither_instance: object | None


class _ASTCFGGraphBuilder:
    def __init__(self, *, max_nodes: int, feature_dim: int):
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self._contract_cache: dict[str, _ContractArtifacts] = {}
        self._graph_cache: dict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def build_graph(self, record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contract_file = record.get("contract_file", "")
        if not contract_file:
            return _build_line_graph(
                record.get("function_code", ""),
                max_nodes=self.max_nodes,
                feature_dim=self.feature_dim,
            )
        cache_key = (
            contract_file,
            record.get("function_signature", "") or record.get("function_name", ""),
            int(record.get("start_line", 0) or 0),
            int(record.get("end_line", 0) or 0),
        )
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            artifacts = self._load_contract_artifacts(contract_file)
            ast_nodes, ast_edges = self._extract_ast_subgraph(artifacts, record)
            cfg_nodes, cfg_edges = self._extract_cfg_subgraph(artifacts, record)
            if not ast_nodes and not cfg_nodes:
                graph = _build_line_graph(
                    record.get("function_code", ""),
                    max_nodes=self.max_nodes,
                    feature_dim=self.feature_dim,
                )
                self._graph_cache[cache_key] = graph
                return graph
            graph = self._fuse_graph(artifacts, ast_nodes, ast_edges, cfg_nodes, cfg_edges)
            self._graph_cache[cache_key] = graph
            return graph
        except Exception:
            graph = _build_line_graph(
                record.get("function_code", ""),
                max_nodes=self.max_nodes,
                feature_dim=self.feature_dim,
            )
            self._graph_cache[cache_key] = graph
            return graph

    def _load_contract_artifacts(self, contract_file: str) -> _ContractArtifacts:
        if contract_file in self._contract_cache:
            return self._contract_cache[contract_file]

        source = Path(contract_file).read_text(encoding="utf-8", errors="replace")
        line_offsets = self._build_line_offsets(source)
        ast_root = self._load_solc_ast(contract_file)
        slither_instance = self._load_slither(contract_file)
        artifacts = _ContractArtifacts(
            source=source,
            line_offsets=line_offsets,
            ast_root=ast_root,
            slither_instance=slither_instance,
        )
        self._contract_cache[contract_file] = artifacts
        return artifacts

    @staticmethod
    def _build_line_offsets(source: str) -> list[int]:
        offsets = [0]
        for idx, char in enumerate(source):
            if char == "\n":
                offsets.append(idx + 1)
        return offsets

    def _offset_to_line(self, offsets: list[int], offset: int) -> int:
        return bisect_right(offsets, max(offset, 0))

    def _parse_src_span(self, artifacts: _ContractArtifacts, src: str | None) -> _SourceSpan | None:
        if not src:
            return None
        parts = src.split(":")
        if len(parts) < 2:
            return None
        try:
            start = int(parts[0])
            length = int(parts[1])
        except ValueError:
            return None
        end_offset = start + max(length - 1, 0)
        return _SourceSpan(
            start_offset=start,
            length=length,
            start_line=self._offset_to_line(artifacts.line_offsets, start),
            end_line=self._offset_to_line(artifacts.line_offsets, end_offset),
        )

    def _load_solc_ast(self, contract_file: str) -> dict | None:
        command = ["solc", "--ast-compact-json", contract_file]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            return None
        output = proc.stdout.strip()
        json_start = output.find("{")
        if json_start < 0:
            return None
        payload = json.loads(output[json_start:])
        if "sources" in payload:
            first_source = next(iter(payload["sources"].values()), None)
            if isinstance(first_source, dict):
                return first_source.get("AST")
        return payload

    def _load_slither(self, contract_file: str):
        try:
            from slither import Slither
        except ImportError:
            return None
        try:
            return Slither(contract_file)
        except Exception:
            return None

    def _extract_ast_subgraph(
        self,
        artifacts: _ContractArtifacts,
        record: dict,
    ) -> tuple[list[dict], list[tuple[int, int]]]:
        root = artifacts.ast_root
        if not root:
            return [], []

        target = self._find_ast_function_node(root, artifacts, record)
        if target is None:
            return [], []

        nodes: list[dict] = []
        edges: list[tuple[int, int]] = []

        def visit(node: dict, parent_idx: int | None = None):
            if len(nodes) >= self.max_nodes:
                return
            node_idx = len(nodes)
            span = self._parse_src_span(artifacts, node.get("src"))
            nodes.append(
                {
                    "kind": f"AST::{node.get('nodeType', 'Unknown')}",
                    "text": self._snippet_for_span(artifacts, span),
                    "start_line": span.start_line if span else 0,
                    "end_line": span.end_line if span else 0,
                }
            )
            if parent_idx is not None:
                edges.append((parent_idx, node_idx))
                edges.append((node_idx, parent_idx))
            for child in self._iter_ast_children(node):
                visit(child, node_idx)

        visit(target)
        return nodes, edges

    def _find_ast_function_node(self, root: dict, artifacts: _ContractArtifacts, record: dict) -> dict | None:
        function_name = record.get("function_name", "")
        target_start = int(record.get("start_line", 0) or 0)
        target_end = int(record.get("end_line", 0) or 0)
        best_match = None
        best_score = None

        def visit(node: dict):
            nonlocal best_match, best_score
            if not isinstance(node, dict):
                return
            if node.get("nodeType") == "FunctionDefinition":
                name = node.get("name") or self._fallback_function_name(node)
                span = self._parse_src_span(artifacts, node.get("src"))
                if span:
                    overlap = min(span.end_line, target_end) - max(span.start_line, target_start)
                    if name == function_name and overlap >= -1:
                        score = abs(span.start_line - target_start) + abs(span.end_line - target_end)
                        if best_score is None or score < best_score:
                            best_match = node
                            best_score = score
            for child in self._iter_ast_children(node):
                visit(child)

        visit(root)
        return best_match

    @staticmethod
    def _fallback_function_name(node: dict) -> str:
        kind = node.get("kind", "")
        if kind == "constructor":
            return "constructor"
        if kind == "receive":
            return "receive"
        if kind == "fallback":
            return "fallback"
        return ""

    @staticmethod
    def _iter_ast_children(node: dict):
        for value in node.values():
            if isinstance(value, dict) and "nodeType" in value:
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "nodeType" in item:
                        yield item

    def _snippet_for_span(self, artifacts: _ContractArtifacts, span: _SourceSpan | None) -> str:
        if span is None or span.length <= 0:
            return ""
        end_offset = min(len(artifacts.source), span.start_offset + span.length)
        return artifacts.source[span.start_offset:end_offset][:240]

    def _extract_cfg_subgraph(
        self,
        artifacts: _ContractArtifacts,
        record: dict,
    ) -> tuple[list[dict], list[tuple[int, int]]]:
        slither_instance = artifacts.slither_instance
        if slither_instance is None:
            return [], []

        target_function = self._find_slither_function(slither_instance, record)
        if target_function is None:
            return [], []

        nodes: list[dict] = []
        edges: list[tuple[int, int]] = []
        node_to_idx: dict[object, int] = {}

        for cfg_node in getattr(target_function, "nodes", []) or []:
            if len(nodes) >= self.max_nodes:
                break
            node_to_idx[cfg_node] = len(nodes)
            span = self._source_mapping_to_span(artifacts, getattr(cfg_node, "source_mapping", None))
            expression = getattr(cfg_node, "expression", None)
            node_type = getattr(cfg_node, "type", None)
            nodes.append(
                {
                    "kind": f"CFG::{node_type}",
                    "text": str(expression) if expression is not None else self._snippet_for_span(artifacts, span),
                    "start_line": span.start_line if span else 0,
                    "end_line": span.end_line if span else 0,
                }
            )

        for cfg_node, src_idx in node_to_idx.items():
            for dst in getattr(cfg_node, "sons", []) or []:
                dst_idx = node_to_idx.get(dst)
                if dst_idx is None:
                    continue
                edges.append((src_idx, dst_idx))
                edges.append((dst_idx, src_idx))

        return nodes, edges

    def _find_slither_function(self, slither_instance, record: dict):
        contract_name = record.get("contract_name", "")
        function_name = record.get("function_name", "")
        target_start = int(record.get("start_line", 0) or 0)
        target_end = int(record.get("end_line", 0) or 0)
        best_match = None
        best_score = None

        for contract in getattr(slither_instance, "contracts", []) or []:
            if getattr(contract, "name", "") != contract_name:
                continue
            for function in getattr(contract, "functions", []) or []:
                if getattr(function, "name", "") != function_name:
                    continue
                span = self._source_mapping_to_span(
                    self._contract_cache[record["contract_file"]],
                    getattr(function, "source_mapping", None),
                )
                if span is None:
                    continue
                overlap = min(span.end_line, target_end) - max(span.start_line, target_start)
                if overlap < -1:
                    continue
                score = abs(span.start_line - target_start) + abs(span.end_line - target_end)
                if best_score is None or score < best_score:
                    best_match = function
                    best_score = score
        return best_match

    def _source_mapping_to_span(self, artifacts: _ContractArtifacts, source_mapping) -> _SourceSpan | None:
        if source_mapping is None:
            return None
        lines = source_mapping.get("lines") if isinstance(source_mapping, dict) else getattr(source_mapping, "lines", None)
        if lines:
            return _SourceSpan(
                start_offset=0,
                length=0,
                start_line=lines[0],
                end_line=lines[-1],
            )
        src = source_mapping.get("src") if isinstance(source_mapping, dict) else getattr(source_mapping, "src", None)
        return self._parse_src_span(artifacts, src)

    def _fuse_graph(
        self,
        artifacts: _ContractArtifacts,
        ast_nodes: list[dict],
        ast_edges: list[tuple[int, int]],
        cfg_nodes: list[dict],
        cfg_edges: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        combined_nodes = (ast_nodes + cfg_nodes)[: self.max_nodes]
        ast_count = min(len(ast_nodes), self.max_nodes)
        cfg_offset = ast_count
        x = np.zeros((self.max_nodes, self.feature_dim), dtype=np.float32)
        adjacency = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)

        for idx, node in enumerate(combined_nodes):
            mask[idx] = 1.0
            self._encode_node_features(x[idx], node)

        for src, dst in ast_edges:
            if src < ast_count and dst < ast_count:
                adjacency[src, dst] = 1.0
        for src, dst in cfg_edges:
            src_idx = src + cfg_offset
            dst_idx = dst + cfg_offset
            if src_idx < self.max_nodes and dst_idx < self.max_nodes:
                adjacency[src_idx, dst_idx] = 1.0

        cfg_nodes_limited = cfg_nodes[: max(self.max_nodes - ast_count, 0)]
        for ast_idx, ast_node in enumerate(ast_nodes[:ast_count]):
            for cfg_idx, cfg_node in enumerate(cfg_nodes_limited):
                overlap = min(ast_node["end_line"], cfg_node["end_line"]) - max(ast_node["start_line"], cfg_node["start_line"])
                if overlap >= 0 and ast_node["start_line"] > 0 and cfg_node["start_line"] > 0:
                    fused_cfg_idx = cfg_offset + cfg_idx
                    adjacency[ast_idx, fused_cfg_idx] = 1.0
                    adjacency[fused_cfg_idx, ast_idx] = 1.0

        active_nodes = int(mask.sum())
        if active_nodes > 0:
            adjacency[:active_nodes, :active_nodes] += np.eye(active_nodes, dtype=np.float32)
        row_sums = x.sum(axis=1, keepdims=True)
        nonzero_rows = row_sums.squeeze(-1) > 0
        x[nonzero_rows] /= row_sums[nonzero_rows]
        return x, adjacency, mask

    def _encode_node_features(self, target: np.ndarray, node: dict):
        kind = node.get("kind", "<unknown>")
        target[_hash_feature_index(kind, self.feature_dim)] += 1.0
        if kind.startswith("AST::"):
            target[_hash_feature_index("<ast_node>", self.feature_dim)] += 1.0
        if kind.startswith("CFG::"):
            target[_hash_feature_index("<cfg_node>", self.feature_dim)] += 1.0
        line_span = max(1, node.get("end_line", 0) - node.get("start_line", 0) + 1)
        target[_hash_feature_index(f"<span_{min(line_span, 10)}>", self.feature_dim)] += 1.0
        for token in _tokens_for_line(node.get("text", "")):
            target[_hash_feature_index(token, self.feature_dim)] += 1.0


class _ASTCFGFunctionGraphDataset(Dataset):
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


class ASTCFGFunctionGNNMultilabelBaseline:
    def __init__(
        self,
        *,
        max_nodes: int = 128,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 3,
        max_pos_weight: float = 8.0,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
        seed: int = 42,
    ):
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_pos_weight = max_pos_weight
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: nn.Module | None = None
        self.history = GNNTrainingHistory(train_losses=[], val_losses=[])
        self.graph_builder = _ASTCFGGraphBuilder(max_nodes=max_nodes, feature_dim=feature_dim)
        self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, num_labels: int):
        layers: list[nn.Module] = []
        input_dim = self.feature_dim
        for _ in range(self.num_layers):
            layers.append(_GCNLayer(input_dim, self.hidden_dim))
            input_dim = self.hidden_dim
        self.model = _FunctionGNNClassifier(
            layers=layers,
            hidden_dim=self.hidden_dim,
            num_labels=num_labels,
            dropout=self.dropout,
        ).to(self.device)

    def _collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        features = np.zeros((batch_size, self.max_nodes, self.feature_dim), dtype=np.float32)
        adjacency = np.zeros((batch_size, self.max_nodes, self.max_nodes), dtype=np.float32)
        masks = np.zeros((batch_size, self.max_nodes), dtype=np.float32)
        labels = np.asarray([item["labels"] for item in batch], dtype=np.float32)

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

    def fit(
        self,
        train_records: list[dict],
        train_labels: np.ndarray,
        val_records: list[dict] | None = None,
        val_labels: np.ndarray | None = None,
    ):
        self._build_model(train_labels.shape[1])
        train_dataset = _ASTCFGFunctionGraphDataset(train_records, train_labels)
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
        if val_records is not None and val_labels is not None and len(val_records) > 0:
            val_dataset = _ASTCFGFunctionGraphDataset(val_records, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )

        best_state = None
        best_val_loss = None
        self.history = GNNTrainingHistory(train_losses=[], val_losses=[])

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
        dataset = _ASTCFGFunctionGraphDataset(
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
