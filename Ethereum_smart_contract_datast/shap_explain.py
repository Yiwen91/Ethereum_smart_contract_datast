#!/usr/bin/env python3
"""
SHAP-based interpretability for experiment models.

Supports:
- TF-IDF tabular models
- CodeBERT text models
- Hybrid CodeBERT + GNN models

Designed for thesis-quality vulnerability explanation reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


from experiment_utils import VULN_TYPES


try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False


try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    plt = None
    _MPL_AVAILABLE = False



# ============================================================
# SHAP REQUIREMENT
# ============================================================

def require_shap():

    if not _SHAP_AVAILABLE:
        raise ImportError(
            "Install SHAP first:\n"
            "pip install shap matplotlib"
        )



# ============================================================
# TOKEN PROCESSING
# ============================================================


def _clean_token(token: str) -> str:
    """
    Convert CodeBERT tokens into readable source code tokens.
    """

    token = (
        token
        .replace("Ġ", "")
        .replace("Ċ", "")
        .replace("ĉ", "")
        .strip()
    )

    return token

# Tokens that are not useful for thesis interpretation.
# They are ignored only in visualization.
_DISPLAY_IGNORE = {

    "_",
    "__",

    "a",
    "b",
    "c",
    "i",
    "j",

    "0",
    "1",
    "2",
    "32",
    "64",
    "128",
    "256",

    "[_",
    "(_",
    "]",
    "[",
    "(",
    ")",
    "{",
    "}",

    "ID",
    "Name",
    "Names",

    "true",
    "false",

    "external",
    "public",
    "private",
    "internal",
}

_PUNCT_ONLY = set(
    "()[]{}.,;:+-*/%=<>!&|^~?@#\"'`\\ "
)

def _is_meaningful_token(token):

    if not token:
        return False


    if len(token) <= 1:
        return False


    if all(
        ch in _PUNCT_ONLY
        for ch in token
    ):
        return False


    bad_tokens = {
        "true",
        "false",
        "null",
        "memory",
        "public",
        "external",
        "internal",
        "private",
    }


    if token in bad_tokens:
        return False


    return True

# ============================================================
# BPE MERGING
# ============================================================


def merge_bpe_tokens(tokens):
    merged = []
    current = ""

    for token in tokens:

        if token in ("<s>", "</s>", "<pad>"):
            continue

        token = token.replace("Ċ", "")

        if token.startswith("Ġ"):
            if current:
                merged.append(current)
            current = token[1:]

        else:
            current += token

    if current:
        merged.append(current)

    return merged

# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class TokenAttribution:

    token: str
    shap_value: float



@dataclass
class FeatureAttribution:

    feature: str
    shap_value: float



@dataclass
class SampleShapExplanation:

    sample_index: int

    contract_file: str

    function_name: str

    label: str

    predicted_probability: float

    base_value: float

    attributions: list[
        TokenAttribution |
        FeatureAttribution
    ]

    explanation_type: str

    function_code_preview: str = ""



@dataclass
class ShapRunSummary:

    model: str

    run_dir: str

    label: str

    num_samples: int

    num_explained: int

    background_samples: int

    max_evals: int | None

    samples: list[SampleShapExplanation] = field(
        default_factory=list
    )

    top_global_tokens: list[dict[str,Any]] = field(
        default_factory=list
    )



# ============================================================
# ATTRIBUTION PROCESSING
# ============================================================


def _top_attributions(
    items,
    limit:int = 15
):

    ranked = sorted(
        items,
        key=lambda x: abs(x.shap_value),
        reverse=True
    )[:limit]


    output = []


    for item in ranked:

        if isinstance(item, TokenAttribution):

            output.append(
                {
                    "token":item.token,
                    "shap_value":float(item.shap_value)
                }
            )

        else:

            output.append(
                {
                    "feature":item.feature,
                    "shap_value":float(item.shap_value)
                }
            )


    return output



def _split_attributions(
    items,
    limit:int = 10
):

    positive = [
        x for x in items
        if x.shap_value > 0
    ]


    negative = [
        x for x in items
        if x.shap_value < 0
    ]


    positive = sorted(
        positive,
        key=lambda x:x.shap_value,
        reverse=True
    )[:limit]


    negative = sorted(
        negative,
        key=lambda x:x.shap_value
    )[:limit]



    def convert(item):

        if isinstance(item,TokenAttribution):

            return {
                "token":item.token,
                "shap_value":float(item.shap_value)
            }


        return {
            "feature":item.feature,
            "shap_value":float(item.shap_value)
        }


    return (
        [convert(x) for x in positive],
        [convert(x) for x in negative]
    )



# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================


def _global_token_importance(samples, limit=20):

    scores = {}
    counts = {}

    for sample in samples:

        for item in sample.attributions:

            token = item.token

            if token in _DISPLAY_IGNORE:
                continue

            scores[token] = scores.get(token, 0) + abs(item.shap_value)
            counts[token] = counts.get(token, 0) + 1

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1] / counts[x[0]],
        reverse=True,
    )

    output = []

    for token, score in ranked[:limit]:

        output.append({
            "token": token,
            "mean_abs_shap": score / counts[token],
            "count": counts[token],
        })

    return output


# ============================================================
# TEXT MODEL SHAP EXPLANATION
# ============================================================

def explain_tabular_label(
    model,
    *,
    texts: list[str],
    label: str,
    label_index: int,
    background_texts: list[str],
    max_evals: int | None = 500,
) -> list[SampleShapExplanation]:

    require_shap()

    label_model = model.models[label_index]

    background = model.transform(background_texts)

    if background.shape[0] == 0:
        raise ValueError(
            "Background samples are empty for tabular SHAP."
        )

    def predict_fn(x):

        if hasattr(x, "toarray"):
            x = x.toarray()

        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        return label_model.predict_proba(x)[:, 1]


    explainer = shap.Explainer(
        predict_fn,
        background,
        max_evals=max_evals,
    )


    features = model.transform(texts)

    shap_values = explainer(features)


    values = np.asarray(shap_values.values)

    base_values = np.asarray(
        shap_values.base_values
    )


    if values.ndim == 3:
        values = values[:, :, 1]


    feature_names = np.asarray(
        model.vectorizer.get_feature_names_out()
    )


    probs = model.predict_proba(texts)[:, label_index]


    explanations = []


    for idx, text in enumerate(texts):

        row = values[idx]


        top_idx = np.argsort(
            np.abs(row)
        )[::-1][:20]


        attrs = []

        for i in top_idx:

            if abs(row[i]) < 1e-8:
                continue

            attrs.append(
                FeatureAttribution(
                    feature=str(feature_names[i]),
                    shap_value=float(row[i]),
                )
            )


        explanations.append(
            SampleShapExplanation(
                sample_index=idx,
                contract_file="",
                function_name="",
                label=label,
                predicted_probability=float(probs[idx]),
                base_value=float(
                    base_values[idx]
                    if np.ndim(base_values)
                    else base_values
                ),
                attributions=attrs,
                explanation_type="tfidf_features",
                function_code_preview=text[:500],
            )
        )


    return explanations


def explain_text_model_label(
    *,
    predict_label_proba: Callable[[list[str]], np.ndarray],
    tokenizer,
    texts:list[str],
    records:list[dict],
    label:str,
    label_index:int,
    background_texts:list[str],
    max_evals:int | None = 300,
):

    require_shap()


    masker = shap.maskers.Text(
        tokenizer,
        mask_token="..."
    )



    def predict_for_shap(masked_texts):

        probs = predict_label_proba(
            list(masked_texts)
        )

        return np.asarray(
            probs,
            dtype=np.float32
        )



    explainer = shap.Explainer(
        predict_for_shap,
        masker
    )


    shap_values = explainer(
        texts,
        max_evals=max_evals
    )



    values = np.asarray(
        shap_values.values
    )


    base_values = np.asarray(
        shap_values.base_values
    )



    probs = predict_label_proba(texts)



    explanations = []



    for idx,text in enumerate(texts):


        # SHAP output shape handling
        if values.ndim == 3:

            row_values = values[idx,:,0]

        else:

            row_values = values[idx]



        encoded = tokenizer(
            text,
            truncation=True,
            max_length=getattr(
                tokenizer,
                "model_max_length",
                512
            ),
            return_tensors=None
        )


        raw_tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )



        tokens = merge_bpe_tokens(
            raw_tokens
        )



        # ------------------------------------------------
        # Aggregate subword SHAP values
        # ------------------------------------------------

        aggregated = {}

        token_count = min(
            len(tokens),
            len(row_values)
        )



        for token_index in range(token_count):


            token = _clean_token(
                tokens[token_index]
            )


            if not _is_meaningful_token(token):
                continue



            shap_value = float(
                row_values[token_index]
            )



            if token in aggregated:

                aggregated[token] += shap_value

            else:

                aggregated[token] = shap_value



        token_attrs = []


        for token,value in aggregated.items():

            token_attrs.append(
                TokenAttribution(
                    token=token,
                    shap_value=value
                )
            )



        token_attrs = sorted(
            token_attrs,
            key=lambda x:
                abs(x.shap_value),
            reverse=True
        )[:30]



        record = (
            records[idx]
            if idx < len(records)
            else {}
        )



        explanations.append(

            SampleShapExplanation(

                sample_index=idx,

                contract_file=str(
                    record.get(
                        "contract_file",
                        ""
                    )
                ),

                function_name=str(
                    record.get(
                        "function_name",
                        ""
                    )
                ),

                label=label,


                predicted_probability=float(
                    probs[idx]
                ),


                base_value=float(
                    base_values[idx]
                )
                if np.ndim(base_values)
                else float(base_values),


                attributions=token_attrs,


                explanation_type=
                    "text_tokens",


                function_code_preview=
                    text[:500]
            )
        )



    return explanations
# ============================================================
# HYBRID MODEL SHAP
# ============================================================


def build_hybrid_text_predict_fn(
    hybrid_model,
    record: dict,
    label_index: int
):

    import torch


    assert hybrid_model.model is not None


    base_batch = hybrid_model._collate(
        [{
            "record": record,
            "labels": np.zeros(
                (1, hybrid_model.model.num_labels),
                dtype=np.float32
            )
        }]
    )


    def predict_logit(masked_texts):

        encoded = hybrid_model.tokenizer(
            list(masked_texts),
            truncation=True,
            padding=True,
            max_length=hybrid_model.max_length,
            return_tensors="pt",
        )


        batch_size = len(masked_texts)


        batch = {
            "input_ids":
                encoded["input_ids"]
                .to(hybrid_model.device),

            "attention_mask":
                encoded["attention_mask"]
                .to(hybrid_model.device),

            "x":
                base_batch["x"]
                .repeat(batch_size,1,1)
                .to(hybrid_model.device),

            "adj":
                base_batch["adj"]
                .repeat(batch_size,1,1)
                .to(hybrid_model.device),

            "mask":
                base_batch["mask"]
                .repeat(batch_size,1)
                .to(hybrid_model.device),
        }


        if "cross_contract" in base_batch:

            batch["cross_contract"] = (
                base_batch["cross_contract"]
                .repeat(batch_size,1)
                .to(hybrid_model.device)
            )


        hybrid_model.model.eval()


        with torch.no_grad():

            logits = hybrid_model.model(**batch)


            value = logits[:, label_index]


        return value.cpu().numpy()



    def predict_probability(masked_texts):

        logits = predict_logit(masked_texts)

        return (
            1 /
            (1 + np.exp(-logits))
        )


    return predict_logit, predict_probability


# ============================================================
# SAVE REPORT
# ============================================================


def save_shap_summary(
    summary:ShapRunSummary,
    output_dir:Path
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    payload={

        "model":
            summary.model,

        "run_dir":
            summary.run_dir,

        "label":
            summary.label,

        "num_samples":
            summary.num_samples,

        "num_explained":
            summary.num_explained,

        "background_samples":
            summary.background_samples,

        "max_evals":
            summary.max_evals,


        "top_global_tokens":
            summary.top_global_tokens,


        "samples":[

            {
                **asdict(sample),

                "top_attributions":
                    _top_attributions(
                        sample.attributions
                    )

            }

            for sample in summary.samples
        ]
    }



    json_path = (
        output_dir /
        "shap_summary.json"
    )


    json_path.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )



    lines=[]


    lines.append(
        "# SHAP Explanation Report"
    )

    lines.append("")


    lines.append(
        f"Model: {summary.model}"
    )


    lines.append(
        f"Target vulnerability: {summary.label}"
    )


    lines.append(
        f"Samples explained: "
        f"{summary.num_explained}"
        f"/{summary.num_samples}"
    )


    lines.append("")


    lines.append(
        "## Global Feature Importance"
    )


    lines.append("")



    for item in summary.top_global_tokens[:20]:

        lines.append(

            f"- **{item['token']}** "
            f"| mean SHAP="
            f"{item['mean_abs_shap']:.6f} "
            f"(n={item['count']})"
        )

    for sample in summary.samples:

        lines.extend(

            [

            "",

            "---",

            "",

            f"## Function: "
            f"{sample.function_name}",


            f"P({sample.label}) = {sample.predicted_probability:.4f}",  

            "",

            "### Increasing vulnerability"
            ]
        )

        positive,negative = (
            _split_attributions(
                sample.attributions
            )
        )

        for item in positive:

            name = (
                item.get("token")
                or
                item.get("feature")
            )

            lines.append(
                f"+ {name}: "
                f"{item['shap_value']:+.6f}"
            )

        lines.append("")

        lines.append(
            "### Reducing vulnerability"
        )

        for item in negative:

            name = (
                item.get("token")
                or
                item.get("feature")
            )

            lines.append(
                f"- {name}: "
                f"{item['shap_value']:+.6f}"
            )



    report_path = (
        output_dir /
        "shap_report.md"
    )


    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


    return json_path



# ============================================================
# PLOT
# ============================================================


def plot_sample_bar(
    sample:SampleShapExplanation,
    output_path:Path,
    limit:int=12
):

    if not _MPL_AVAILABLE:
        return



    top=_top_attributions(
        sample.attributions,
        limit
    )


    if not top:
        return



    labels=[

        x.get("token")
        or
        x.get("feature")

        for x in top

    ]


    values=[

        x["shap_value"]

        for x in top

    ]



    labels=labels[::-1]

    values=values[::-1]



    fig,ax=plt.subplots(

        figsize=(
            8,
            max(
                4,
                len(labels)*0.35
            )
        )

    )



    colors=[

        "#d62728"
        if x>0
        else
        "#1f77b4"

        for x in values

    ]



    ax.barh(
        labels,
        values,
        color=colors
    )


    ax.axvline(
        0,
        color="black",
        linewidth=0.8
    )


    ax.set_xlabel(
        "SHAP contribution"
    )


    ax.set_title(
        f"{sample.function_name} - "
        f"{sample.label}"
    )


    fig.tight_layout()


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)
