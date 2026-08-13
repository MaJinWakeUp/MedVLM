#!/usr/bin/env python3
"""Run the LMOD dual-purpose benchmark on a local GPU.

This is the command-line equivalent of colab_full_dataset_inference.ipynb.  It
indexes LMOD sample folders, runs anatomy and diagnosis inference, writes
resumable checkpoints, computes metrics, and saves plots/CSV/JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


TARGET_DATASETS = ("OIMHS", "REFUGE", "ORIGA", "G1020", "IDRiD")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MH_PROMPT = (
    "This is an ophthalmology OCT image. Based on the image, please tell me "
    "the stage of macular hole decision. Then, give a detailed justification "
    "and explanation for your answer. Follow the format: Stage: <AN INTEGER>; "
    "Justification: <EXPLANATION>."
)
GLAUCOMA_PROMPT = (
    "This is a color fundus image of type Fundus RGB Images. Based on the "
    "image, please tell me whether this image contains glaucoma, then give "
    "detailed justifications. Follow the format: GLAUCOMA / NON-GLAUCOMA; "
    "Explanation: <JUSTIFICATION>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-GPU version of colab_full_dataset_inference.ipynb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument(
        "--tasks", choices=("both", "anatomy", "diagnosis"), default="both"
    )
    parser.add_argument("--max-oct-samples", type=int)
    parser.add_argument("--max-cfp-samples", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="resume from model-specific JSON checkpoints",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help="index/validate extracted data without loading a model",
    )
    return parser.parse_args()


def model_family(model_id: str) -> str:
    name = model_id.lower()
    if "internvl" in name:
        return "internvl"
    if "qwen" in name:
        return "qwen"
    raise ValueError(
        f"Unsupported model family for {model_id!r}. Use an InternVL or Qwen VL model."
    )


def model_tag(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def validate_dataset_folders(data_root: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    for name in TARGET_DATASETS:
        folder = data_root / name
        if folder.is_dir() and any(folder.iterdir()):
            print(f"[data] {name}: {folder}")
        else:
            print(f"[data] {name}: extracted folder not found")


def infer_source(info_path: Path, data_root: Path, dataset_id: str) -> str | None:
    haystack = " ".join([dataset_id, *info_path.relative_to(data_root).parts]).lower()
    for name in TARGET_DATASETS:
        if name.lower() in haystack:
            return name
    return None


def normalize_glaucoma(value: Any) -> str | None:
    """Normalize only explicit labels; do not turn 'Unspecified' into healthy."""
    if value is None:
        return None
    label = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if label in {"glaucoma", "1", "1.0", "true", "yes", "positive"}:
        return "Glaucoma"
    if label in {
        "non glaucoma", "nonglaucoma", "normal", "healthy", "0", "0.0",
        "false", "no", "negative",
    }:
        return "Non-Glaucoma"
    return None


def find_images(sample_dir: Path) -> tuple[Path | None, Path | None]:
    clean = sample_dir / "visualization.png"
    clean_path = clean if clean.is_file() else None
    annotated_path = None
    annotated_dir = sample_dir / "annotated"
    if annotated_dir.is_dir():
        for name in (
            "bbox_annotated.png", "annotated_bounding_box.png", "annotated.png"
        ):
            candidate = annotated_dir / name
            if candidate.is_file():
                annotated_path = candidate
                break
        if annotated_path is None:
            annotated_path = next(iter(sorted(annotated_dir.glob("*.png"))), None)
    return clean_path or annotated_path, annotated_path or clean_path


def index_datasets(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oct_samples: list[dict[str, Any]] = []
    cfp_samples: list[dict[str, Any]] = []
    info_files = []
    for dataset_name in TARGET_DATASETS:
        dataset_dir = data_root / dataset_name
        if dataset_dir.is_dir():
            info_files.extend(dataset_dir.rglob("information.json"))
    info_files.sort()
    print(f"[index] found {len(info_files)} information.json files")
    ignored = 0
    invalid = 0
    for info_path in tqdm(info_files, desc="Indexing", unit="sample"):
        try:
            with info_path.open(encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        metadata = meta.get("metadata") or {}
        source = infer_source(info_path, data_root, str(metadata.get("dataset_id", "")))
        if source is None:
            ignored += 1
            continue
        sample_dir = info_path.parent
        clean_image, annotated_image = find_images(sample_dir)
        if clean_image is None:
            invalid += 1
            continue
        bboxes = (meta.get("annotations") or {}).get("bounding_boxes") or []
        common = {
            "id": sample_dir.name,
            "source_dataset": source,
            "dir": str(sample_dir),
            "info_path": str(info_path),
            "clean_image": str(clean_image),
            "annotated_image": str(annotated_image),
            "bboxes": bboxes,
        }
        image_type = str(meta.get("image_type", "")).lower()
        if source == "OIMHS" or "oct" in image_type:
            stage = metadata.get("stage_of_macular_hole_decision")
            try:
                stage = int(stage) if stage is not None else None
            except (TypeError, ValueError):
                stage = None
            oct_samples.append({**common, "modality": "OCT", "mh_stage": stage})
        else:
            cfp_samples.append(
                {
                    **common,
                    "modality": "CFP",
                    "glaucoma_label": normalize_glaucoma(metadata.get("glaucoma_label")),
                }
            )
    print(f"[index] OCT={len(oct_samples)}, CFP={len(cfp_samples)}")
    print(f"[index] CFP sources: {dict(Counter(s['source_dataset'] for s in cfp_samples))}")
    print(
        "[index] diagnosis labels: "
        f"OCT={sum(s['mh_stage'] in (1, 2, 3, 4) for s in oct_samples)}, "
        f"CFP={sum(s['glaucoma_label'] is not None for s in cfp_samples)}"
    )
    if ignored or invalid:
        print(f"[index] ignored unrelated={ignored}, invalid/unreadable={invalid}")
    return oct_samples, cfp_samples


def load_local_ml_dependencies() -> None:
    global torch, T, InterpolationMode
    import torch as torch_module
    import torchvision.transforms as transforms_module
    from torchvision.transforms.functional import InterpolationMode as interpolation_mode

    torch = torch_module
    T = transforms_module
    InterpolationMode = interpolation_mode


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32" or device.type == "cpu":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )


class VLMEngine:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
        cache_dir: Path | None,
        trust_remote_code: bool,
    ) -> None:
        import transformers

        self.model_id = model_id
        self.family = model_family(model_id)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.tokenizer = None
        common = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir:
            common["cache_dir"] = str(cache_dir)
        print(f"[model] loading {model_id} ({self.family}, {device}, {dtype})")
        if self.family == "internvl":
            auto_model = transformers.AutoModel
            try:
                self.model = auto_model.from_pretrained(
                    model_id, use_flash_attn=True, **common
                )
            except (ImportError, ModuleNotFoundError):
                print("[model] FlashAttention unavailable; using standard attention")
                self.model = auto_model.from_pretrained(
                    model_id, use_flash_attn=False, **common
                )
            self.model = self.model.eval().to(device)
            tokenizer_args = {"trust_remote_code": trust_remote_code, "use_fast": False}
            if cache_dir:
                tokenizer_args["cache_dir"] = str(cache_dir)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, **tokenizer_args
            )
        else:
            model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_class is None:
                model_class = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_class is None:
                raise RuntimeError(
                    "This transformers version has no multimodal auto-model class. "
                    "Upgrade transformers as described in LOCAL_GPU.md."
                )
            # device_map avoids a second full-size copy during placement and can shard
            # Qwen across local GPUs when necessary.
            self.model = model_class.from_pretrained(
                model_id,
                device_map="auto" if device.type == "cuda" else None,
                **common,
            ).eval()
            processor_args = {"trust_remote_code": trust_remote_code}
            if cache_dir:
                processor_args["cache_dir"] = str(cache_dir)
            self.processor = transformers.AutoProcessor.from_pretrained(
                model_id, **processor_args
            )

    def internvl_pixels(self, image: Image.Image) -> torch.Tensor:
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return transform(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def infer(self, image: Image.Image, prompt: str) -> str:
        if self.family == "internvl":
            pixels = self.internvl_pixels(image)
            with torch.inference_mode():
                return str(
                    self.model.chat(
                        self.tokenizer,
                        pixels,
                        f"<image>\n{prompt}",
                        {"max_new_tokens": self.max_new_tokens, "do_sample": False},
                    )
                )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            # Compatibility path for older Qwen2.5-VL processors.
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        input_device = next(self.model.parameters()).device
        inputs = inputs.to(input_device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def build_anatomy_prompt(bboxes: list[dict[str, Any]], modality: str) -> str:
    region_ids = [str(box.get("annotation_ID", i + 1)) for i, box in enumerate(bboxes)]
    if modality == "OCT":
        description, options = "ophthalmology OCT image", "irc, retina, choroid, mh"
    else:
        description, options = (
            "ophthalmology color fundus image",
            "optic disc, optic cup, fovea, disc",
        )
    answer_format = "; ".join(
        f"Region ID: {region_id}; Type: <answer>" for region_id in region_ids
    )
    return (
        f"This is an {description}. Please identify the type of each labeled "
        f"bounding box in this image. Options can be: {options}. Please just "
        f"follow the format: {answer_format}"
    )


def normalize_region(value: str) -> str:
    value = re.sub(r"[^a-z]+", " ", value.lower()).strip()
    aliases = {
        "optic disk": "optic disc",
        "disc": "optic disc",
        "disk": "optic disc",
        "macular hole": "mh",
        "intraretinal cyst": "irc",
        "intraretinal cysts": "irc",
    }
    return aliases.get(value, value)


def parse_anatomy_response(response: str) -> dict[str, str]:
    predictions = {}
    pattern = re.compile(
        r"region\s*id\s*[:#]?\s*(\d+)\s*;?\s*type\s*:\s*"
        r"(.+?)(?=\s*;\s*region\s*id|\n\s*region\s*id|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for region_id, region_type in pattern.findall(response):
        predictions[region_id] = normalize_region(region_type.splitlines()[0])
    return predictions


def parse_mh_stage(response: str) -> int | None:
    match = re.search(r"stage\s*[:\-]?\s*([1-4])\b", response, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b([1-4])\b", response)
    return int(match.group(1)) if match else None


def parse_glaucoma(response: str) -> str:
    text = response.upper()
    if re.search(r"\b(?:NON[- ]?GLAUCOMA|NO GLAUCOMA|NOT GLAUCOMA)\b", text):
        return "Non-Glaucoma"
    if "GLAUCOMA" in text:
        return "Glaucoma"
    return "Unknown"


def load_checkpoint(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError("checkpoint root is not a list")
        print(f"[resume] loaded {len(value)} records from {path}")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot resume from {path}: {exc}") from exc


def append_error(path: Path, task: str, sample: dict[str, Any], exc: Exception) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "sample_id": f"{sample['source_dataset']}_{sample['id']}",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=4),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_anatomy(
    samples: list[dict[str, Any]], engine: VLMEngine, checkpoint: Path,
    error_path: Path, interval: int, resume: bool, model_id: str,
) -> list[dict[str, Any]]:
    results = load_checkpoint(checkpoint, resume)
    processed = {row["sample_id"] for row in results}
    remaining = [
        sample for sample in samples
        if sample["bboxes"]
        and f"{sample['source_dataset']}_{sample['id']}" not in processed
    ]
    print(f"[anatomy] completed={len(processed)}, remaining={len(remaining)}")
    for index, sample in enumerate(tqdm(remaining, desc="Anatomy", unit="image"), 1):
        sample_id = f"{sample['source_dataset']}_{sample['id']}"
        try:
            with Image.open(sample["annotated_image"]) as opened:
                image = opened.convert("RGB")
            response = engine.infer(
                image, build_anatomy_prompt(sample["bboxes"], sample["modality"])
            )
            truth = {
                str(box.get("annotation_ID", i + 1)): normalize_region(
                    str(box.get("region_type", ""))
                )
                for i, box in enumerate(sample["bboxes"])
            }
            predictions = parse_anatomy_response(response)
            correct = sum(predictions.get(key) == value for key, value in truth.items())
            results.append(
                {
                    "sample_id": sample_id,
                    "model_name": model_id,
                    "dataset": sample["source_dataset"],
                    "modality": sample["modality"],
                    "ground_truth": truth,
                    "predictions": predictions,
                    "accuracy": correct / len(truth) if truth else 0.0,
                    "correct_count": correct,
                    "total_count": len(truth),
                    "response": response,
                }
            )
        except Exception as exc:  # keep a full error log and retry on the next run
            append_error(error_path, "anatomy", sample, exc)
            tqdm.write(f"[anatomy] ERROR {sample_id}: {type(exc).__name__}: {exc}")
        if index % interval == 0:
            atomic_json_dump(results, checkpoint)
    atomic_json_dump(results, checkpoint)
    return results


def run_diagnosis(
    samples: list[dict[str, Any]], engine: VLMEngine, checkpoint: Path,
    error_path: Path, interval: int, resume: bool, model_id: str,
) -> list[dict[str, Any]]:
    results = load_checkpoint(checkpoint, resume)
    processed = {row["sample_id"] for row in results}
    eligible = [
        sample for sample in samples
        if (sample["modality"] == "OCT" and sample.get("mh_stage") in (1, 2, 3, 4))
        or (sample["modality"] == "CFP" and sample.get("glaucoma_label") is not None)
    ]
    remaining = [
        sample for sample in eligible
        if f"{sample['source_dataset']}_{sample['id']}" not in processed
    ]
    print(f"[diagnosis] completed={len(processed)}, remaining={len(remaining)}")
    for index, sample in enumerate(tqdm(remaining, desc="Diagnosis", unit="image"), 1):
        sample_id = f"{sample['source_dataset']}_{sample['id']}"
        try:
            with Image.open(sample["clean_image"]) as opened:
                image = opened.convert("RGB")
            if sample["modality"] == "OCT":
                response = engine.infer(image, MH_PROMPT)
                prediction, truth = parse_mh_stage(response), sample["mh_stage"]
                task = "Macular Hole Staging"
            else:
                response = engine.infer(image, GLAUCOMA_PROMPT)
                prediction, truth = parse_glaucoma(response), sample["glaucoma_label"]
                task = "Glaucoma Detection"
            results.append(
                {
                    "sample_id": sample_id,
                    "model_name": model_id,
                    "dataset": sample["source_dataset"],
                    "modality": sample["modality"],
                    "diag_task": task,
                    "ground_truth": truth,
                    "prediction": prediction,
                    "correct": bool(prediction == truth),
                    "response": response,
                }
            )
        except Exception as exc:
            append_error(error_path, "diagnosis", sample, exc)
            tqdm.write(f"[diagnosis] ERROR {sample_id}: {type(exc).__name__}: {exc}")
        if index % interval == 0:
            atomic_json_dump(results, checkpoint)
    atomic_json_dump(results, checkpoint)
    return results


def anatomy_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    true, predicted = [], []
    for row in results:
        if row["modality"] != "OCT":
            continue
        for region_id, label in row["ground_truth"].items():
            true.append(label)
            predicted.append(row["predictions"].get(region_id, "unknown"))
    f1 = (
        precision_recall_fscore_support(
            true, predicted, average="macro", zero_division=0
        )[2]
        if true else None
    )
    return {"true": true, "predicted": predicted, "macro_f1": f1}


def diagnosis_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    oct_rows = [
        row for row in results
        if row["modality"] == "OCT" and row["prediction"] in (1, 2, 3, 4)
    ]
    cfp_rows = [
        row for row in results
        if row["modality"] == "CFP"
        and row["prediction"] in ("Glaucoma", "Non-Glaucoma")
    ]
    metrics: dict[str, Any] = {"oct_rows": oct_rows, "cfp_rows": cfp_rows}
    if oct_rows:
        true = [int(row["ground_truth"]) for row in oct_rows]
        pred = [int(row["prediction"]) for row in oct_rows]
        metrics["oct"] = {
            "true": true,
            "predicted": pred,
            "accuracy": accuracy_score(true, pred),
            "qwk": cohen_kappa_score(true, pred, weights="quadratic"),
            "mae": float(np.mean(np.abs(np.asarray(true) - np.asarray(pred)))),
        }
    if cfp_rows:
        true = [row["ground_truth"] for row in cfp_rows]
        pred = [row["prediction"] for row in cfp_rows]
        matrix = confusion_matrix(true, pred, labels=["Glaucoma", "Non-Glaucoma"])
        tp, fn, fp, tn = matrix.ravel()
        metrics["cfp"] = {
            "true": true,
            "predicted": pred,
            "accuracy": accuracy_score(true, pred),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
        }
    return metrics


def save_outputs(
    results_dir: Path, tag: str, model_id: str,
    task1: list[dict[str, Any]], task2: list[dict[str, Any]],
) -> None:
    anatomy = anatomy_metrics(task1)
    diagnosis = diagnosis_metrics(task2)
    pd.DataFrame(task1).to_csv(
        results_dir / f"anatomical_recognition_{tag}_predictions.csv", index=False
    )
    pd.DataFrame(task2).to_csv(
        results_dir / f"diagnosis_{tag}_predictions.csv", index=False
    )
    oct_metrics = diagnosis.get("oct", {})
    cfp_metrics = diagnosis.get("cfp", {})
    summary = {
        "model_name": model_id,
        "model_tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {"anatomy": len(task1), "diagnosis": len(task2)},
        "oct_oimhs": {
            "anatomical_recognition_macro_f1": anatomy["macro_f1"],
            "mh_staging_accuracy": oct_metrics.get("accuracy"),
            "mh_staging_qwk": oct_metrics.get("qwk"),
            "mh_staging_mae": oct_metrics.get("mae"),
        },
        "cfp_glaucoma": {
            "combined_accuracy": cfp_metrics.get("accuracy"),
            "combined_sensitivity": cfp_metrics.get("sensitivity"),
            "combined_specificity": cfp_metrics.get("specificity"),
        },
    }
    atomic_json_dump(summary, results_dir / f"evaluation_summary_{tag}_report.json")
    print(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(1, 3, figsize=(22, 6))
    titles = ["OCT Anatomy", "OCT Macular Hole Stage", "CFP Glaucoma"]
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_axis_off()
    if anatomy["true"]:
        labels = ["irc", "retina", "choroid", "mh"]
        matrix = confusion_matrix(anatomy["true"], anatomy["predicted"], labels=labels)
        axes[0].set_axis_on()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"OCT Anatomy (macro F1={anatomy['macro_f1']:.3f})")
    if oct_metrics:
        matrix = confusion_matrix(
            oct_metrics["true"], oct_metrics["predicted"], labels=[1, 2, 3, 4]
        )
        axes[1].set_axis_on()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", cbar=False,
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4], ax=axes[1])
        axes[1].set_title(f"OCT MH Stage (accuracy={oct_metrics['accuracy']:.1%})")
    if cfp_metrics:
        labels = ["Glaucoma", "Non-Glaucoma"]
        matrix = confusion_matrix(
            cfp_metrics["true"], cfp_metrics["predicted"], labels=labels
        )
        axes[2].set_axis_on()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=axes[2])
        axes[2].set_title(f"CFP Glaucoma (accuracy={cfp_metrics['accuracy']:.1%})")
    figure.suptitle(f"LMOD local benchmark — {model_id}")
    figure.tight_layout()
    figure.savefig(results_dir / f"confusion_matrices_{tag}.png", dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1")
    data_root = args.data_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] data={data_root}")
    print(f"[paths] results={results_dir}")
    validate_dataset_folders(data_root)
    oct_samples, cfp_samples = index_datasets(data_root)
    if not oct_samples and not cfp_samples:
        raise RuntimeError(
            f"No LMOD samples found under {data_root}. See LOCAL_GPU.md for the layout."
        )
    oct_samples = oct_samples[: args.max_oct_samples] if args.max_oct_samples is not None else oct_samples
    cfp_samples = cfp_samples[: args.max_cfp_samples] if args.max_cfp_samples is not None else cfp_samples
    print(f"[run] selected OCT={len(oct_samples)}, CFP={len(cfp_samples)}")
    if args.index_only:
        print("[run] index-only validation complete; model was not loaded")
        return 0

    load_local_ml_dependencies()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[gpu] torch={torch.__version__}, device={device}, dtype={dtype}")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"[gpu] {props.name}, total VRAM={props.total_memory / 2**30:.1f} GiB")
    else:
        print("[gpu] WARNING: running these models on CPU will be very slow")

    engine = VLMEngine(
        args.model, device, dtype, args.max_new_tokens,
        args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        args.trust_remote_code,
    )
    tag = model_tag(args.model)
    error_path = results_dir / f"errors_{tag}.jsonl"
    task1_path = results_dir / f"task1_anatomy_{tag}_results.json"
    task2_path = results_dir / f"task2_diagnosis_{tag}_results.json"
    selected = oct_samples + cfp_samples
    task1 = (
        run_anatomy(
            selected, engine, task1_path, error_path, args.checkpoint_interval,
            args.resume, args.model,
        )
        if args.tasks in ("both", "anatomy")
        else load_checkpoint(task1_path, True)
    )
    task2 = (
        run_diagnosis(
            selected, engine, task2_path, error_path, args.checkpoint_interval,
            args.resume, args.model,
        )
        if args.tasks in ("both", "diagnosis")
        else load_checkpoint(task2_path, True)
    )
    save_outputs(results_dir, tag, args.model, task1, task2)
    print(f"[done] results saved to {results_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. The latest periodic checkpoint can be resumed.", file=sys.stderr)
        raise SystemExit(130)
