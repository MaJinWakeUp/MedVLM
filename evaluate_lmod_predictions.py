#!/usr/bin/env python3
"""Evaluate saved LMOD recognition and diagnosis predictions.

Primary metrics:
  * recognition: precision, recall, F1, hallucination resistance (HR)
  * diagnosis: glaucoma accuracy and macular-hole staging accuracy
  * invalid-response rate for each task

Reference labels are rebuilt from the extracted LMOD folders rather than
trusted from prediction files. Invalid or missing outputs count as incorrect
for recall/diagnosis accuracy and are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from local_full_dataset_inference import (
    index_datasets,
    normalize_region,
    validate_dataset_folders,
)


ANATOMY_PATTERN = re.compile(
    r"region\s*id\s*:\s*(\d+)\s*;\s*type\s*:\s*([^;\n]+)",
    re.IGNORECASE,
)
MH_PATTERN = re.compile(
    r"^\s*(?:[-#>]\s*)*(?:\*\*)?\s*stage\s*(?:\*\*)?\s*:\s*"
    r"(?:\*\*)?\s*([+-]?\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)
GLAUCOMA_PATTERN = re.compile(
    r"^\s*(?:[-#>]\s*)*(?:\*\*)?\s*"
    r"(non[- ]?glaucoma|glaucoma)\s*(?:\*\*)?\s*;\s*"
    r"(?:\*\*)?\s*(?:explanation|justification)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LMOD metrics from saved prediction checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results_rcd"))
    parser.add_argument(
        "--model-tag",
        help=(
            "Filename tag, e.g. RCD_qwen3_5_9b. If omitted, require exactly "
            "one matching task1/task2 checkpoint pair in --results-dir."
        ),
    )
    parser.add_argument("--task1-path", type=Path, help="Recognition checkpoint override")
    parser.add_argument("--task2-path", type=Path, help="Diagnosis checkpoint override")
    parser.add_argument(
        "--output-json", type=Path,
        help="Output JSON path (default: <results-dir>/lmod_metrics_<tag>.json)",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        help="One-row Table-ready CSV (default: <results-dir>/lmod_metrics_<tag>.csv)",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction checkpoint not found: {path}")
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON list of records in {path}")
    return records


def records_by_id(records: list[dict[str, Any]], source: Path) -> dict[str, list[dict[str, Any]]]:
    """Group records without collapsing repeated basenames across split folders."""
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Record without sample_id in {source}")
        indexed.setdefault(sample_id, []).append(row)
    return indexed


def infer_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    results_dir = args.results_dir.expanduser().resolve()
    task1 = args.task1_path.expanduser().resolve() if args.task1_path else None
    task2 = args.task2_path.expanduser().resolve() if args.task2_path else None
    tag = args.model_tag

    if task1 is not None or task2 is not None:
        if task1 is None or task2 is None:
            raise ValueError("Provide both --task1-path and --task2-path")
        if tag is None:
            prefix = "task1_anatomy_"
            suffix = "_results.json"
            name = task1.name
            tag = name[len(prefix) : -len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else "custom"
        return task1, task2, tag

    if tag:
        return (
            results_dir / f"task1_anatomy_{tag}_results.json",
            results_dir / f"task2_diagnosis_{tag}_results.json",
            tag,
        )

    pairs = []
    prefix, suffix = "task1_anatomy_", "_results.json"
    for candidate in sorted(results_dir.glob(f"{prefix}*{suffix}")):
        candidate_tag = candidate.name[len(prefix) : -len(suffix)]
        diagnosis = results_dir / f"task2_diagnosis_{candidate_tag}_results.json"
        if diagnosis.is_file():
            pairs.append((candidate, diagnosis, candidate_tag))
    if len(pairs) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint pair in {results_dir}, found {len(pairs)}; "
            "pass --model-tag or explicit checkpoint paths"
        )
    return pairs[0]


def strict_anatomy_parse(response: Any) -> dict[str, str] | None:
    if not isinstance(response, str) or not response.strip():
        return None
    predictions: dict[str, str] = {}
    for region_id, region_type in ANATOMY_PATTERN.findall(response):
        normalized = normalize_region(region_type.strip().strip("*_` .,:"))
        if normalized:
            # Models often repeat an analysis and then a final answer. The last
            # required-format occurrence is treated as the final prediction.
            predictions[str(int(region_id))] = normalized
    return predictions or None


def strict_mh_parse(response: Any) -> int | None:
    if not isinstance(response, str):
        return None
    match = MH_PATTERN.search(response)
    if not match:
        return None
    stage = int(match.group(1))
    return stage if stage in {1, 2, 3, 4} else None


def strict_glaucoma_parse(response: Any) -> str | None:
    if not isinstance(response, str):
        return None
    match = GLAUCOMA_PATTERN.search(response)
    if not match:
        return None
    label = re.sub(r"[- ]", "", match.group(1).lower())
    return "Non-Glaucoma" if label == "nonglaucoma" else "Glaucoma"


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def recognition_references(
    samples: list[dict[str, Any]],
) -> list[tuple[str, dict[str, str]]]:
    references = []
    for sample in samples:
        boxes = sample.get("bboxes") or []
        if not boxes:
            continue
        sample_id = f"{sample['source_dataset']}_{sample['id']}"
        references.append(
            (
                sample_id,
                {
                    str(box.get("annotation_ID", index + 1)): normalize_region(
                        str(box.get("region_type", ""))
                    )
                    for index, box in enumerate(boxes)
                },
            )
        )
    return references


def evaluate_recognition_scope(
    references: list[tuple[str, dict[str, str]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    correct = 0
    predicted_regions = 0
    reference_regions = 0
    hallucinated_regions = 0
    invalid = 0
    missing = 0
    unparseable = 0
    hr_values = []
    occurrence: Counter[str] = Counter()
    matched_records = 0

    for sample_id, truth in references:
        reference_regions += len(truth)
        position = occurrence[sample_id]
        occurrence[sample_id] += 1
        rows = predictions.get(sample_id, [])
        row = rows[position] if position < len(rows) else None
        if row is None:
            invalid += 1
            missing += 1
            continue
        matched_records += 1
        parsed = strict_anatomy_parse(row.get("response"))
        if parsed is None:
            invalid += 1
            unparseable += 1
            continue

        predicted_ids = set(parsed)
        truth_ids = set(truth)
        invented_ids = predicted_ids - truth_ids
        predicted_regions += len(predicted_ids)
        hallucinated_regions += len(invented_ids)
        correct += sum(
            region_id in truth and parsed[region_id] == truth[region_id]
            for region_id in predicted_ids
        )
        # LMOD: HR_i = 1 - |{r in P_i : r not in T_i}| / |P_i|.
        hr_values.append(1.0 - len(invented_ids) / len(predicted_ids))

    precision = safe_divide(correct, predicted_regions)
    recall = safe_divide(correct, reference_regions)
    f1 = harmonic_mean(precision, recall)
    expected = len(references)
    prefixes = {sample_id.split("_", 1)[0] for sample_id, _ in references}
    candidate_records = sum(
        len(rows)
        for sample_id, rows in predictions.items()
        if sample_id.split("_", 1)[0] in prefixes
    )
    extra = max(0, candidate_records - matched_records)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_resistance": (
            sum(hr_values) / len(hr_values) if hr_values else None
        ),
        "hallucination_resistance_micro": (
            1.0 - safe_divide(hallucinated_regions, predicted_regions)
            if predicted_regions else None
        ),
        "invalid_response_rate": safe_divide(invalid, expected),
        "counts": {
            "expected_samples": expected,
            "valid_samples": expected - invalid,
            "invalid_samples": invalid,
            "missing_prediction_records": missing,
            "unparseable_response_records": unparseable,
            "extra_prediction_records": extra,
            "correct_regions": correct,
            "predicted_regions": predicted_regions,
            "reference_regions": reference_regions,
            "hallucinated_region_ids": hallucinated_regions,
            "hr_defined_samples": len(hr_values),
        },
    }


def evaluate_diagnosis_scope(
    references: list[tuple[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    parser: Callable[[Any], Any | None],
) -> dict[str, Any]:
    correct = 0
    invalid = 0
    missing = 0
    unparseable = 0
    parsed_labels: Counter[str] = Counter()
    occurrence: Counter[str] = Counter()
    for sample_id, truth in references:
        position = occurrence[sample_id]
        occurrence[sample_id] += 1
        rows = predictions.get(sample_id, [])
        row = rows[position] if position < len(rows) else None
        if row is None:
            invalid += 1
            missing += 1
            continue
        predicted = parser(row.get("response"))
        if predicted is None:
            invalid += 1
            unparseable += 1
            continue
        parsed_labels[str(predicted)] += 1
        correct += predicted == truth

    expected = len(references)
    valid = expected - invalid
    return {
        # Primary accuracy counts invalid/missing responses as incorrect.
        "accuracy": safe_divide(correct, expected),
        "valid_response_accuracy": safe_divide(correct, valid),
        "invalid_response_rate": safe_divide(invalid, expected),
        "counts": {
            "expected_samples": expected,
            "valid_samples": valid,
            "invalid_samples": invalid,
            "missing_prediction_records": missing,
            "unparseable_response_records": unparseable,
            "correct_samples": correct,
            "parsed_prediction_distribution": dict(sorted(parsed_labels.items())),
        },
    }


def subset_references(
    references: list[tuple[str, dict[str, str]]], prefix: str
) -> list[tuple[str, dict[str, str]]]:
    return [item for item in references if item[0].startswith(prefix + "_")]


def flatten_for_table(report: dict[str, Any]) -> dict[str, Any]:
    recognition = report["recognition"]["overall"]
    glaucoma = report["diagnosis"]["glaucoma_detection"]
    mh = report["diagnosis"]["macular_hole_staging"]
    return {
        "model": report["model"],
        "evaluation_complete": report["evaluation_status"]["complete"],
        "recognition_precision": recognition["precision"],
        "recognition_recall": recognition["recall"],
        "recognition_f1": recognition["f1"],
        "recognition_hr": recognition["hallucination_resistance"],
        "recognition_invalid_response_rate": recognition["invalid_response_rate"],
        "glaucoma_accuracy": glaucoma["accuracy"],
        "glaucoma_invalid_response_rate": glaucoma["invalid_response_rate"],
        "macular_hole_staging_accuracy": mh["accuracy"],
        "macular_hole_staging_invalid_response_rate": mh["invalid_response_rate"],
    }


def validate_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite(child, f"{path}[{index}]")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    validate_dataset_folders(data_root)
    oct_samples, cfp_samples = index_datasets(data_root)
    all_samples = oct_samples + cfp_samples

    task1_path, task2_path, tag = infer_checkpoint_paths(args)
    task1_records = load_records(task1_path)
    task2_records = load_records(task2_path)
    task1_by_id = records_by_id(task1_records, task1_path)
    task2_by_id = records_by_id(task2_records, task2_path)

    recognition_refs = recognition_references(all_samples)
    mh_refs = [
        (f"{sample['source_dataset']}_{sample['id']}", sample["mh_stage"])
        for sample in oct_samples
        if sample.get("mh_stage") in {1, 2, 3, 4}
    ]
    glaucoma_refs = [
        (f"{sample['source_dataset']}_{sample['id']}", sample["glaucoma_label"])
        for sample in cfp_samples
        if sample.get("glaucoma_label") in {"Glaucoma", "Non-Glaucoma"}
    ]

    recognition_duplicate_keys = sum(
        count > 1 for count in Counter(key for key, _ in recognition_refs).values()
    )
    task1_duplicate_keys = sum(len(rows) > 1 for rows in task1_by_id.values())

    report = {
        "model": task1_records[0].get("model_name", tag) if task1_records else tag,
        "model_tag": tag,
        "inputs": {
            "data_root": str(data_root),
            "recognition_checkpoint": str(task1_path),
            "diagnosis_checkpoint": str(task2_path),
        },
        "metric_conventions": {
            "recognition_precision": "correct (region ID, normalized type) pairs / all unique predicted region IDs (micro)",
            "recognition_recall": "correct (region ID, normalized type) pairs / all LMOD reference region IDs (micro; invalid responses contribute no correct regions)",
            "recognition_f1": "harmonic mean of the reported micro precision and recall",
            "hallucination_resistance": "macro mean over parseable samples of LMOD HR_i = 1 - invented predicted IDs / predicted IDs",
            "diagnosis_accuracy": "correct strict-format predictions / all reference samples; invalid and missing responses count as incorrect",
            "invalid_response_rate": "missing or not strictly parseable responses / all reference samples",
            "duplicate_sample_ids": "repeated basenames under dataset split folders are paired to prediction records by deterministic occurrence order",
        },
        "input_audit": {
            "recognition_reference_rows": len(recognition_refs),
            "recognition_prediction_rows": len(task1_records),
            "recognition_reference_duplicate_keys": recognition_duplicate_keys,
            "recognition_prediction_duplicate_keys": task1_duplicate_keys,
            "diagnosis_prediction_rows": len(task2_records),
        },
        "recognition": {
            "overall": evaluate_recognition_scope(recognition_refs, task1_by_id),
            "by_dataset": {
                dataset: evaluate_recognition_scope(
                    subset_references(recognition_refs, dataset), task1_by_id
                )
                for dataset in ("OIMHS", "REFUGE", "ORIGA", "G1020", "IDRiD")
            },
        },
        "diagnosis": {
            "glaucoma_detection": evaluate_diagnosis_scope(
                glaucoma_refs, task2_by_id, strict_glaucoma_parse
            ),
            "macular_hole_staging": evaluate_diagnosis_scope(
                mh_refs, task2_by_id, strict_mh_parse
            ),
        },
    }
    missing_by_task = {
        "recognition": report["recognition"]["overall"]["counts"][
            "missing_prediction_records"
        ],
        "glaucoma_detection": report["diagnosis"]["glaucoma_detection"][
            "counts"
        ]["missing_prediction_records"],
        "macular_hole_staging": report["diagnosis"]["macular_hole_staging"][
            "counts"
        ]["missing_prediction_records"],
    }
    report["evaluation_status"] = {
        "complete": not any(missing_by_task.values()),
        "missing_prediction_records_by_task": missing_by_task,
    }
    validate_finite(report)

    results_dir = args.results_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json else results_dir / f"lmod_metrics_{tag}.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv else results_dir / f"lmod_metrics_{tag}.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    flat = flatten_for_table(report)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    print(json.dumps(flat, indent=2))
    if not report["evaluation_status"]["complete"]:
        print(
            "WARNING: evaluation is incomplete; missing prediction records: "
            + json.dumps(missing_by_task, sort_keys=True)
        )
    print(f"Detailed metrics: {output_json}")
    print(f"Table-ready row : {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
