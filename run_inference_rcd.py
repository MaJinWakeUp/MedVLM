#!/usr/bin/env python3
"""Run the LMOD benchmark with Qwen3.5-9B hosted by Clemson RCD-LLM.

The hosted API is OpenAI-compatible. Images are encoded locally as data URLs;
model weights are not downloaded and no local GPU is used.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from local_full_dataset_inference import (
    atomic_json_dump,
    index_datasets,
    load_checkpoint,
    model_tag,
    run_anatomy,
    run_diagnosis,
    save_outputs,
    validate_dataset_folders,
)

DEFAULT_BASE_URL = "https://llm.rcd.clemson.edu/v1"
DEFAULT_MODEL = "qwen3.5-9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LMOD inference through Clemson RCD-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results_rcd"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("RCD_LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible RCD API base URL",
    )
    parser.add_argument(
        "--tasks", choices=("both", "anatomy", "diagnosis"), default="both"
    )
    parser.add_argument("--max-oct-samples", type=int)
    parser.add_argument("--max-cfp-samples", type=int)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="save frequently so interrupted hosted requests are not repeated",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=2048,
        help="downscale larger images before upload; 0 preserves original size",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from model-specific JSON checkpoints",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Qwen thinking consumes the same max-token budget",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="skip the initial GET /models availability check",
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help="index/validate extracted data without contacting RCD",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="print RCD model IDs containing 'qwen' and exit",
    )
    args = parser.parse_args()
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.max_image_side < 0:
        parser.error("--max-image-side must be non-negative")
    for name in ("max_oct_samples", "max_cfp_samples"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


class RCDQwenEngine:
    """Adapter exposing the same ``infer(image, prompt)`` API as the local model."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_new_tokens: int,
        timeout: float,
        max_retries: int,
        enable_thinking: bool,
        jpeg_quality: int,
        max_image_side: int,
    ) -> None:
        api_key = os.getenv("RCD_LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "RCD_LLM_API_KEY is not set. Export it before hosted inference."
            )
        if not base_url.strip():
            raise ValueError("RCD base URL is empty")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required. Install requirements-rcd.txt."
            ) from exc

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.jpeg_quality = jpeg_quality
        self.max_image_side = max_image_side
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def available_models(self) -> list[str]:
        return sorted(str(item.id) for item in self.client.models.list().data)

    def ensure_model_available(self) -> None:
        available = self.available_models()
        if self.model_name not in available:
            qwen_models = [name for name in available if "qwen" in name.lower()]
            raise RuntimeError(
                f"RCD model {self.model_name!r} is unavailable. "
                f"Available Qwen models: {qwen_models}"
            )
        print(f"[rcd] model available: {self.model_name}")

    def image_data_url(self, image: Image.Image) -> str:
        converted = image.convert("RGB")
        if self.max_image_side and max(converted.size) > self.max_image_side:
            converted.thumbnail(
                (self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS
            )
        buffer = io.BytesIO()
        converted.save(buffer, format="JPEG", quality=self.jpeg_quality)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def record_usage(self, response: Any) -> None:
        self.usage["requests"] += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage[key] += int(getattr(usage, key, 0) or 0)

    @staticmethod
    def _text_field(value: Any) -> str:
        """Normalize text returned as a string or structured content list."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                text = (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()
        return str(value or "").strip()

    @classmethod
    def response_text(cls, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("RCD returned no completion choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = cls._text_field(getattr(message, "content", None))
        if content:
            return content

        # RCD qwen3.5-9b currently returns its complete answer here even when
        # enable_thinking=False. Other compatible servers use reasoning_content.
        reasoning = cls._text_field(getattr(message, "reasoning", None))
        if not reasoning:
            reasoning = cls._text_field(
                getattr(message, "reasoning_content", None)
            )

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise RuntimeError(
                "RCD exhausted max_tokens before returning a complete answer; "
                "increase --max-new-tokens"
            )
        if reasoning:
            return reasoning
        raise RuntimeError(
            "RCD returned no content or reasoning "
            f"(finish_reason={finish_reason!r})"
        )

    def infer(self, image: Image.Image, prompt: str) -> str:
        request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_data_url(image)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            },
        }
        response = self.client.chat.completions.create(**request)
        self.record_usage(response)
        return self.response_text(response)


def main() -> int:
    args = parse_args()
    engine = None
    if args.list_models:
        engine = RCDQwenEngine(
            args.model_name, args.base_url, args.max_new_tokens,
            args.request_timeout, args.max_retries, args.enable_thinking,
            args.jpeg_quality, args.max_image_side,
        )
        for name in engine.available_models():
            if "qwen" in name.lower():
                print(name)
        return 0

    data_root = args.data_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] data={data_root}")
    print(f"[paths] results={results_dir}")
    validate_dataset_folders(data_root)
    oct_samples, cfp_samples = index_datasets(data_root)
    if not oct_samples and not cfp_samples:
        raise RuntimeError(f"No LMOD samples found under {data_root}")
    if args.max_oct_samples is not None:
        oct_samples = oct_samples[: args.max_oct_samples]
    if args.max_cfp_samples is not None:
        cfp_samples = cfp_samples[: args.max_cfp_samples]
    print(f"[run] selected OCT={len(oct_samples)}, CFP={len(cfp_samples)}")
    if args.index_only:
        print("[run] index-only validation complete; RCD was not contacted")
        return 0

    engine = RCDQwenEngine(
        args.model_name, args.base_url, args.max_new_tokens,
        args.request_timeout, args.max_retries, args.enable_thinking,
        args.jpeg_quality, args.max_image_side,
    )
    if not args.skip_model_check:
        engine.ensure_model_available()

    run_name = f"RCD/{args.model_name}"
    tag = model_tag(run_name)
    error_path = results_dir / f"errors_{tag}.jsonl"
    task1_path = results_dir / f"task1_anatomy_{tag}_results.json"
    task2_path = results_dir / f"task2_diagnosis_{tag}_results.json"
    samples = oct_samples + cfp_samples

    task1 = (
        run_anatomy(
            samples, engine, task1_path, error_path, args.checkpoint_interval,
            args.resume, run_name,
        )
        if args.tasks in ("both", "anatomy")
        else load_checkpoint(task1_path, True)
    )
    task2 = (
        run_diagnosis(
            samples, engine, task2_path, error_path, args.checkpoint_interval,
            args.resume, run_name,
        )
        if args.tasks in ("both", "diagnosis")
        else load_checkpoint(task2_path, True)
    )
    save_outputs(results_dir, tag, run_name, task1, task2)
    usage_path = results_dir / f"usage_{tag}.json"
    atomic_json_dump(
        {
            "model_name": args.model_name,
            "base_url": args.base_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **engine.usage,
        },
        usage_path,
    )
    print(f"[rcd] usage={json.dumps(engine.usage)}")
    print(f"[done] results saved to {results_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        raise SystemExit(130)
