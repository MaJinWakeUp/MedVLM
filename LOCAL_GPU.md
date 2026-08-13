# LMOD inference: local GPU and Clemson RCD-LLM

The two command-line runners share the same extracted-dataset indexer, prompts,
checkpoint format, metrics, CSV exports, and plots:

- `local_full_dataset_inference.py` loads InternVL or Qwen on a local GPU.
- `run_inference_rcd.py` sends images to RCD-hosted `qwen3.5-9b`; it downloads no
  model weights and does not use a local GPU.

## 1. Data folder layout

The scripts now expect data to already be extracted. Use this layout:

```text
LMOD/
├── local_full_dataset_inference.py
├── run_inference_rcd.py
├── requirements-local.txt
├── requirements-rcd.txt
├── data/
│   ├── OIMHS/
│   ├── REFUGE/
│   ├── ORIGA/
│   ├── G1020/
│   └── IDRiD/
├── results/                   # local-GPU outputs
└── results_rcd/               # hosted RCD outputs
```

Every sample directory must eventually contain:

```text
data/OIMHS/100_0/
├── information.json
├── visualization.png
└── annotated/
    └── bbox_annotated.png
```

`annotated_bounding_box.png` is also recognized. Extra split levels are fine,
for example `data/REFUGE/train/V0001/` and
`data/IDRiD/Testing/IDRiD_100/`, because indexing is recursive within each of
the five named dataset folders.

Validate all extracted folders without loading or contacting a model:

```bash
python local_full_dataset_inference.py --data-root data --index-only
python run_inference_rcd.py --data-root data --index-only
```

## 2. Local-GPU environment and run

Use Python 3.10 or 3.11. Install the CUDA build of PyTorch matching your NVIDIA
driver first, using the command generated at
<https://pytorch.org/get-started/locally/>, then:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the appropriate CUDA PyTorch + torchvision build here.
python -m pip install -r requirements-local.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Qwen3.5 may require Transformers from its main branch if the installed release
does not recognize `qwen3_5`:

```bash
python -m pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

Start with a four-image smoke test:

```bash
python local_full_dataset_inference.py \
  --data-root data \
  --results-dir results \
  --model OpenGVLab/InternVL3_5-8B \
  --max-oct-samples 2 \
  --max-cfp-samples 2
```

Run the full extracted dataset by removing both `--max-...` options:

```bash
python local_full_dataset_inference.py \
  --data-root data \
  --results-dir results \
  --model OpenGVLab/InternVL3_5-8B
```

Qwen alternatives are `Qwen/Qwen3.5-9B` and
`Qwen/Qwen2.5-VL-7B-Instruct`. Use `CUDA_VISIBLE_DEVICES=1` before the command
to choose a physical GPU.

## 3. RCD-LLM environment and run

The RCD runner uses an OpenAI-compatible API. Install its lighter dependency
set and export the API key. Do not put the key in source code or a command-line
argument:

```bash
python -m venv .venv-rcd
source .venv-rcd/bin/activate
python -m pip install -r requirements-rcd.txt
export RCD_LLM_API_KEY="YOUR_RCD_KEY"
export RCD_LLM_BASE_URL="https://llm.rcd.clemson.edu/v1"
```

The live service model ID verified during development is `qwen3.5-9b`. You can
check the current Qwen model list without sending an image:

```bash
python run_inference_rcd.py --list-models
```

Run a small smoke test before submitting the full benchmark:

```bash
python run_inference_rcd.py \
  --data-root data \
  --results-dir results_rcd \
  --model-name qwen3.5-9b \
  --max-oct-samples 2 \
  --max-cfp-samples 2
```

Then run the full dataset:

```bash
python run_inference_rcd.py \
  --data-root data \
  --results-dir results_rcd \
  --model-name qwen3.5-9b
```

The RCD run performs two requests for most eligible samples (anatomy and
diagnosis), so the full benchmark can make many thousands of hosted requests.
It saves after each request by default and resumes automatically. Use
`--tasks anatomy` or `--tasks diagnosis` to run one pass. Because medical images
are uploaded to the RCD service, confirm that your data-use rules permit this.

Useful RCD controls:

```text
--request-timeout 600
--max-retries 5
--max-new-tokens 1024
--max-image-side 2048
--enable-thinking          # disabled by default
--no-resume
--skip-model-check
```

## 4. Outputs and resume behavior

Keep the same results directory and model name to resume. Local and RCD output
names have different model tags and do not overwrite one another:

```text
results_rcd/
├── task1_anatomy_RCD_qwen3_5_9b_results.json
├── task2_diagnosis_RCD_qwen3_5_9b_results.json
├── anatomical_recognition_RCD_qwen3_5_9b_predictions.csv
├── diagnosis_RCD_qwen3_5_9b_predictions.csv
├── evaluation_summary_RCD_qwen3_5_9b_report.json
├── confusion_matrices_RCD_qwen3_5_9b.png
├── errors_RCD_qwen3_5_9b.jsonl
└── usage_RCD_qwen3_5_9b.json
```

Failed samples are logged and not marked complete, so the next resumed run
retries them. `--no-resume` starts from empty in memory but does not delete old
files; successful completion overwrites the model-specific checkpoints.

## 5. LMOD evaluation

After inference, compute recognition precision, recall, F1, hallucination
resistance (HR), diagnosis accuracy, and invalid-response rates with:

```bash
python evaluate_lmod_predictions.py \
  --data-root data \
  --results-dir results_rcd \
  --model-tag RCD_qwen3_5_9b
```

The evaluator rebuilds reference labels from the extracted dataset and strictly
reparses each raw response. Its primary conventions are:

- Recognition precision is the number of correct `(region ID, type)` pairs
  divided by all unique predicted region IDs; recall uses all reference region
  IDs. F1 is their harmonic mean.
- Per-image HR follows LMOD exactly:
  `1 - invented predicted IDs / predicted IDs`. The reported HR is the macro
  mean over parseable responses because HR is undefined for an empty predicted
  set. Empty and missing outputs are captured by invalid-response rate.
- Glaucoma and macular-hole accuracies use all labeled reference samples as the
  denominator. Invalid or missing responses count as incorrect.
- A response is invalid when it is missing or cannot be parsed in the required
  prompt format. In particular, macular-hole stages outside 1–4 are invalid.

Outputs:

```text
results_rcd/lmod_metrics_RCD_qwen3_5_9b.json  # detailed counts and breakdowns
results_rcd/lmod_metrics_RCD_qwen3_5_9b.csv   # one row for the results table
```
