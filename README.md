# EvoCross-JIT

Research code for **An Evolution-Aware Two-Tower Interaction Framework for Just-in-Time Defect Prediction**.

EvoCross-JIT combines historical expert metrics with current-commit semantic and structural representations. Bidirectional cross-attention and gated fusion exchange information between the two towers.

## Reproducibility

This repository provides the implementation, chronological CSV splits and original result-table images. The table images are reference results and have not been regenerated with this release. See [IMPLEMENT.md](IMPLEMENT.md) for the effective configuration, required external inputs and known differences from the manuscript. Complete dataset training and GPU reproduction have not been verified for this release.

## Environment

The manuscript reports Python 3.9.19 and PyTorch 2.4. The dependency file is a proposed compatible environment, not a recovered lockfile from the original experiments. The Transformers version retains the original Hugging Face AdamW implementation; Tree-sitter 0.20.1 retains the shared-library parser API used by the supplied code.

```bash
python -m venv .venv
# Activate the virtual environment using the command appropriate to your shell.
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

Install a PyTorch build appropriate to your CUDA environment if GPU training is required. Smoke tests passed on Python 3.8 / PyTorch 1.12.1 CPU; the manuscript GPU environment was not available.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Shared full-model and ablation command-line entry point |
| `evocross/` | Model components, variants, validation, training, metrics |
| `semantic_feature.py` | Project-specific CodeBERT fine-tuning and vector extraction |
| `structure_feature/` | Method context, AST/DFG, GraphCodeBERT extraction |
| `discussion/compare_llms.py` | Optional provider-configured LLM evaluation |
| `dataset/` | Original Go, JDT, OpenStack and Platform CSV splits |
| `results/` | Original result-table images, unchanged |
| `tests/` | Smoke tests using generated synthetic inputs |
| `IMPLEMENT.md` | Data contracts, mathematical mapping and known discrepancies |

## Inputs to prepare

1. Obtain local CodeBERT and GraphCodeBERT model/tokenizer directories. Model weights are not committed.
2. Supply enriched CSV files with `_id`, `lang`, `old_file`, `old_code_diff`, `repo_path`, and `file_path`. The included CSVs do **not** contain these context fields. Repository association and context recovery described in Section 3.1 must be performed separately; that preprocessing stage is not included.
3. Build a Tree-sitter shared library from grammar checkouts compatible with Tree-sitter 0.20.1. Set every example path below for your own environment.

The basic CSVs contain `_id`, `date` (Unix time), `label`, `msg`, `code`, and 14 expert metrics. The source spelling `entrophy` is intentionally preserved for compatibility. Preserve dataset source-code and message contents when preparing features.

## Feature extraction

Run commands from the repository root. Uppercase paths below are placeholders; replace them before execution.

```bash
python semantic_feature.py --projects go jdt openstack platform --model_path PATH_TO_CODEBERT --data_dir dataset --output_dir features/semantic --checkpoint_dir models --epochs 10 --train_batch_size 16 --use_contrastive

python structure_feature/code_parser/build.py --grammar-dir PATH_TO_GRAMMAR_CHECKOUTS --output PATH_TO_LANGUAGE_LIBRARY --languages java go python

python structure_feature/extract_context.py --projects go jdt openstack platform --data_dir PATH_TO_ENRICHED_CSVS --output_dir features/structure --langlib PATH_TO_LANGUAGE_LIBRARY

python structure_feature/extract_ast_dfg.py --projects go jdt openstack platform --input_dir features/structure --langlib PATH_TO_LANGUAGE_LIBRARY

python structure_feature/extract_graphcodebert_vectors.py --projects go jdt openstack platform --input_dir features/structure --model_dir PATH_TO_GRAPHCODEBERT --langlib PATH_TO_LANGUAGE_LIBRARY --device cuda
```

Semantic extraction produces `features/semantic/PROJECT_SPLIT/COMMIT_ID.pt`. Structural extraction produces `method_code.jsonl`, `ast_dfg.jsonl`, and `ast_commit_vec.jsonl` inside `features/structure/PROJECT_SPLIT/`. The final structural vector must have 1536 elements; the semantic vector must have 768.

The structural pipeline retains source limitations documented in IMPLEMENT.md, including C/C++ DFG aliases and nested-method selection. Do not present these as validated language-specific analyses.

## Training and evaluation

```bash
python train.py --projects go --validate-only
python train.py --projects go --depth 1 --device cuda
python train.py --projects jdt --depth 1 --device cuda
python train.py --projects openstack platform --depth 2 --device cuda
```

The depth examples reflect the Table 3/Table 6 correspondence; they are not a claim of reproduced scores. Defaults are 60 epochs, patience 10, learning rate 1e-4, seed 42. Each run records effective configuration, coverage counts, checkpoint, scaler, validation history, test probabilities, and final metrics under `runs/PROJECT/VARIANT_dDEPTH/`. A completed run is not overwritten; use a new `--output-dir` for another seed or configuration.

Default windows preserve the supplied per-project code: Go/Platform `[1,5,10]`, OpenStack `[1,10,20]`, JDT `[1,20,30]`. Section 4.5 lists `[1,5,10,20,30,40]` without resolving the per-project choice. To run that literal set, explicitly pass `--window-sizes 1 5 10 20 30 40` and use a separate output directory.

```bash
python train.py --projects go --variant temporal_only
python train.py --projects go --variant current_only
python train.py --projects go --variant attention_only
python train.py --projects go --variant gmu_only
python train.py --projects go --variant concat
python train.py --projects go --depth 3
```

Paper ablations use one fusion stage. All variants share the feature intersection and eligible temporal indices, so their evaluation population is explicit and consistent. Missing vectors are excluded and reported; missing vector directories, malformed dimensions, invalid required values, and empty eligible splits are errors.

## Optional LLM comparison

```bash
python -m pip install -r requirements-llm.txt
# Set LLM_API_KEY in your shell; .env.example is documentation only.
python discussion/compare_llms.py --projects go --model EXACT_PROVIDER_MODEL_ID --base-url YOUR_PROVIDER_ENDPOINT --api-key-env LLM_API_KEY --output-dir runs/llm/EXPERIMENT_NAME
```

Use the exact model identifiers and endpoints available to your account. API execution sends commit content to that provider and may incur charges. No API evaluation has been verified for this release. The script reports discrete-label F1, precision and recall; confidence values are retained only as response metadata. Failed requests raise errors rather than becoming clean predictions. Historical defect labels are not included in prompts.

## Attribution

Parser code retains its Microsoft copyright notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A project-wide license for the original research code and dataset has not been specified.
