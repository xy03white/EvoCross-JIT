# EvoCross-JIT

Research code for **An Evolution-Aware Two-Tower Interaction Framework for Just-in-Time Defect Prediction**.

EvoCross-JIT combines historical expert metrics with current-commit semantic and structural representations. Bidirectional cross-attention and gated fusion exchange information between the two towers.


```bash
python -m venv .venv
# Activate the virtual environment using the command appropriate to your shell.
python -m pip install -r requirements.txt
```

## Data preparation

The dataset repositories can be found in the open-source repository of the paper "Deep Just-in-Time Defect Prediction: How Far Are We?" at https://github.com/ZZR0/ISSTA21-JIT-DP, specifically under the path Data_Extraction/git_base/git_datasets/{project}/repo_urls.txt.


## Feature extraction

Run commands from the repository root. Please replace the uppercase paths before execution.

```bash
python semantic_feature.py --projects go jdt openstack platform --model_path PATH_TO_CODEBERT --data_dir dataset --output_dir features/semantic --checkpoint_dir models --epochs 10 --train_batch_size 16 --use_contrastive

python structure_feature/code_parser/build.py --grammar-dir PATH_TO_GRAMMAR_CHECKOUTS --output PATH_TO_LANGUAGE_LIBRARY --languages java go python

python structure_feature/extract_context.py --projects go jdt openstack platform --data_dir PATH_TO_ENRICHED_CSVS --output_dir features/structure --langlib PATH_TO_LANGUAGE_LIBRARY

python structure_feature/extract_ast_dfg.py --projects go jdt openstack platform --input_dir features/structure --langlib PATH_TO_LANGUAGE_LIBRARY

python structure_feature/extract_graphcodebert_vectors.py --projects go jdt openstack platform --input_dir features/structure --model_dir PATH_TO_GRAPHCODEBERT --langlib PATH_TO_LANGUAGE_LIBRARY --device cuda
```

Semantic extraction produces `features/semantic/PROJECT_SPLIT/COMMIT_ID.pt`. Structural extraction produces `method_code.jsonl`, `ast_dfg.jsonl`, and `ast_commit_vec.jsonl` inside `features/structure/PROJECT_SPLIT/`.
