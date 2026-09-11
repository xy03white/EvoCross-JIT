# Implementation notes

## Scope

This document specifies the released implementation, its data contracts and its relationship to the manuscript. The reference result tables have not been reproduced using this release. The differences below must be taken into account when comparing results.

## Mathematical mapping

| Manuscript | Implementation | Behavior |
| --- | --- | --- |
| Section 3.2.1, Eq. (1) | `evocross.models.Time2Vec` | One linear and seven sine components for normalized time gaps |
| Eq. (2) | `LSTMAttentionEncoder` | Four LSTM layers (128 hidden), two Transformer layers, four heads |
| Eqs. (3)-(4) | `LSTMAttentionEncoder`, `MultiScaleLSTMAttention` | Attention pooling within a window and learned softmax weights across windows |
| Section 3.2.2, semantic input | `semantic_feature.CommitDataset` | `[CLS] msg [ADD] added [DEL] removed`, followed by the original tokenizer special-token handling |
| Eqs. (5)-(6) | `RobertaBinaryClassifierWithContrastive` | Source contrastive ratio loss retained; discrepancy detailed below |
| Structural encoding, Eq. (7) | `extract_graphcodebert_vectors.py` | Graph-guided attention, CLS+mean method vector, mean anchor and softmax dot-product aggregation |
| Current Tower | `CurrentTower` | 768+1536 -> 512 -> 256 -> 128 MLP with ReLU/dropout |
| Eqs. (8)-(9) | `CrossAttention` | Four-head cross-attention, residual, dropout and LayerNorm |
| Eqs. (10)-(11) | `GMU` | Sigmoid gate and tanh projections of original/cross-view vectors |
| Section 3.4 | `metrics.search_threshold` | Validation-only Youden J coarse search and local F1 refinement |
| RQ2 / RQ3 | `variants.py`, `train.py` | Two single-tower variants, attention-only, GMU-only, concat, and full depths 1-3 |

Cross-attention receives one pooled vector per tower, represented as a sequence of length one. This preserves the original implementation; attention is not over all history tokens at this stage.

## Data contracts and chronology

The original CSV files are distributed unchanged. Commit IDs are strings and labels are binary. `date` or `time` must be numeric. The 14 metric columns are `ns, nd, nf, entrophy, la, ld, lt, fix, ndev, age, nuc, exp, rexp, sexp`. Optional text missingness does not remove a valid temporal record; missing required numerical fields produce explicit errors.

Training rows fit StandardScaler and the time-gap mean/standard deviation. Those values transform validation and test rows. Split boundaries and adjacent split ID overlap are checked before training. Feature IDs are intersected before window construction, matching the original source policy. This means a window counts feature-matched commits, not necessarily all raw commits. Each window includes the current commit and previous commits within the same split; it does not borrow training history for validation/test. The first `max(window_sizes)-1` matched records are ineligible. Time gaps restart at zero at the beginning of each window before normalization, preserving the original arithmetic.

Semantic PT files contain `feature` tensors of shape `(768,)`. Structural JSONL records contain `_id`, `vec` of length 1536, and optionally `num_methods`. Invalid vectors are not silently resized. Loading local PyTorch files assumes they are trusted artifacts.

Context extraction requires enriched CSVs; one row describes a changed source file. Method JSONL records carry `_id`, `method_index`, `lang`, `method_code`, `repo_path` and `file_path`. AST/DFG JSONL records carry `code_tokens` and `dfg`. Structural artifacts should contain affected-method contexts; whole-file fallback records from earlier extraction runs must be regenerated.

## Training and calibration

The shared training loop uses one eligible commit per optimizer update, Adam, learning rate 1e-4, gradient clipping at 1.0, and validation-based early stopping. There is no minibatch or optimizer-order change in the dual-tower trainer. Historical windows are prepared once to avoid repeated DataFrame slicing at every epoch.

The trainer uses ordinary binary cross-entropy with logits, without focal modulation or positive-class weighting. The manuscript does not explicitly specify the final tower loss. Default epochs/patience are 60/10, following Section 4.5. Semantic defaults are aligned to 10 epochs and batch size 16; supervised contrastive learning is enabled by default, with `--no_contrastive` available as an explicit ablation.

Threshold search retains the source 2nd-98th percentile interval, 0.05 coarse steps, Youden J criterion, +/-0.05 local interval and 0.001 F1 refinement. Ties keep the first candidate. `labels=[0,1]` makes confusion-matrix shape explicit for a single-class validation split. Test evaluation runs only after restoring the best validation checkpoint. A zero validation F1 can still save the first checkpoint. ROC AUC is null for single-class test labels; AUC-PR uses scikit-learn average precision, preserving the source metric convention rather than trapezoidal integration.

## Known limitations and manuscript differences

1. **Contrastive objective:** Eq. (5) averages positive-pair log probabilities with all non-self samples in the denominator. The source instead computes a log ratio of summed positive similarities to summed negative similarities, skips single-class batches, and includes numerical clamps. These are not equivalent. The implemented ratio objective therefore does not implement Eq. (5) exactly.
2. **Windows:** Section 4.5 lists six window sizes, whereas the code uses project-specific triples. Defaults retain those triples; an explicit CLI override allows the literal six-window configuration. The original run configurations behind all tables are not available.
3. **GraphCodeBERT pooling:** Mean pooling includes every returned sequence position, including padding, and is concatenated with CLS. The paper does not specify this detail. It is retained, not silently replaced with masked pooling.
4. **Parser limitations:** The source maps C++ data flow to the Java routine and C data flow to the Python routine. Those aliases remain disclosed source limitations; they are not validated C/C++ data-flow analyses. The context traversal may select both enclosing and nested methods, although the manuscript says innermost method. Multi-file records can reuse method indices; original resume keys are commit ID plus method index. These behaviors affect feature extraction and should be considered when interpreting coverage and results.
5. **Missing preprocessing:** Section 3.1 repository association and pre-change-file recovery code was not supplied. The included CSVs lack enriched fields, and GraphCodeBERT weights/features were not supplied. These external artifacts are required for full reproduction.
6. **Semantic numerical handling:** Existing clamping, invalid-batch handling and gradient accumulation behavior are retained. The default accumulation factor is one. The source contrastive loss can skip contributions for batches without suitable pairs. Do not interpret a completed run as proof that the mathematical objective matches Eq. (5).
7. **Result provenance:** The result images are reference artifacts. Training loss, defaults, feature eligibility and parameter initialization differ from earlier experimental code and can alter scores. Old dual-tower checkpoints are not automatically compatible with the cleaned state dict.

## Tests and validation scope

Run `python tests/smoke_test.py` from the repository root. The tests generate synthetic vectors and labels, check threshold handling and finite forward/backward computation for six model variants, and exercise a one-epoch checkpoint/save/reload/evaluation cycle. They do not contain real experiment records, provider credentials or private paths.

The tests have passed on Python 3.8 / PyTorch 1.12.1 CPU. Full dataset training, GPU execution, real GraphCodeBERT feature extraction, paid provider calls and the manuscript environment have not been verified for this release. Missing enriched inputs and structural feature vectors prevent a complete reproduction from the included CSVs alone.
