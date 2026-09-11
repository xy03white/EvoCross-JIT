"""CPU smoke tests; run from the repository root with python tests/smoke_test.py."""

import sys, json, tempfile
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evocross.metrics import search_threshold
from evocross.training import build_model, forward, run_project

torch.set_num_threads(1)
torch.manual_seed(42)
semantic, structural = torch.randn(768), torch.randn(1536)
for labels in [[0, 0], [1, 1]]:
    assert np.isfinite(search_threshold(labels, [0.2, 0.8])).all()
for variant in [
    "full",
    "temporal_only",
    "current_only",
    "attention_only",
    "gmu_only",
    "concat",
]:
    model = build_model(variant, [1, 2], 1)
    sample = (
        "test",
        {w: (torch.randn(w, 14), torch.randn(w, 1)) for w in [1, 2]},
        semantic,
        structural,
        1,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        forward(model, sample, variant, "cpu"), torch.tensor(1.0)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
from structure_feature.extract_context import parse_unified_diff_old_lines

assert parse_unified_diff_old_lines("@@ -1,2 +1,2 @@\n-a\n+b\n c") == {1, 2}
import pandas as pd
from evocross.config import FEATURE_COLUMNS

with tempfile.TemporaryDirectory() as temp:
    base = Path(temp)
    for si, split in enumerate(["train", "val", "test"]):
        rows = []
        (base / "semantic" / f"go_{split}").mkdir(parents=True)
        (base / "structure" / f"go_{split}").mkdir(parents=True)
        structures = []
        for i in range(4):
            key = f"{split}_{i}"
            rows.append(
                dict(
                    _id=key,
                    date=si * 100 + i,
                    label=i % 2,
                    **{k: float(i) for k in FEATURE_COLUMNS},
                )
            )
            torch.save(
                {"feature": torch.randn(768)},
                base / "semantic" / f"go_{split}" / f"{key}.pt",
            )
            structures.append(
                json.dumps({"_id": key, "vec": torch.randn(1536).tolist()})
            )
        pd.DataFrame(rows).to_csv(base / f"go_{split}.csv", index=False)
        (base / "structure" / f"go_{split}" / "ast_commit_vec.jsonl").write_text(
            "\n".join(structures), encoding="utf-8"
        )
    args = Namespace(
        seed=42,
        window_sizes=[1, 2],
        data_dir=str(base),
        semantic_dir=str(base / "semantic"),
        structural_dir=str(base / "structure"),
        output_dir=str(base / "runs"),
        variant="full",
        depth=1,
        validate_only=False,
        device="cpu",
        learning_rate=1e-4,
        epochs=1,
        patience=1,
    )
    run_project(args, "go")
    report = json.loads((base / "runs/go/full_d1/metrics.json").read_text())
    assert report["best_epoch"] == 1 and report["counts"]["test"]["evaluated"] == 3
print(
    "PASS: thresholds, six model variants, diff parsing, and synthetic training/checkpoint/evaluation."
)
