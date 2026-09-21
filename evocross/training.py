import argparse
import json
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, PROJECT_WINDOWS, SEMANTIC_DIM, STRUCT_DIM
from .metrics import compute_metrics, search_threshold
from .models import DualTowerWithGMU
from .variants import CurrentOnlyModel, DualTowerModel, TemporalOnlyModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vectors(directory, project, split, dimension):
    directory = Path(directory) / f"{project}_{split}"
    vectors = {}
    if dimension == STRUCT_DIM:
        path = directory / "ast_commit_vec.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if str(record["_id"]) in vectors:
                    raise ValueError(f"Duplicate structural ID in {path}")
                vectors[str(record["_id"])] = np.asarray(
                    record["vec"], dtype=np.float32
                )
    else:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.pt")):
            value = torch.load(path, map_location="cpu")
            if isinstance(value, dict):
                value = value.get("feature")
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            vectors[path.stem] = np.asarray(value, dtype=np.float32)
    for commit_id, value in vectors.items():
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(
                f"Invalid {dimension}-dimensional vector: {directory}/{commit_id}"
            )
    return vectors


def read_split(path):
    frame = pd.read_csv(path, dtype={"_id": str})
    time_column = "date" if "date" in frame.columns else "time"
    required = ["_id", "label", time_column] + FEATURE_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame[required].isnull().any().any():
        raise ValueError(f"{path}: missing required values")
    if frame["_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate commit IDs")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be binary")
    if not np.isfinite(
        frame[FEATURE_COLUMNS + [time_column]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path}: non-finite features or timestamps")
    return frame.sort_values(time_column).reset_index(drop=True), time_column


def prepare_data(args, project, windows):
    frames = {}
    for split in ("train", "val", "test"):
        frames[split], time_column = read_split(
            Path(args.data_dir) / f"{project}_{split}.csv"
        )
    for left, right in (("train", "val"), ("val", "test")):
        if frames[left][time_column].max() > frames[right][time_column].min():
            raise ValueError(f"Non-chronological split boundary: {left}/{right}")
        if set(frames[left]["_id"]) & set(frames[right]["_id"]):
            raise ValueError(f"Overlapping commit IDs: {left}/{right}")
    scaler = StandardScaler().fit(frames["train"][FEATURE_COLUMNS])
    times = frames["train"][time_column].to_numpy(dtype=np.float64)
    deltas = np.diff(times, prepend=times[0])
    time_mean, time_std = float(deltas.mean()), float(deltas.std()) or 1.0
    samples, counts = {}, {}
    for split, frame in frames.items():
        frame[FEATURE_COLUMNS] = scaler.transform(frame[FEATURE_COLUMNS])
        semantic = load_vectors(args.semantic_dir, project, split, SEMANTIC_DIM)
        structural = load_vectors(args.structural_dir, project, split, STRUCT_DIM)
        valid_ids = set(semantic) & set(structural)
        original_count = len(frame)
        # Preserve the original feature-intersection and within-split window policy.
        frame = frame[frame["_id"].isin(valid_ids)].reset_index(drop=True)
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        times = frame[time_column].to_numpy(dtype=np.float64)
        records = []
        for index in range(max(windows) - 1, len(frame)):
            commit_id = frame.iloc[index]["_id"]
            history = {}
            for window in windows:
                start = index - window + 1
                gaps = np.diff(times[start : index + 1], prepend=times[start])
                normalized = (
                    torch.tensor(gaps.reshape(-1, 1), dtype=torch.float32) - time_mean
                ) / (time_std + 1e-9)
                history[window] = (
                    torch.from_numpy(values[start : index + 1].copy()),
                    normalized,
                )
            records.append(
                (
                    commit_id,
                    history,
                    torch.from_numpy(semantic[commit_id]),
                    torch.from_numpy(structural[commit_id]),
                    int(frame.iloc[index]["label"]),
                )
            )
        if not records:
            raise ValueError(
                f"No usable samples in {project}/{split}; check feature coverage and windows"
            )
        samples[split] = records
        counts[split] = {
            "input": original_count,
            "feature_matched": len(frame),
            "evaluated": len(records),
        }
    return samples, scaler, time_mean, time_std, counts


def build_model(variant, windows, depth):
    if variant == "full":
        return DualTowerWithGMU(window_sizes=windows, num_blocks=depth)
    if variant == "temporal_only":
        return TemporalOnlyModel(windows)
    if variant == "current_only":
        return CurrentOnlyModel()
    return DualTowerModel(windows, variant, num_blocks=1)


def forward(model, sample, variant, device):
    _, history, semantic, structural, _ = sample
    history = {key: (x.to(device), t.to(device)) for key, (x, t) in history.items()}
    if variant == "temporal_only":
        return model.forward_single(history)
    if variant == "current_only":
        return model.forward_single(semantic.to(device), structural.to(device))
    return model.forward_single(history, semantic.to(device), structural.to(device))[-1]


def predict(model, samples, variant, device):
    model.eval()
    with torch.no_grad():
        probabilities = [
            torch.sigmoid(forward(model, sample, variant, device)).item()
            for sample in samples
        ]
    return [sample[-1] for sample in samples], probabilities


def run_project(args, project):
    set_seed(args.seed)
    windows = args.window_sizes or PROJECT_WINDOWS[project]
    if any(window < 1 for window in windows) or len(set(windows)) != len(windows):
        raise ValueError("Window sizes must be distinct positive integers")
    destination = Path(args.output_dir) / project / f"{args.variant}_d{args.depth}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metrics.json").exists():
        raise FileExistsError(
            f"Completed run already exists: {destination}; choose another output directory"
        )
    samples, scaler, time_mean, time_std, counts = prepare_data(args, project, windows)
    if args.validate_only:
        print(json.dumps({"project": project, "counts": counts}, indent=2))
        return
    device = torch.device(args.device)
    model = build_model(args.variant, windows, args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score, best_threshold, best_epoch, stale = -1.0, 0.5, 0, 0
    checkpoint = destination / "checkpoint.pt"
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for sample in samples["train"]:
            optimizer.zero_grad()
            logits = forward(model, sample, args.variant, device)
            label = torch.tensor(float(sample[-1]), device=device)
            loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        labels, probabilities = predict(model, samples["val"], args.variant, device)
        threshold, score, _, _ = search_threshold(labels, probabilities)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / len(samples["train"]),
                "validation_f1": float(score),
                "threshold": float(threshold),
            }
        )
        logging.info(
            "%s epoch=%d loss=%.5f validation_f1=%.4f",
            project,
            epoch + 1,
            history[-1]["loss"],
            score,
        )
        if score > best_score:
            best_score, best_threshold, best_epoch, stale = (
                float(score),
                float(threshold),
                epoch + 1,
                0,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "window_sizes": windows,
                    "time_mean": time_mean,
                    "time_std": time_std,
                    "variant": args.variant,
                    "depth": args.depth,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    labels, probabilities = predict(model, samples["test"], args.variant, device)
    metrics = {
        "project": project,
        "variant": args.variant,
        "depth": args.depth,
        "window_sizes": windows,
        "best_epoch": best_epoch,
        "validation_f1": best_score,
        "threshold": best_threshold,
        "test": compute_metrics(labels, probabilities, best_threshold),
        "counts": counts,
        "config": vars(args),
        "history": history,
        "torch_version": torch.__version__,
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    joblib.dump(scaler, destination / "scaler.joblib")
    pd.DataFrame(
        {
            "_id": [s[0] for s in samples["test"]],
            "label": labels,
            "probability": probabilities,
            "prediction": [int(p >= best_threshold) for p in probabilities],
        }
    ).to_csv(destination / "predictions.csv", index=False)
    print(json.dumps(metrics["test"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects", nargs="+", choices=sorted(PROJECT_WINDOWS), required=True
    )
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--semantic-dir", default="features/semantic")
    parser.add_argument("--structural-dir", default="features/structure")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "temporal_only",
            "current_only",
            "attention_only",
            "gmu_only",
            "concat",
        ],
        default="full",
    )
    parser.add_argument("--depth", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--window-sizes", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0:
        parser.error("epochs, patience and learning rate must be positive")
    if args.variant != "full" and args.depth != 1:
        parser.error("Paper ablations use one fusion stage; use --depth 1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for project in args.projects:
        run_project(args, project)


if __name__ == "__main__":
    main()
