"""Optional API-based comparison; predictions are discrete decisions, not calibrated probabilities."""

import os
import argparse
import hashlib
import json
import time
import random
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

PROJECTS = ["jdt", "platform", "go", "openstack"]
PROJECT_WINDOW = {
    "platform": [1, 5, 10],
    "go": [1, 5, 10],
    "jdt": [1, 20, 30],
    "openstack": [1, 10, 20],
}

CSV_DIR = "./dataset"
METHOD_CODE_DIR = "./features/structure"
RESULT_DIR = "./runs/llm"

FEATURE_COLUMNS = [
    "ns",
    "nd",
    "nf",
    "entrophy",
    "la",
    "ld",
    "lt",
    "fix",
    "ndev",
    "age",
    "nuc",
    "exp",
    "rexp",
    "sexp",
]
LABEL_COLUMN = "label"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(project: str):
    logger = logging.getLogger(project)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(
        os.path.join(RESULT_DIR, f"{project}_llm_eval.log"), encoding="utf-8"
    )
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_method_code_map(project: str, split: str, logger=None) -> Dict[str, List[str]]:
    jsonl_path = os.path.join(
        METHOD_CODE_DIR, f"{project}_{split}", "method_code.jsonl"
    )
    if not os.path.exists(jsonl_path):
        msg = f"method_code.jsonl not found for {project}_{split}, using empty."
        if logger:
            logger.warning(msg)
        else:
            logging.warning(msg)
        return {}
    method_map = defaultdict(dict)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                _id = data.get("_id")
                method_idx = data.get("method_index", 0)
                code = data.get("method_code", "")
                if _id is None:
                    continue
                method_map[_id][method_idx] = code
            except Exception:
                continue
    result = {}
    for _id, idx_dict in method_map.items():
        sorted_indices = sorted(idx_dict.keys())
        result[_id] = [idx_dict[i] for i in sorted_indices]
    return result


def parse_code_field(code_str: str) -> str:
    if not code_str:
        return "No code changes."
    try:
        import ast

        parts = ast.literal_eval(code_str)
        return "\n".join(parts)
    except:
        return code_str


def get_history_features(
    df: pd.DataFrame, idx: int, window_sizes: List[int]
) -> Tuple[str, str]:

    max_window = max(window_sizes)
    start = max(0, idx - (max_window - 1))
    history_rows = df.iloc[start:idx]
    current_row = df.iloc[idx]

    feature_def = (
        "Feature Definition of Historical Commits:\n"
        "For each commit in history, we extract several numeric features, including:\n"
        "- ns: number of changed subsystems\n"
        "- nd: number of changed directories\n"
        "- nf: number of modified files\n"
        "- entrophy: entropy of code changes\n"
        "- la: lines added\n"
        "- ld: lines deleted\n"
        "- lt: lines of code in files before the commit\n"
        "- fix: whether the previous commit was a fix\n"
        "- ndev: number of developers who modified the file\n"
        "- age: file age\n"
        "- nuc: number of unique changes\n"
        "- exp, rexp, sexp: developer experience metrics\n"
    )

    def format_val(val):
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        elif isinstance(val, (float, np.floating)):

            if val.is_integer():
                return str(int(val))
            else:
                return str(val)
        else:
            return str(val)

    history_lines = []
    num_history = len(history_rows)
    for i, (_, row) in enumerate(history_rows.iterrows()):
        feat_strs = []
        for col in FEATURE_COLUMNS:
            val = row[col]
            feat_strs.append(f"{col}: {format_val(val)}")
        commit_idx = -(num_history - i)
        line = f"Commit {commit_idx}: " + ", ".join(feat_strs)
        history_lines.append(line)

    if num_history == 0:
        history_text = feature_def + "\nNo historical commits available."
    else:
        history_text = (
            feature_def
            + f"\nThe Feature Values from the Last {num_history} Commits:\n"
            + "\n".join(history_lines)
        )

    curr_strs = []
    for col in FEATURE_COLUMNS:
        val = current_row[col]
        curr_strs.append(f"{col}: {format_val(val)}")
    current_text = "Current Commit: " + ", ".join(curr_strs)

    return history_text, current_text


class LLMInference:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        extra_params: Optional[Dict] = None,
        cache_file: str = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.extra_params = extra_params or {}
        self.cache_file = cache_file
        self.cache = {}
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)

    def save_cache(self):
        if self.cache_file:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)

    def predict(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Tuple[int, float]:
        cache_key = hashlib.sha256(
            (str(self.client.base_url) + self.model + prompt).encode("utf-8")
        ).hexdigest()
        if cache_key in self.cache:
            result = self.cache[cache_key]
            return result["pred"], result["conf"]

        messages = [
            {
                "role": "system",
                "content": "You are a code defect prediction assistant. Respond only in JSON format.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            for key, val in self.extra_params.items():
                kwargs[key] = val

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            pred = int(data.get("prediction", 0))
            conf = float(data.get("confidence", 0.5))
            if pred not in (0, 1) or not 0 <= conf <= 1:
                raise ValueError("Invalid API prediction or confidence")
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise RuntimeError("API response failed; no prediction was recorded") from e

        self.cache[cache_key] = {"pred": pred, "conf": conf}
        if len(self.cache) % 10 == 0:
            self.save_cache()
        return pred, conf


def build_prompt(
    msg: str,
    code: str,
    method_codes: List[str],
    history_text: str,
    current_features: str,
) -> str:

    method_context = (
        "\n".join([f"Method {i+1}:\n{code}" for i, code in enumerate(method_codes)])
        if method_codes
        else "No method-level context available."
    )

    prompt = f"""You are given information about a code commit. Determine if this commit is likely to introduce a bug (defect). Output a JSON object with two fields:
- "prediction": Output ONLY one number: 0 or 1: 0 means clean commit, 1 means defect-inducing commit. Do NOT output any explanation
- "confidence": a float between 0 and 1 indicating your certainty.

{history_text}

{current_features}

Commit message:
{msg}

Code changes (added/removed lines):
{code}

Method-level context:
{method_context}

Based on the provided information, output your prediction in JSON format with fields "prediction" (0 or 1) and "confidence" (0-1).
"""
    return prompt


def evaluate_llm_on_project(
    project: str, df_test: pd.DataFrame, llm: LLMInference, window_sizes: List[int]
):
    logger = get_logger(project)
    logger.info(f"Evaluating {project} with model {llm.model}")

    method_code_map = load_method_code_map(project, "test", logger)
    results = []

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        _id = str(row["_id"])
        msg = row.get("msg", "")
        code = parse_code_field(row.get("code", ""))
        label = int(row[LABEL_COLUMN])

        history_text, current_features = get_history_features(
            df_test, idx, window_sizes
        )
        method_codes = method_code_map.get(_id, [])

        prompt = build_prompt(msg, code, method_codes, history_text, current_features)
        pred, conf = llm.predict(prompt)

        results.append(
            {
                "_id": _id,
                "true_label": label,
                "pred_label": pred,
                "confidence": conf,
            }
        )

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx+1}/{len(df_test)} samples")

    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    metrics = {
        "project": project,
        "model": llm.model,
        "window_sizes": window_sizes,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_samples": len(results),
    }

    result_file = os.path.join(
        RESULT_DIR, f"{project}_{llm.model.replace('/', '_')}_results.json"
    )
    with open(result_file, "w") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2)

    logger.info(f"Results for {project} {llm.model}: F1={f1:.4f}, Recall={recall:.4f}")
    return metrics


def main():
    global CSV_DIR, METHOD_CODE_DIR, RESULT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="+", choices=PROJECTS, required=True)
    parser.add_argument(
        "--model", required=True, help="Exact model identifier from your provider"
    )
    parser.add_argument("--base-url", required=True, help="API endpoint for this model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--data-dir", default=CSV_DIR)
    parser.add_argument("--method-dir", default=METHOD_CODE_DIR)
    parser.add_argument("--output-dir", default=RESULT_DIR)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Set the {args.api_key_env} environment variable")
    CSV_DIR, METHOD_CODE_DIR, RESULT_DIR = (
        args.data_dir,
        args.method_dir,
        args.output_dir,
    )
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_seed(42)
    llm = LLMInference(
        api_key,
        args.base_url,
        args.model,
        cache_file=os.path.join(RESULT_DIR, "cache.json"),
    )
    all_metrics = []
    for project in args.projects:
        path = os.path.join(CSV_DIR, f"{project}_test.csv")
        frame = pd.read_csv(path).fillna("")
        time_column = "date" if "date" in frame.columns else "time"
        frame = frame.sort_values(time_column).reset_index(drop=True)
        all_metrics.append(
            evaluate_llm_on_project(project, frame, llm, PROJECT_WINDOW[project])
        )
        llm.save_cache()
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(RESULT_DIR, "summary.csv"), index=False
    )


if __name__ == "__main__":
    main()
