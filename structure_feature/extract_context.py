"""Extract affected method contexts from enriched CSV files."""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

import pandas as pd
from tqdm import tqdm

from tree_sitter import Language, Parser

PROJECTS = ["jdt", "go"]
SPLITS = ["train", "val", "test"]

DATA_BASE = "./dataset/enriched"
OUT_BASE = "./features/structure"

LANG_ALIASES = {
    "c#": "c_sharp",
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "c++": "cpp",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "js": "javascript",
    "ts": "javascript",
    "py": "python",
    "golang": "go",
}

FUNC_NODE_TYPES: Dict[str, Set[str]] = {
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "cpp": {"function_definition"},
    "c": {"function_definition"},
    "c_sharp": {"method_declaration", "constructor_declaration"},
    "python": {"function_definition"},
    "javascript": {
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function",
    },
    "php": {"function_definition", "method_declaration"},
    "ruby": {"method", "singleton_method"},
}


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_lang(lang: str) -> str:
    if not isinstance(lang, str):
        return "unknown"
    key = lang.strip().lower()
    return LANG_ALIASES.get(key, key)


def load_language(langlib_path: str, lang: str) -> Language:

    try:
        return Language(langlib_path, lang)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load from {langlib_path} language='{lang}', "
            f"Check that the shared library includes this language. Error: {e}"
        )


def build_parser(langlib_path: str, lang: str) -> Tuple[Parser, Language]:
    language = load_language(langlib_path, lang)
    parser = Parser()
    parser.set_language(language)
    return parser, language


def read_existing_ids(jsonl_path: str) -> Set[str]:

    existing = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    mid = f"{item.get('_id')}@{item.get('method_index', 0)}"
                    existing.add(mid)
                except Exception:
                    continue
    return existing


HUNK_RE = re.compile(r"^@@\s*-(\d+),?(\d*)\s+\+(\d+),?(\d*)\s*@@")


def parse_unified_diff_old_lines(diff_text: str) -> Set[int]:

    affected_old_lines: Set[int] = set()
    if not diff_text or "@@" not in diff_text:
        return affected_old_lines

    lines = diff_text.splitlines()
    old_line = None
    new_line = None

    for line in lines:
        m = HUNK_RE.match(line)
        if m:

            old_start = int(m.group(1))
            old_len = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_len = int(m.group(4) or "1")
            old_line = old_start
            new_line = new_start
            continue

        if old_line is None or new_line is None:
            continue

        if line.startswith(" "):
            old_line += 1
            new_line += 1
        elif line.startswith("-"):
            affected_old_lines.add(old_line)
            old_line += 1
        elif line.startswith("+"):

            if old_line > 0:
                affected_old_lines.add(max(1, old_line))
            new_line += 1
        else:

            pass

    return affected_old_lines


def node_text(source_bytes: bytes, node) -> bytes:
    return source_bytes[node.start_byte : node.end_byte]


def point_in_node(row0: int, node) -> bool:

    sr, _ = node.start_point
    er, _ = node.end_point
    return (row0 >= sr) and (row0 <= er)


def smallest_enclosing_function(node, lang: str) -> Optional[object]:

    fn_types = FUNC_NODE_TYPES.get(lang, set())
    cur = node
    while cur is not None:
        if cur.type in fn_types:
            return cur
        cur = cur.parent
    return None


def collect_function_nodes_for_lines(
    source: str, parser: Parser, lang: str, target_lines_1based: Set[int]
) -> List[Tuple[object, str]]:

    source_bytes = source.encode("utf-8", errors="ignore")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    selected = []
    seen_ranges = set()

    def dfs(node):

        sr, _ = node.start_point
        er, _ = node.end_point

        for ln1 in target_lines_1based:
            row0 = ln1 - 1
            if row0 < sr:
                continue
            if row0 > er:
                continue

            fn = smallest_enclosing_function(node, lang)
            if fn is not None:
                key = (fn.start_byte, fn.end_byte)
                if key not in seen_ranges:
                    code = node_text(source_bytes, fn).decode("utf-8", errors="ignore")
                    selected.append((fn, code))
                    seen_ranges.add(key)
            break

        for ch in node.children:
            dfs(ch)

    dfs(root)
    return selected


def process_split(
    project: str, split: str, langlib_path: str, limit: Optional[int] = None
):
    in_csv = os.path.join(DATA_BASE, f"{project}_{split}.csv")
    out_dir = os.path.join(OUT_BASE, f"{project}_{split}")
    ensure_dir(out_dir)
    out_jsonl = os.path.join(out_dir, "method_code.jsonl")

    if not os.path.exists(in_csv):
        print(f"[WARN] Missing dataset: {in_csv}")
        return

    df = pd.read_csv(in_csv)
    required = {"_id", "lang", "old_file", "old_code_diff", "repo_path", "file_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Enriched CSV missing columns: {sorted(missing)}")
    if limit is not None:
        df = df.head(limit)

    done_keys = read_existing_ids(out_jsonl)

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{project}-{split}"):
            _id = str(row.get("_id", ""))
            lang_raw = str(row.get("lang", "unknown"))
            lang = normalize_lang(lang_raw)

            old_file = row.get("old_file", "")
            diff_text = row.get("old_code_diff", "")

            repo_path = row.get("repo_path", "")
            file_path = row.get("file_path", "")

            if (
                not _id
                or not isinstance(old_file, str)
                or not isinstance(diff_text, str)
            ):
                continue

            if old_file.strip() == "":
                continue

            affected_old = parse_unified_diff_old_lines(diff_text)

            if not affected_old:
                continue

            try:
                parser, _ = build_parser(langlib_path, lang)
            except Exception as e:
                print(f"[LANG-ERR] {_id} lang={lang}: {e}")
                continue

            try:
                fn_nodes = collect_function_nodes_for_lines(
                    old_file, parser, lang, affected_old
                )
            except Exception as e:
                print(f"[AST-ERR] {_id}: {e}")
                fn_nodes = []

            if not fn_nodes:
                continue

            for idx, (_, code) in enumerate(fn_nodes):
                key = f"{_id}@{idx}"
                if key in done_keys:
                    continue
                rec = {
                    "_id": _id,
                    "repo_path": repo_path,
                    "file_path": file_path,
                    "lang": lang,
                    "method_index": idx,
                    "method_code": code,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract method-level context from enhanced JIT dataset using Tree-sitter."
    )
    ap.add_argument(
        "--projects", nargs="*", default=PROJECTS, help=f"Default: {PROJECTS}"
    )
    ap.add_argument("--splits", nargs="*", default=SPLITS, help=f"Default: {SPLITS}")
    ap.add_argument(
        "--langlib",
        type=str,
        default="my-languages.dll",
        help="Path to a compiled Tree-sitter shared library",
    )
    ap.add_argument("--limit", type=int, default=None, help="Maximum records per split")
    ap.add_argument("--data_dir", default=DATA_BASE)
    ap.add_argument("--output_dir", default=OUT_BASE)
    return ap.parse_args()


def main():
    global DATA_BASE, OUT_BASE
    args = parse_args()
    DATA_BASE, OUT_BASE = args.data_dir, args.output_dir
    langlib_path = args.langlib
    if not os.path.exists(langlib_path):
        raise FileNotFoundError(
            f"Tree-sitter library not found: {langlib_path}, provide the library path with --langlib."
        )

    for prj in args.projects:
        for sp in args.splits:
            process_split(prj, sp, langlib_path, limit=args.limit)

    print(
        "\n[OK] Context extraction complete. Output: ./features/structure/{project}_{split}/method_code.jsonl"
    )


if __name__ == "__main__":
    main()
