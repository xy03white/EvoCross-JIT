"""Extract syntax tokens and data-flow edges from method contexts."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Set, Dict, Any, Tuple, List

from tqdm import tqdm

from tree_sitter import Language, Parser

INPUT_BASE = "./features/structure"

DEFAULT_LANGLIB = "./structure_feature/vendor/build/my-languages.dll"

DEFAULT_PROJECTS = ["jdt", "go"]
DEFAULT_SPLITS = ["train", "val", "test"]

current_dir = Path(__file__).parent

parser_dir = current_dir / "code_parser"
sys.path.insert(0, str(parser_dir))

try:

    from code_parser.utils import (
        remove_comments_and_docstrings,
        tree_to_token_index,
        index_to_code_token,
    )

    from code_parser.DFG import (
        DFG_python,
        DFG_java,
        DFG_ruby,
        DFG_go,
        DFG_php,
        DFG_javascript,
        DFG_csharp,
    )
except Exception as e:

    raise ImportError(
        "Cannot import code_parser. Check the source directory and dependencies.\n"
        "Required functions: remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, DFG_python, DFG_java, ...\n"
        f"Details: {e}"
    ) from e

DFG_FUNCTION = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
    "cpp": DFG_java,
    "c": DFG_python,
}


def build_parsers(langlib_path: str) -> Dict[str, List]:

    parsers = {}
    for lang in DFG_FUNCTION.keys():
        try:
            LANGUAGE = Language(langlib_path, lang)
            parser = Parser()
            parser.set_language(LANGUAGE)
            parsers[lang] = [parser, DFG_FUNCTION[lang]]
        except Exception as e:

            print(
                f"[WARN] Cannot load language '{lang}' Tree-sitter parser (langlib={langlib_path}): {e}"
            )
    return parsers


def extract_dataflow(code: str, parser_pair, lang: str) -> Tuple[List[str], List]:

    try:

        try:
            code = remove_comments_and_docstrings(code, lang)
        except TypeError:
            try:
                code = remove_comments_and_docstrings(code, lang=lang)
            except Exception:

                try:
                    code = remove_comments_and_docstrings(code)
                except Exception:
                    pass

        if lang == "php":
            code = "<?php" + code + "?>"
        try:
            tree = parser_pair[0].parse(bytes(code, "utf8"))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code_lines = code.split("\n")

            code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code_token) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code_token)
            try:

                DFG, _ = parser_pair[1](root_node, index_to_code, {})
            except Exception:

                DFG = []

            DFG = sorted(DFG, key=lambda x: x[1])
            indexs = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg = new_DFG
        except Exception:
            dfg = []
            code_tokens = []
    except Exception:
        dfg = []
        code_tokens = []
    return code_tokens, dfg


def read_done_keys(path: str) -> Set[str]:
    s = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as fin:
                for ln in fin:
                    try:
                        obj = json.loads(ln)
                        mid = f"{obj.get('_id')}@{obj.get('method_index', 0)}"
                        s.add(mid)
                    except Exception:
                        continue
        except Exception:
            pass
    return s


def process_split(
    project: str,
    split: str,
    langlib_path: str,
    projects_parsers: Dict[str, List],
    limit: int = None,
):
    in_dir = os.path.join(INPUT_BASE, f"{project}_{split}")
    in_jsonl = os.path.join(in_dir, "method_code.jsonl")
    out_jsonl = os.path.join(in_dir, "ast_dfg.jsonl")

    if not os.path.exists(in_jsonl):
        print(f"[WARN] Input does not exist: {in_jsonl}; skipping")
        return

    ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
    ensure_dir(in_dir)

    done_keys = read_done_keys(out_jsonl)

    total = 0
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    processed = 0
    written = 0
    with open(in_jsonl, "r", encoding="utf-8") as fin, open(
        out_jsonl, "a", encoding="utf-8"
    ) as fout:
        pbar = tqdm(fin, total=total, desc=f"{project}-{split}")
        for line in pbar:
            if limit is not None and processed >= limit:
                break
            processed += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            _id = str(rec.get("_id", ""))
            method_index = (
                int(rec.get("method_index", 0))
                if rec.get("method_index") is not None
                else 0
            )
            raw_lang = rec.get("lang", "")

            lang = str(raw_lang).strip().lower()
            if lang == "c#":
                lang = "c_sharp"
            if lang == "c++":
                lang = "cpp"

            code = (
                rec.get("method_code")
                or rec.get("complete")
                or rec.get("old_file")
                or ""
            )
            if not code or not _id:
                continue

            key = f"{_id}@{method_index}"
            if key in done_keys:
                pbar.set_postfix_str(f"skip {_id}")
                continue

            parser_pair = projects_parsers.get(lang)
            if parser_pair is None:

                print(f"[SKIP] {_id} lang={lang} unsupported or parser not loaded")
                continue

            try:
                code_tokens, dfg = extract_dataflow(code, parser_pair, lang)
            except Exception as e:
                print(f"[ERR] extract_dataflow failed for {_id} lang={lang}: {e}")
                code_tokens, dfg = [], []

            out_obj = {
                "_id": _id,
                "method_index": method_index,
                "lang": lang,
                "code_tokens": code_tokens,
                "dfg": dfg,
            }

            for k in ("repo_path", "file_path"):
                if k in rec and k not in out_obj:
                    out_obj[k] = rec[k]

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

        pbar.close()

    print(
        f"[DONE] {project}-{split} processed={processed}, written={written}, out={out_jsonl}"
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract AST+DFG (GraphCodeBERT inputs) from method_code.jsonl from the supplied parser implementation."
    )
    ap.add_argument(
        "--projects", nargs="+", default=DEFAULT_PROJECTS, help="Project names"
    )
    ap.add_argument(
        "--splits", nargs="+", default=DEFAULT_SPLITS, help="Splits: train val test"
    )
    ap.add_argument(
        "--langlib",
        type=str,
        default=DEFAULT_LANGLIB,
        help="Path to the Tree-sitter multi-language shared library",
    )
    ap.add_argument("--limit", type=int, default=None, help="Maximum records per split")
    ap.add_argument("--input_dir", default=INPUT_BASE)
    return ap.parse_args()


def main():
    global INPUT_BASE
    args = parse_args()
    INPUT_BASE = args.input_dir
    langlib_path = args.langlib
    if not os.path.exists(langlib_path):
        raise FileNotFoundError(
            f"Tree-sitter library not found: {langlib_path}; provide a valid --langlib path"
        )

    parsers = build_parsers(langlib_path)
    if not parsers:
        raise RuntimeError("No language parsers loaded; check the supplied library")

    for prj in args.projects:
        for sp in args.splits:
            process_split(prj, sp, langlib_path, parsers, limit=args.limit)

    print("[ALL DONE] extract_ast_dfg finished.")


if __name__ == "__main__":
    main()
