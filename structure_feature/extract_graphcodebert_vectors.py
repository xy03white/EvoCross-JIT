"""Encode methods with GraphCodeBERT and aggregate commit vectors."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

PROJECTS = ["platform"]
SPLITS = ["train", "val", "test"]
INPUT_BASE = "./features/structure"

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
        "Cannot import code_parser. Check the source directory.\n" f"Details: {e}"
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

    from tree_sitter import Language, Parser

    parsers = {}
    for lang in DFG_FUNCTION.keys():
        try:
            LANGUAGE = Language(langlib_path, lang)
            parser = Parser()
            parser.set_language(LANGUAGE)
            parsers[lang] = [parser, DFG_FUNCTION[lang]]
        except Exception as e:
            print(f"[WARN] Cannot load language '{lang}' parser: {e}")
    return parsers


def extract_dataflow(code: str, parser_pair, lang: str) -> Tuple[List[str], List]:

    try:
        try:
            code = remove_comments_and_docstrings(code, lang)
        except TypeError:
            try:
                code = remove_comments_and_docstrings(code, lang=lang)
            except Exception:
                code = remove_comments_and_docstrings(code)
        if lang == "php":
            code = "<?php" + code + "?>"
        tree = parser_pair[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
        index_to_code = {
            idx: (i, tok) for i, (idx, tok) in enumerate(zip(tokens_index, code_tokens))
        }
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
        new_DFG = [d for d in DFG if d[1] in indexs]
        return code_tokens, new_DFG
    except Exception:
        return [], []


def build_graph_attention_mask(position_idx, code_ids, dfg_to_code, dfg_to_dfg):

    L = len(position_idx)
    attn_mask = np.zeros((L, L), dtype=np.bool_)
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    attn_mask[:node_index, :node_index] = True
    for idx, i in enumerate(code_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True
    return attn_mask


def convert_example_to_inputs(
    code_tokens, dfg, tokenizer, config_code_length=150, config_data_flow_length=64
):

    toks_processed = []
    for idx, tok in enumerate(code_tokens):
        tok_piece = (
            tokenizer.tokenize("@ " + tok)[1:] if idx != 0 else tokenizer.tokenize(tok)
        )
        toks_processed.append(tok_piece)
    toks_flat = [y for x in toks_processed for y in x]
    max_code_part = (
        config_code_length
        + config_data_flow_length
        - 2
        - min(len(dfg), config_data_flow_length)
    )
    toks_flat = toks_flat[:max_code_part]
    code_tokens_for_model = [tokenizer.cls_token] + toks_flat + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens_for_model)
    position_idx = [
        i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens_for_model))
    ]
    dfg_nodes = [x[0] for x in dfg][:config_data_flow_length]
    code_tokens_for_model += dfg_nodes
    position_idx += [0 for _ in dfg_nodes]
    code_ids += [tokenizer.unk_token_id for _ in dfg_nodes]
    padding_length = config_code_length + config_data_flow_length - len(code_ids)
    if padding_length > 0:
        position_idx += [tokenizer.pad_token_id] * padding_length
        code_ids += [tokenizer.pad_token_id] * padding_length
    reverse_index = {x[1]: i for i, x in enumerate(dfg[:config_data_flow_length])}
    dfg_to_dfg = [
        [reverse_index[i] for i in x[-1] if i in reverse_index]
        for x in dfg[:config_data_flow_length]
    ]
    ori2cur_pos = {-1: (0, 0)}
    for i, piece in enumerate(toks_processed):
        ori2cur_pos[i] = (
            (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(piece))
            if i > 0
            else (0, len(piece))
        )
    dfg_to_code = []
    for x in dfg[:config_data_flow_length]:
        a, b = ori2cur_pos.get(x[1], (0, 0))
        length_cls = 1
        dfg_to_code.append((a + length_cls, b + length_cls))
    return code_ids, position_idx, dfg_to_code, dfg_to_dfg


def process_single_split(project, split, model, tokenizer, parsers, args):

    input_dir = os.path.join(args.input_dir, f"{project}_{split}")
    in_jsonl = os.path.join(input_dir, "ast_dfg.jsonl")
    out_jsonl = os.path.join(input_dir, "ast_commit_vec.jsonl")

    if not os.path.exists(in_jsonl):
        print(f"[WARN] {in_jsonl} does not exist; skipping")
        return 0, 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    total = sum(1 for _ in open(in_jsonl, "r", encoding="utf-8"))
    commit_dict = defaultdict(list)

    with open(in_jsonl, "r", encoding="utf-8") as fin:
        pbar = tqdm(fin, total=total, desc=f"{project}-{split}")
        for line in pbar:
            try:
                rec = json.loads(line)
            except:
                continue
            _id = str(rec.get("_id", ""))
            method_index = int(rec.get("method_index", 0))
            lang = rec.get("lang", "").strip().lower()
            if lang == "c#":
                lang = "c_sharp"
            if lang == "c++":
                lang = "cpp"
            code_tokens = rec.get("code_tokens", [])
            dfg = rec.get("dfg", [])
            if (not code_tokens) and ("method_code" in rec or "complete" in rec):
                code_source = (
                    rec.get("method_code")
                    or rec.get("complete")
                    or rec.get("old_file", "")
                )
                parser_pair = parsers.get(lang)
                if parser_pair:
                    code_tokens, dfg = extract_dataflow(code_source, parser_pair, lang)

            code_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_example_to_inputs(
                code_tokens, dfg, tokenizer, args.code_length, args.data_flow_length
            )
            attn_mask = build_graph_attention_mask(
                position_idx, code_ids, dfg_to_code, dfg_to_dfg
            )
            code_ids_t = torch.tensor(
                code_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            position_idx_t = torch.tensor(
                position_idx, dtype=torch.long, device=device
            ).unsqueeze(0)
            attn_mask_t = torch.tensor(
                attn_mask.astype(np.int64), device=device
            ).unsqueeze(0)

            try:
                with torch.no_grad():
                    inputs_embeddings = model.embeddings.word_embeddings(code_ids_t)
                    nodes_mask = (
                        torch.tensor(position_idx, device=device) == 0
                    ).unsqueeze(0)
                    token_mask = (
                        torch.tensor(position_idx, device=device) >= 2
                    ).unsqueeze(0)
                    attn_mask_bool = torch.tensor(
                        attn_mask.astype(np.bool_), device=device
                    ).unsqueeze(0)
                    nodes_to_token_mask = (
                        nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask_bool
                    )
                    denom = nodes_to_token_mask.sum(-1, keepdim=True).float() + 1e-10
                    nodes_to_token_mask = nodes_to_token_mask.float() / denom
                    avg_embeddings = torch.einsum(
                        "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
                    )
                    inputs_embeddings = (
                        inputs_embeddings * (~nodes_mask)[:, :, None]
                        + avg_embeddings * nodes_mask[:, :, None]
                    )
                    out = model(
                        inputs_embeds=inputs_embeddings,
                        attention_mask=attn_mask_t,
                        position_ids=position_idx_t,
                    )
                    seq_vec = out[0].detach().cpu()
                    cls_vec = seq_vec[:, 0, :]
                    mean_vec = seq_vec.mean(dim=1)
                    method_vec = torch.cat([cls_vec, mean_vec], dim=-1).squeeze(0)
                    commit_dict[_id].append(method_vec)
            except Exception as e:
                print(f"[ERR] forward {_id}@{method_index} failed: {e}")

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for _id, vecs in commit_dict.items():
            method_vecs = torch.stack(vecs, dim=0)
            anchor = method_vecs.mean(dim=0, keepdim=True).T
            scores = torch.softmax(torch.matmul(method_vecs, anchor), dim=0)
            commit_vec = torch.sum(scores * method_vecs, dim=0)
            out_obj = {"_id": _id, "vec": commit_vec.tolist(), "num_methods": len(vecs)}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[DONE] {project}-{split}: commits={len(commit_dict)}")
    return len(commit_dict), len(commit_dict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", default=PROJECTS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--model_dir", default="./graphcodebert-base")
    ap.add_argument(
        "--langlib", default="./structure_feature/vendor/build/my-languages.so"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--code_length", type=int, default=150)
    ap.add_argument("--data_flow_length", type=int, default=64)
    ap.add_argument("--input_dir", default=INPUT_BASE)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Loading GraphCodeBERT...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = RobertaModel.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device).eval()
    parsers = build_parsers(args.langlib)

    total = 0
    for project in args.projects:
        for split in args.splits:
            print(f"\nProcessing: {project}-{split}")
            c, _ = process_single_split(project, split, model, tokenizer, parsers, args)
            total += c
    print(f"\n[ALL DONE] Total commits={total}")


if __name__ == "__main__":
    main()
