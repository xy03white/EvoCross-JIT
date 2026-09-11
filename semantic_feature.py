"""Project-specific CodeBERT training and semantic embedding extraction."""

import os

# This pipeline uses PyTorch; do not initialize optional TensorFlow backends.
os.environ.setdefault("USE_TF", "0")
import ast
import argparse
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_added_removed(code_str: str):
    added_lines, removed_lines = [], []
    try:
        code_list = ast.literal_eval(code_str)
        for entry in code_list:
            if not isinstance(entry, str):
                continue
            if "added_code:" in entry and "removed_code:" in entry:
                parts = entry.split("removed_code:")
                added = parts[0].replace("added_code:", "").strip()
                removed = parts[1].strip()
                if added:
                    added_lines.append(added)
                if removed:
                    removed_lines.append(removed)
    except Exception:
        return "", ""
    return " ".join(added_lines), " ".join(removed_lines)


class CommitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RobertaTokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        _id = row["_id"]
        msg = str(row.get("msg", "") or "")
        code_field = str(row.get("code", "") or "")
        label = int(row.get("label", 0))

        add_part, del_part = parse_added_removed(code_field)
        text = f"[CLS] {msg} [ADD] {add_part} [DEL] {del_part}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "id": _id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.float),
        }


class RobertaBinaryClassifierWithContrastive(nn.Module):
    def __init__(self, encoder: RobertaModel, hidden_size=768, dropout_prob=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.out(cls).squeeze(-1)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        if labels is None:
            return probs, cls

        loss_bce = F.binary_cross_entropy(probs, labels)
        return loss_bce, probs, cls

    @staticmethod
    def supervised_contrastive_loss(cls_features, labels, temperature=0.7):
        """Preserve the supplied ratio loss; see IMPLEMENT.md for its Eq. (5) mismatch."""

        if torch.isnan(cls_features).any() or torch.isinf(cls_features).any():
            return torch.tensor(0.0, device=cls_features.device, requires_grad=True)

        labels = labels.long()

        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=cls_features.device, requires_grad=True)

        cls_features = F.normalize(cls_features, dim=1)
        similarity = torch.matmul(cls_features, cls_features.T) / temperature
        similarity = torch.clamp(similarity, min=-10.0, max=10.0)

        mask_pos = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask_pos.fill_diagonal_(0)
        mask_neg = 1 - mask_pos
        mask_neg.fill_diagonal_(0)

        exp_sim = torch.exp(similarity)
        pos_sum = torch.sum(exp_sim * mask_pos, dim=1)
        denom = torch.sum(exp_sim * mask_neg, dim=1) + 1e-8

        loss_per_sample = -torch.log(pos_sum / denom)
        valid_mask = (torch.sum(mask_pos, dim=1) > 0).float()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=cls_features.device, requires_grad=True)

        loss = torch.sum(loss_per_sample * valid_mask) / valid_mask.sum()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=cls_features.device, requires_grad=True)
        return loss


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            probs, _ = model(input_ids, attention_mask, labels=None)
            y_prob.extend(probs.detach().cpu().numpy().tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}


def train_for_project(args, project, tokenizer, device):
    logging.info(
        f"=== Train semantic model for project: {project} (contrastive={args.use_contrastive}) ==="
    )
    train_csv = os.path.abspath(os.path.join(args.data_dir, f"{project}_train.csv"))
    val_csv = os.path.abspath(os.path.join(args.data_dir, f"{project}_val.csv"))

    if not os.path.exists(train_csv):
        logging.error(f"Train file not found: {train_csv}")
        return None

    df_train = pd.read_csv(train_csv).fillna("")
    df_val = pd.read_csv(val_csv).fillna("")
    if df_val.empty:
        raise ValueError("Validation data must not be empty")

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_dataset = CommitDataset(df_train, tokenizer, max_len=args.max_seq_length)
    val_dataset = CommitDataset(df_val, tokenizer, max_len=args.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    config = RobertaConfig.from_pretrained(args.model_path)
    encoder = RobertaModel.from_pretrained(args.model_path, config=config)
    encoder.resize_token_embeddings(len(tokenizer))
    model = RobertaBinaryClassifierWithContrastive(encoder).to(device)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    total_steps = max(
        1, int(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    patience = 0
    best_ckpt_dir = os.path.abspath(
        os.path.join(args.checkpoint_dir, f"{project}_semantic")
    )
    os.makedirs(best_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(best_ckpt_dir, "checkpoint-best-f1.bin")

    global_step = 0
    for epoch in range(int(args.epochs)):
        logging.info(f"Project {project} - Epoch {epoch+1}/{args.epochs}")
        model.train()
        tr_loss = 0.0
        nb_tr_steps = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"{project}_train_epoch{epoch+1}")
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if isinstance(model, torch.nn.DataParallel):
                loss_bce, probs, cls_features = model.module(
                    input_ids, attention_mask, labels
                )
            else:
                loss_bce, probs, cls_features = model(input_ids, attention_mask, labels)

            if torch.isnan(loss_bce) or torch.isinf(loss_bce):
                logging.warning(f"NaN/Inf in BCE loss at step {step}. Skipping batch.")
                continue

            loss = loss_bce

            if args.use_contrastive and labels.size(0) > 1:

                if torch.isnan(cls_features).any() or torch.isinf(cls_features).any():
                    logging.warning(
                        f"NaN/Inf in cls_features at step {step}. Skipping contrastive part."
                    )
                else:
                    if isinstance(model, torch.nn.DataParallel):
                        loss_contrast = model.module.supervised_contrastive_loss(
                            cls_features, labels, temperature=args.temperature
                        )
                    else:
                        loss_contrast = model.supervised_contrastive_loss(
                            cls_features, labels, temperature=args.temperature
                        )

                    if not (torch.isnan(loss_contrast) or torch.isinf(loss_contrast)):
                        loss = loss_bce + args.contrastive_weight * loss_contrast
                    else:
                        logging.warning(
                            f"NaN/Inf in contrastive loss at step {step}. Using BCE only."
                        )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        if len(df_val) > 0:
            eval_metrics = evaluate_model(
                model.module if isinstance(model, torch.nn.DataParallel) else model,
                val_loader,
                device,
            )
            logging.info(f"Val metrics: {eval_metrics}")
            cur_f1 = eval_metrics["f1"]
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                patience = 0
                state_to_save = (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
                torch.save(
                    {"model_state_dict": state_to_save, "epoch": epoch, "f1": cur_f1},
                    best_ckpt_path,
                )
                logging.info(
                    f"Saved best checkpoint to {best_ckpt_path} (f1={cur_f1:.4f})"
                )
            else:
                patience += 1
                logging.info(f"No improvement. Patience {patience}/{args.patience}")
                if patience >= args.patience:
                    logging.info("Early stopping triggered.")
                    break
        else:
            state_to_save = (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                {"model_state_dict": state_to_save, "epoch": epoch, "f1": -1.0},
                best_ckpt_path,
            )
            logging.info("No val set; saved last epoch checkpoint.")
            break

    return best_ckpt_path if os.path.exists(best_ckpt_path) else None


def extract_semantic_features(model_path_ckpt, project, split, tokenizer, device, args):
    input_csv = os.path.abspath(os.path.join(args.data_dir, f"{project}_{split}.csv"))
    out_dir = os.path.abspath(os.path.join(args.output_dir, f"{project}_{split}"))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        logging.warning(f"Input csv not found: {input_csv}")
        return

    df = pd.read_csv(input_csv).fillna("")
    df = df.reset_index(drop=True)

    config = RobertaConfig.from_pretrained(args.model_path)
    encoder = RobertaModel.from_pretrained(args.model_path, config=config)
    encoder.resize_token_embeddings(len(tokenizer))
    model = RobertaBinaryClassifierWithContrastive(encoder).to(device)

    ckpt = torch.load(model_path_ckpt, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    dataset = CommitDataset(df, tokenizer, max_len=args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    errors_log = os.path.abspath("semantic_extraction_errors.log")
    for batch in tqdm(dataloader, desc=f"extract_{project}_{split}"):
        ids = batch["id"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            _, cls = model(input_ids, attention_mask, labels=None)
            cls_cpu = cls.detach().cpu()

        for i, _id in enumerate(ids):
            try:
                vec = cls_cpu[i].contiguous().to(torch.float32)
                label_int = int(labels[i].item())
                save_path = os.path.join(out_dir, f"{_id}.pt")
                torch.save({"feature": vec, "label": label_int}, save_path)
            except Exception as e:
                logging.warning(f"Failed saving {_id}: {e}")
                with open(errors_log, "a") as f:
                    f.write(f"{project},{split},{_id},{type(e).__name__},{str(e)}\n")

    logging.info(f"Extraction finished for {project}_{split}, saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="CodeBERT fine-tuning with supervised contrastive learning"
    )
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--checkpoint_dir", default="models")
    parser.add_argument("--output_dir", default="features/semantic")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--projects", nargs="+", required=True, help="Project list")
    parser.add_argument("--model_path", type=str, default="./codebert-base")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_contrastive",
        action="store_true",
        default=True,
        help="Enable supervised contrastive learning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
        help="Weight of contrastive loss (default 0.1 for stability)",
    )

    parser.add_argument(
        "--no_contrastive",
        dest="use_contrastive",
        action="store_false",
        help="Explicit semantic ablation; not the full model",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if args.use_contrastive:
        logging.info(
            f"Contrastive learning ENABLED: temp={args.temperature}, weight={args.contrastive_weight}"
        )
    else:
        logging.info("Contrastive learning DISABLED (BCE only)")

    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ADD]", "[DEL]"]})

    for project in args.projects:
        logging.info(f"Processing project: {project}")
        ckpt_path = train_for_project(args, project, tokenizer, device)
        if ckpt_path is None:
            logging.error(f"No checkpoint for {project}, skipping extraction.")
            continue

        for split in ["train", "val", "test"]:
            try:
                extract_semantic_features(
                    ckpt_path, project, split, tokenizer, device, args
                )
            except Exception as e:
                logging.error(f"Extraction failed for {project}_{split}: {e}")

    logging.info("All done.")


if __name__ == "__main__":
    main()
