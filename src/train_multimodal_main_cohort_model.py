#!/usr/bin/env python
# coding: utf-8

"""
Multimodal fusion training script for the main cohort.

Highlights:
- Trains ECG + tabular fusion model for 24h/48h/72h mortality tasks.
- Uses validation AUPRC for model selection and early stopping.
- Aggregates multiple runs and writes only three output CSV files:
  `proba_train.csv`, `proba_val.csv`, and `proba_test.csv`.

Example:
  python src/train_multimodal_main_cohort_model.py \\
    --csv data/ecg_multi.csv \\
    --npz data/multi_ecg_x.npz \\
    --out_dir outputs/ensemble2 \\
    --n_runs 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tsai.all import InceptionTime


@dataclass(frozen=True)
class TrainConfig:
    csv_path: str = "data/ecg_multi.csv"
    npz_path: str = "data/multi_ecg_x.npz"
    out_dir: str = "outputs/ensemble2"

    mort_idx: int = 2  # 0=24h, 1=48h, 2=72h
    use_tab: bool = True

    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-4
    t_max: int = 100
    eta_min: float = 1e-6

    focal_gamma: float = 2.0
    pos_weight_cap: float = 150.0

    seed: int = 20240901
    n_runs: int = 10

    # Checkpoint policy: select best by validation AUPRC and keep latest checkpoint.
    early_stop_patience: int = 5
    auprc_min_delta: float = 0.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def assert_out_dir_not_on_autodl_fs(out_dir: str) -> None:
    """Prevent writing outputs to read-only /autodl-fs in some AutoDL setups."""
    norm = os.path.abspath(os.path.expanduser(out_dir)).rstrip(os.sep) or "/"
    if norm == "/autodl-fs" or norm.startswith("/autodl-fs/"):
        raise SystemExit(
            "Invalid out_dir: writing under /autodl-fs is blocked (often read-only).\n"
            "Please use a writable directory, for example:\n"
            "  --out_dir outputs/ensemble2\n"
            "Input data paths via --csv and --npz can still point to /autodl-fs/data/..."
        )


def require_input_file(path: str, kind: str) -> str:
    """Validate required input file existence with readable diagnostics."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p):
        return p
    parent = os.path.dirname(p) or "."
    extra = ""
    if not os.path.exists(parent):
        extra = f"\nParent directory does not exist: {parent!r}."
    elif os.path.isdir(parent):
        try:
            names = sorted(os.listdir(parent))
            show = names if len(names) <= 30 else names[:30] + ["..."]
            extra = f"\nDirectory listing for {parent!r}: {show}"
        except OSError as e:
            extra = f"\nCannot read directory {parent!r}: {e}"
    raise SystemExit(
        f"Missing {kind} input file: {p}{extra}\n"
        "Please provide valid local paths, e.g., --csv /path/to/ecg_multi.csv --npz /path/to/multi_ecg_x.npz"
    )


def safe_index_by_file_name(
    df: pd.DataFrame,
    file_name_all: np.ndarray,
    *,
    fold_col: str = "general_strat_fold",
    file_col: str = "file_name",
) -> tuple[tuple[list[int], list[str]], tuple[list[int], list[str]], tuple[list[int], list[str]]]:
    train_ids = df.loc[df[fold_col].isin(range(0, 7)), file_col].astype(str).tolist()
    val_ids = df.loc[df[fold_col].isin([7]), file_col].astype(str).tolist()
    test_ids = df.loc[df[fold_col].isin(range(8, 10)), file_col].astype(str).tolist()

    fname2idx = {str(fn): i for i, fn in enumerate(file_name_all)}

    def _map(ids: Iterable[str], split: str) -> tuple[list[int], list[str]]:
        idx, fns, miss = [], [], 0
        for fn in ids:
            j = fname2idx.get(fn)
            if j is None:
                miss += 1
                continue
            idx.append(j)
            fns.append(fn)
        if miss > 0:
            print(f"[WARN] {split}: {miss} file_name entries were not found in npz.file_name and were skipped.")
        return idx, fns

    return _map(train_ids, "train"), _map(val_ids, "val"), _map(test_ids, "test")


class ECGTabDataset(Dataset):
    """Dataset containing ECG (12 leads), tabular features, and 24/48/72h labels."""

    def __init__(self, x_ecg: np.ndarray, x_tab: np.ndarray, y_multi: np.ndarray):
        self.x_ecg = torch.as_tensor(x_ecg, dtype=torch.float32)
        self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
        self.y = torch.as_tensor(y_multi, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x_ecg.shape[0])

    def __getitem__(self, i: int):
        return {"ecg": self.x_ecg[i], "tab": self.x_tab[i], "y": self.y[i]}


class TabMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 64, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalInception(nn.Module):
    """InceptionTime ECG encoder + tabular MLP branch with single-logit output."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_tab_in: int,
        *,
        d_ecg_out: int = 256,
        d_tab_out: int = 64,
        nf: int = 32,
        dropout_head: float = 0.1,
    ):
        super().__init__()
        self.ecg_encoder = InceptionTime(c_in=c_in, c_out=d_ecg_out, seq_len=seq_len, nf=nf, residual=True)
        self.tab_encoder = TabMLP(d_in=d_tab_in, d_hidden=d_tab_out, d_out=d_tab_out, p=0.1)
        self.head = nn.Sequential(nn.Dropout(dropout_head), nn.Linear(d_ecg_out + d_tab_out, 1))

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        z_ecg = self.ecg_encoder(x_ecg)
        z_tab = self.tab_encoder(x_tab.float())
        z = torch.cat([z_ecg, z_tab], dim=-1)
        return self.head(z)


class FocalLossWithLogits(nn.Module):
    """Binary focal loss implemented on logits, with optional pos_weight."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be none/mean/sum")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - pt).clamp(min=1e-6).pow(self.gamma)
        loss = focal_factor * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    train: bool,
    device: torch.device,
    mort_idx: int,
    use_tab: bool,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_cuda = device.type == "cuda"

    if train:
        assert optimizer is not None and scaler is not None
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        x = batch["ecg"].to(device)
        x_tab = batch["tab"].to(device)
        y_all = batch["y"].to(device)
        y = y_all[:, mort_idx].float()
        if y.ndim == 1:
            y = y.unsqueeze(1)

        with amp.autocast(device_type, enabled=use_cuda, dtype=torch.float32):
            logits = model(x, x_tab) if use_tab else model(x)
            loss = criterion(logits, y)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).long()

        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((pred.squeeze(1) == y.squeeze(1).long()).sum().item())
        total_n += int(x.size(0))

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


@torch.no_grad()
def collect_probas(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mort_idx: int,
    use_tab: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, proba) in DataLoader order; no AUC/threshold post-processing here."""
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_cuda = device.type == "cuda"
    with amp.autocast(device_type, enabled=use_cuda, dtype=torch.float32):
        for batch in loader:
            x = batch["ecg"].to(device)
            x_tab = batch["tab"].to(device)
            y_all = batch["y"].to(device)
            y = y_all[:, mort_idx].long()
            logits = model(x, x_tab) if use_tab else model(x)
            prob1 = torch.sigmoid(logits).squeeze(1)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob1.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _label_suffix(mort_idx: int) -> str:
    return {0: "death_24h", 1: "death_48h", 2: "death_72h"}.get(mort_idx, f"task_{mort_idx}")


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute AUPRC; return None if metric is not well-defined."""
    try:
        from sklearn.metrics import average_precision_score

        y_true = np.asarray(y_true).astype(int).ravel()
        y_prob = np.asarray(y_prob).astype(np.float64).ravel()
        if len(np.unique(y_true)) <= 1:
            return None
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def proba_ensemble_dataframe(
    file_names: list[str],
    y_true: np.ndarray,
    prob_matrix: np.ndarray,
    label_col: str,
    subject_ids: list | None = None,
    *,
    run_prefix: str = "proba_run",
) -> pd.DataFrame:
    """Build a per-split ensemble probability table, shape (n_samples, n_runs)."""
    cols: dict[str, object] = {"file_name": file_names, label_col: y_true.astype(np.int64)}
    if subject_ids is not None:
        cols["subject_id"] = subject_ids
    n_runs = int(prob_matrix.shape[1])
    for i in range(n_runs):
        cols[f"{run_prefix}{i}"] = prob_matrix[:, i].astype(np.float64)
    cols["proba_mean"] = prob_matrix.mean(axis=1).astype(np.float64)
    cols["proba_std"] = prob_matrix.std(axis=1).astype(np.float64)
    return pd.DataFrame(cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", dest="csv_path", default=TrainConfig.csv_path)
    parser.add_argument("--npz", dest="npz_path", default=TrainConfig.npz_path)
    parser.add_argument("--out_dir", dest="out_dir", default=TrainConfig.out_dir)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--n_runs",
        type=int,
        default=TrainConfig.n_runs,
        help="Number of ensemble runs; final outputs are three CSV files in out_dir with all proba_run* columns merged by split.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=TrainConfig.early_stop_patience,
        help="Early-stop patience based on validation AUPRC (recommended >=5).",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        csv_path=args.csv_path,
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        n_runs=args.n_runs,
        early_stop_patience=args.patience,
    )

    assert_out_dir_not_on_autodl_fs(cfg.out_dir)
    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("[Env] torch:", torch.__version__, "| device:", device)

    csv_path = require_input_file(cfg.csv_path, "CSV")
    npz_path = require_input_file(cfg.npz_path, "NPZ")
    df = pd.read_csv(csv_path)
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["arr_0"]
    file_name_all = data["file_name"]

    # Tabular split indices (already included in npz)
    x_train_tab = data["x_train"]
    x_val_tab = data["x_val"]
    x_test_tab = data["x_test"]

    # Labels (already included in npz)
    y_train_24, y_val_24, y_test_24 = data["y_train_24_death"], data["y_val_24_death"], data["y_test_24_death"]
    y_train_48, y_val_48, y_test_48 = data["y_train_48_death"], data["y_val_48_death"], data["y_test_48_death"]
    y_train_72, y_val_72, y_test_72 = data["y_train_72_death"], data["y_val_72_death"], data["y_test_72_death"]

    y_train_multi = np.stack([y_train_24, y_train_48, y_train_72], axis=1)
    y_val_multi = np.stack([y_val_24, y_val_48, y_val_72], axis=1)
    y_test_multi = np.stack([y_test_24, y_test_48, y_test_72], axis=1)

    (idx_train, fn_train), (idx_val, fn_val), (idx_test, fn_test) = safe_index_by_file_name(df, file_name_all)
    x_ecg_train, x_ecg_val, x_ecg_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    print("[Data] ECG:", x_ecg_train.shape, x_ecg_val.shape, x_ecg_test.shape)

    train_ds = ECGTabDataset(x_ecg_train, x_train_tab, y_train_multi)
    val_ds = ECGTabDataset(x_ecg_val, x_val_tab, y_val_multi)
    test_ds = ECGTabDataset(x_ecg_test, x_test_tab, y_test_multi)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    batch = next(iter(train_loader))
    x_ecg, x_tab, y_all = batch["ecg"], batch["tab"], batch["y"]
    B, C, L = x_ecg.shape
    D_TAB = x_tab.shape[1]
    assert C == 12 and L == 1000, f"ECG shape must be (B,12,1000), got {(B,C,L)}"
    assert y_all.ndim == 2 and y_all.shape[1] == 3, f"y shape must be (B,3), got {tuple(y_all.shape)}"

    print(f"[Sanity] batch ecg/tab/y: {(B,C,L)} / {tuple(x_tab.shape)} / {tuple(y_all.shape)}")
    print(f"[Task] mortality@{[24,48,72][cfg.mort_idx]}h | pos_rate={float(y_all[:, cfg.mort_idx].float().mean()):.4f}")

    label_col = _label_suffix(cfg.mort_idx)
    fn2subj: dict[str, object] | None = None
    if "subject_id" in df.columns and "file_name" in df.columns:
        fn2subj = dict(zip(df["file_name"].astype(str), df["subject_id"]))

    def _subj(fns: list[str]) -> list | None:
        if fn2subj is None:
            return None
        return [fn2subj.get(fn, np.nan) for fn in fns]

    # Estimate pos_weight from 72h training labels and cap for numerical stability.
    y72 = torch.as_tensor(y_train_72, device=device)
    p_cnt = int((y72 == 1).sum().item())
    n_cnt = int((y72 == 0).sum().item())
    ratio = min(n_cnt / max(p_cnt, 1), cfg.pos_weight_cap)
    pos_weight = torch.tensor([ratio], device=device, dtype=torch.float32)
    criterion_train = FocalLossWithLogits(gamma=cfg.focal_gamma, pos_weight=pos_weight, reduction="mean")
    criterion_val = FocalLossWithLogits(gamma=cfg.focal_gamma, pos_weight=None, reduction="mean")

    print(f"[Loss] focal_gamma={cfg.focal_gamma} | pos_weight(train)={float(pos_weight.item()):.2f} (n/p={n_cnt}/{p_cnt})")

    train_probs_runs: list[np.ndarray] = []
    val_probs_runs: list[np.ndarray] = []
    test_probs_runs: list[np.ndarray] = []
    y_tr_ref: np.ndarray | None = None
    y_va_ref: np.ndarray | None = None
    y_te_ref: np.ndarray | None = None

    for run in range(cfg.n_runs):
        run_dir = os.path.join(cfg.out_dir, f"run_{run:02d}")
        ensure_dir(run_dir)
        set_seed(cfg.seed + run)

        model: nn.Module
        if cfg.use_tab:
            model = MultiModalInception(c_in=C, seq_len=L, d_tab_in=D_TAB).to(device)
        else:
            model = InceptionTime(c_in=C, c_out=1, seq_len=L, nf=32, residual=True).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=cfg.eta_min)
        scaler = amp.GradScaler(device=("cuda" if device.type == "cuda" else "cpu"))

        # Track best checkpoint by validation AUPRC and use patience-based early stop.
        best_val_auprc = -float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(run_dir, "best_by_val_auprc.ckpt")
        latest_path = os.path.join(run_dir, "latest.ckpt")

        for epoch in range(1, cfg.t_max + 1):
            t0 = time.time()
            tr_loss, tr_acc = one_epoch(
                model,
                train_loader,
                train=True,
                device=device,
                mort_idx=cfg.mort_idx,
                use_tab=cfg.use_tab,
                criterion=criterion_train,
                optimizer=optimizer,
                scaler=scaler,
            )
            val_loss, val_acc = one_epoch(
                model,
                val_loader,
                train=False,
                device=device,
                mort_idx=cfg.mort_idx,
                use_tab=cfg.use_tab,
                criterion=criterion_val,
            )
            scheduler.step()

            # Compute validation AUPRC from model probabilities on val split.
            y_true_v, y_prob_v = collect_probas(model, val_loader, device, cfg.mort_idx, cfg.use_tab)
            val_auprc = compute_auprc(y_true_v, y_prob_v)

            torch.save({"epoch": epoch, "model_state": model.state_dict()}, latest_path)
            improved = (
                val_auprc is not None
                and (float(val_auprc) - cfg.auprc_min_delta) > best_val_auprc
            )
            if improved:
                best_val_auprc = float(val_auprc)
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_auprc": best_val_auprc,
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1

            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            auprc_str = f"{val_auprc:.4f}" if val_auprc is not None else "nan"
            print(
                f"[run {run:02d} | epoch {epoch:03d}] "
                f"train loss/acc {tr_loss:.4f}/{tr_acc:.4f} | "
                f"val loss/acc {val_loss:.4f}/{val_acc:.4f} | "
                f"val_auprc {auprc_str} (best {best_val_auprc:.4f}) | "
                f"lr {lr_now:.2e} | {dt:.1f}s"
            )

            if epochs_no_improve >= cfg.early_stop_patience:
                print(
                    f"[run {run:02d}][EarlyStop] val AUPRC did not improve for "
                    f"{cfg.early_stop_patience} epochs; stopping at epoch {epoch}."
                )
                break

        if not os.path.isfile(best_path):
            print(f"[WARN run {run:02d}] best-by-AUPRC checkpoint missing; using latest.ckpt.")
            shutil.copy2(latest_path, best_path)

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        y_tr, p_tr = collect_probas(model, train_loader, device, cfg.mort_idx, cfg.use_tab)
        y_va, p_va = collect_probas(model, val_loader, device, cfg.mort_idx, cfg.use_tab)
        y_te, p_te = collect_probas(model, test_loader, device, cfg.mort_idx, cfg.use_tab)

        if y_tr_ref is None:
            y_tr_ref, y_va_ref, y_te_ref = y_tr, y_va, y_te
        else:
            if not (np.array_equal(y_tr_ref, y_tr) and np.array_equal(y_va_ref, y_va) and np.array_equal(y_te_ref, y_te)):
                raise RuntimeError("Inconsistent y_true across runs; please verify DataLoader order.")

        train_probs_runs.append(p_tr.astype(np.float64))
        val_probs_runs.append(p_va.astype(np.float64))
        test_probs_runs.append(p_te.astype(np.float64))

        print(f"[DONE run {run:02d}] best ckpt: {best_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert y_tr_ref is not None and y_va_ref is not None and y_te_ref is not None
    P_tr = np.stack(train_probs_runs, axis=1)
    P_va = np.stack(val_probs_runs, axis=1)
    P_te = np.stack(test_probs_runs, axis=1)

    sub_tr, sub_va, sub_te = _subj(fn_train), _subj(fn_val), _subj(fn_test)

    path_train = os.path.join(cfg.out_dir, "proba_train.csv")
    path_val = os.path.join(cfg.out_dir, "proba_val.csv")
    path_test = os.path.join(cfg.out_dir, "proba_test.csv")

    proba_ensemble_dataframe(fn_train, y_tr_ref, P_tr, label_col, sub_tr).to_csv(path_train, index=False)
    proba_ensemble_dataframe(fn_val, y_va_ref, P_va, label_col, sub_va).to_csv(path_val, index=False)
    proba_ensemble_dataframe(fn_test, y_te_ref, P_te, label_col, sub_te).to_csv(path_test, index=False)

    print(f"[SAVE] Aggregated {cfg.n_runs} runs into three CSV files:")
    print(f"  {path_train}")
    print(f"  {path_val}")
    print(f"  {path_test}")


if __name__ == "__main__":
    main()
