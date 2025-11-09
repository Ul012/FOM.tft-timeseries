# src/utils/json_results.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def _detect_lr_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if "learning_rate" in lc or lc.startswith("lr") or "lr-" in lc:
            return c
    return None


def export_run_jsons_from_metrics(
    run_id: str,
    logs_run_dir: Path,
    results_dir: Path,          # z. B. Path("results")/"evaluation"/run_id
    meta: dict | None = None,   # beliebige Zusatzinfos (cfg, fit_time, etc.)
) -> tuple[Path, Path]:
    """
    Liest logs/<...>/<run_id>/metrics.csv und erzeugt:
      results/evaluation/<run_id>/results.json   (epocheweise)
      results/evaluation/<run_id>/summary.json   (aggregiert)
    """
    metrics_csv = logs_run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv nicht gefunden: {metrics_csv}")

    df = pd.read_csv(metrics_csv).dropna(how="all", axis=0).reset_index(drop=True)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    # pro Epoche die letzte Zeile (robust ggü. Step-Logs)
    df_epoch = df.groupby("epoch").last().reset_index()

    # Spalten erkennen
    train_loss = None
    if "train_loss_epoch" in df_epoch.columns:
        train_loss = "train_loss_epoch"
    elif "train_loss_step" in df_epoch.columns:
        train_loss = "train_loss_step"

    val_loss = "val_loss" if "val_loss" in df_epoch.columns else None

    lr_col = _detect_lr_col(df)
    if lr_col and lr_col not in df_epoch.columns and lr_col in df.columns:
        df_lr = df.groupby("epoch")[[lr_col]].last().reset_index()
        df_epoch = df_epoch.merge(df_lr, on="epoch", how="left")

    # results.json – epocheweise Liste
    results = []
    for _, row in df_epoch.iterrows():
        item = {"epoch": int(row["epoch"])}
        if train_loss:
            item["train_loss"] = float(row[train_loss])
        if val_loss:
            item["val_loss"] = float(row[val_loss])
        if lr_col and lr_col in df_epoch.columns and pd.notna(row[lr_col]):
            item["learning_rate"] = float(row[lr_col])
        results.append(item)

    # summary.json – kompakt
    summary = {
        "run_id": run_id,
        "logs_dir": str(logs_run_dir),
        "n_epochs": int(df_epoch["epoch"].max()) + 1 if len(df_epoch) else 0,
        "metrics": {},
        "meta": meta or {},
    }

    # Final-/Best-Werte
    final_row = df_epoch.iloc[-1]
    if train_loss:
        summary["metrics"]["final_train_loss"] = float(final_row[train_loss])
    if val_loss:
        summary["metrics"]["final_val_loss"] = float(final_row[val_loss])
        # best val
        if df_epoch[val_loss].notna().any():
            idx = df_epoch[val_loss].astype(float).idxmin()
            summary["metrics"]["best_val_loss"] = float(df_epoch.loc[idx, val_loss])
            summary["metrics"]["best_val_epoch"] = int(df_epoch.loc[idx, "epoch"])

    if lr_col and lr_col in df_epoch.columns:
        try:
            summary["metrics"]["initial_lr"] = float(df_epoch.loc[df_epoch["epoch"].min(), lr_col])
            summary["metrics"]["final_lr"] = float(df_epoch.loc[df_epoch["epoch"].max(), lr_col])
        except Exception:
            pass

    # schreiben (ohne zweiten Zeitstempel)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    summary_path = results_dir / "summary.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results_path, summary_path
