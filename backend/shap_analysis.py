"""SHAP feature-channel importance analysis for SilverBullet checkpoints.

Uses shap.GradientExplainer to attribute model output scores back to each of the F
feature-map channels.  The model's length-conditioning inputs are held fixed at the
dataset mean so the explainer sees a single [F, S, S] input.  SHAP values are
aggregated spatially (mean |shap| over the S×S grid per channel) to produce a
per-feature importance score.

Usage
-----
    python -m backend.shap_analysis --mode context-vs-generated
    python -m backend.shap_analysis --mode all
    python -m backend.shap_analysis --mode context-vs-generated --n-bg 100 --n-explain 200

Outputs (shap_reports/{mode}/)
------------------------------
    shap_summary.json   mean |SHAP| per feature, ranked descending
    shap_bar.png        horizontal bar chart
    beeswarm.png        feature-value vs SHAP scatter coloured by sign
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from backend.feature_cache import FeatureCache
from backend.feature_registry import get_feature_keys
from backend.predict import _load_model_from_checkpoint
from backend.train import feature_map_to_tensor, load_json_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ["context-vs-generated", "reference-vs-generated", "model-vs-model"]

MODE_CKPT: dict[str, str] = {
    m: f"models/{m}/best.pth" for m in MODES
}
MODE_DATA: dict[str, str] = {
    m: f"data/{m}/test.json" for m in MODES
}

# Rough sentence splitter — avoids booting the full coref pipeline just to get counts.
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _approx_sent_count(text: str) -> int:
    return max(1, len([s for s in _SENT_RE.split(text.strip()) if s.strip()]))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class _FixedLengthWrapper(nn.Module):
    """Wraps TextSimilarityCNN, holding lengths fixed at the dataset mean.

    GradientExplainer needs a single differentiable tensor input; this wrapper
    presents only the [B, F, S, S] feature map tensor as the input surface.
    """

    def __init__(self, model: nn.Module, mean_lengths: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean_lengths", mean_lengths.unsqueeze(0))  # [1, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = self.mean_lengths.expand(x.size(0), -1)
        return self.model(x, lengths)  # keep [B, 1] — GradientExplainer needs 2-D output


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_tensors(
    data_path: str,
    feature_keys: list[str],
    cache: FeatureCache,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Return (X, lengths, labels) for all cache-hit samples in *data_path*.

    X:        [N, F, S, S]   feature map tensors
    lengths:  [N, 2]         [log(n+1), log(m+1)] approximate sentence counts
    labels:   list[float]    ground-truth labels
    """
    pairs, labels_raw = load_json_data(data_path)
    skipped = 0
    X_list, L_list, y_list = [], [], []

    for (t1, t2), label in zip(pairs, labels_raw):
        label = float(label)
        cached = cache.get_features(t1, t2)
        if cached is None:
            skipped += 1
            continue
        try:
            tensor = feature_map_to_tensor(cached, feature_keys)  # [F, S, S]
        except KeyError:
            skipped += 1
            continue

        n = _approx_sent_count(t1)
        m = _approx_sent_count(t2)
        L_list.append([math.log(n + 1), math.log(m + 1)])
        X_list.append(tensor)
        y_list.append(label)

    if skipped:
        print(f"  [warn] {skipped}/{len(data)} samples skipped (cache miss or key mismatch)")

    X = torch.stack(X_list).to(device)                             # [N, F, S, S]
    L = torch.tensor(L_list, dtype=torch.float32).to(device)      # [N, 2]
    return X, L, y_list  # y_list already filtered to cache-hits only


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def _aggregate_shap(raw: np.ndarray) -> np.ndarray:
    """Spatially aggregate raw SHAP values from [N, F, S, S] → [N, F].

    Each channel's importance = mean of |shap| values over the S×S spatial grid.
    Using absolute values because a negative SHAP value on a high-signal cell is
    just as informative as a positive one.
    """
    return np.abs(raw).mean(axis=(2, 3))   # [N, F]


def run_shap(
    mode: str,
    n_bg: int = 50,
    n_explain: int | None = None,
    device_str: str = "cpu",
) -> dict:
    """Run SHAP analysis for one mode.

    Args:
        mode:      evaluation mode string
        n_bg:      number of background samples for GradientExplainer
        n_explain: number of test samples to explain (None = all)
        device_str: torch device string

    Returns:
        dict with keys 'feature_keys', 'mean_abs_shap', 'mean_shap' (signed),
        'feature_map_means' — all indexed by feature key.
    """
    import shap  # lazy import so the rest of the codebase stays light

    device = torch.device(device_str)
    feature_keys = get_feature_keys(mode)
    F = len(feature_keys)

    ckpt_path = MODE_CKPT.get(mode)
    data_path = MODE_DATA.get(mode)
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")

    print(f"\n{'='*60}")
    print(f"Mode: {mode}  |  {F} features")
    print(f"{'='*60}")

    # --- Load model ---
    print("Loading checkpoint …")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, arch = _load_model_from_checkpoint(checkpoint, device)
    model.eval()
    use_len = getattr(model, "use_length_cond", False)
    print(f"  {arch}")

    # --- Load data ---
    print("Loading test-split feature maps from cache …")
    cache = FeatureCache()
    X, L, labels = _load_tensors(data_path, feature_keys, cache, device)
    N = X.size(0)
    print(f"  {N} samples loaded  (F={F}, S={X.shape[-1]})")

    if N < n_bg + 1:
        raise ValueError(f"Too few cached samples ({N}) for n_bg={n_bg}")

    # --- Build wrapper ---
    mean_L = L.mean(dim=0)        # [2]
    if use_len:
        wrapper = _FixedLengthWrapper(model, mean_L).to(device)
        wrapper.eval()
    else:
        # Model ignores lengths; wrapper still forwards X only
        class _NoLenWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.model = m
            def forward(self, x): return self.model(x)  # keep [B, 1]
        wrapper = _NoLenWrapper(model).to(device)
        wrapper.eval()

    # --- SHAP ---
    rng = np.random.default_rng(42)
    idx_bg  = rng.choice(N, size=min(n_bg, N // 2), replace=False)
    idx_exp = (
        rng.choice(N, size=min(n_explain, N), replace=False)
        if n_explain is not None
        else np.arange(N)
    )

    bg   = X[idx_bg]    # [n_bg, F, S, S]
    expl = X[idx_exp]   # [M, F, S, S]

    print(f"Running GradientExplainer (bg={len(idx_bg)}, explain={len(idx_exp)}) …")
    explainer   = shap.GradientExplainer(wrapper, bg)
    shap_raw = np.array(explainer.shap_values(expl))
    # SHAP 0.46+ appends an output-node axis: (M, F, S, S, n_out) → drop it
    if shap_raw.ndim == 5:
        shap_raw = shap_raw[..., 0]   # (M, F, S, S)

    # Aggregate spatially
    shap_feat   = _aggregate_shap(shap_raw)         # [M, F] — mean |shap|
    shap_signed = shap_raw.mean(axis=(2, 3))        # [M, F] — mean signed shap
    feat_means  = expl.cpu().numpy().mean(axis=(2, 3))  # [M, F] — mean feature value

    mean_abs    = shap_feat.mean(axis=0)     # [F]
    mean_signed = shap_signed.mean(axis=0)  # [F]
    feat_global = feat_means.mean(axis=0)   # [F]

    ranked_idx = np.argsort(mean_abs)[::-1]

    result = {
        "mode":           mode,
        "n_samples":      int(len(idx_exp)),
        "feature_keys":   feature_keys,
        "mean_abs_shap":  {feature_keys[i]: float(mean_abs[i])    for i in range(F)},
        "mean_shap":      {feature_keys[i]: float(mean_signed[i]) for i in range(F)},
        "mean_feat_val":  {feature_keys[i]: float(feat_global[i]) for i in range(F)},
        "ranked":         [feature_keys[i] for i in ranked_idx],
    }

    # --- Plots ---
    out_dir = Path(f"shap_reports/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_bar(mean_abs, feature_keys, ranked_idx, mode, out_dir)
    _plot_beeswarm(shap_signed, feat_means, feature_keys, mode, out_dir)

    # --- JSON report ---
    report_path = out_dir / "shap_summary.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {report_path}")

    # --- Console table ---
    print(f"\n  {'Rank':<5} {'Feature':<45} {'mean|SHAP|':>10}  {'direction':>10}")
    print(f"  {'-'*75}")
    for rank, i in enumerate(ranked_idx, 1):
        direction = "↑ pos" if mean_signed[i] > 0 else "↓ neg"
        print(f"  {rank:<5} {feature_keys[i]:<45} {mean_abs[i]:>10.5f}  {direction:>10}")

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _short_name(key: str) -> str:
    replacements = {
        "mixedbread-ai/mxbai-embed-large-v1":     "mxbai_cos",
        "PREC_mixedbread-ai/mxbai-embed-large-v1": "mxbai_prec",
        "REC_mixedbread-ai/mxbai-embed-large-v1":  "mxbai_rec",
        "PREC_Qwen/Qwen3-Embedding-0.6B":          "qwen_prec",
    }
    return replacements.get(key, key)


def _plot_bar(
    mean_abs: np.ndarray,
    feature_keys: list[str],
    ranked_idx: np.ndarray,
    mode: str,
    out_dir: Path,
) -> None:
    labels  = [_short_name(feature_keys[i]) for i in ranked_idx]
    values  = mean_abs[ranked_idx]
    colors  = plt.cm.viridis(np.linspace(0.85, 0.25, len(labels)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.35)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (spatially aggregated)", fontsize=10)
    ax.set_title(f"Feature importance — {mode}", fontsize=12, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def _plot_beeswarm(
    shap_signed: np.ndarray,
    feat_means: np.ndarray,
    feature_keys: list[str],
    mode: str,
    out_dir: Path,
) -> None:
    """One row per feature; dots are individual samples; colour = feature value."""
    F = len(feature_keys)
    mean_abs = np.abs(shap_signed).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(12, max(5, F * 0.38)))

    for row_idx, fi in enumerate(order):
        x = shap_signed[:, fi]
        v = feat_means[:, fi]
        vmin, vmax = v.min(), v.max()
        norm_v = (v - vmin) / (vmax - vmin + 1e-9)

        # jitter y slightly
        rng = np.random.default_rng(fi)
        y   = row_idx + rng.uniform(-0.3, 0.3, size=len(x))
        sc  = ax.scatter(x, y, c=norm_v, cmap="RdBu_r", alpha=0.55,
                         s=12, linewidths=0, vmin=0, vmax=1)

    ax.set_yticks(range(F))
    ax.set_yticklabels([_short_name(feature_keys[i]) for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (signed, spatially averaged)", fontsize=10)
    ax.set_title(f"SHAP beeswarm — {mode}\n(colour = feature value: blue=low, red=high)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc, ax=ax, label="Normalised feature value", fraction=0.015, pad=0.02)
    plt.tight_layout()
    path = out_dir / "beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SHAP feature importance for SilverBullet")
    parser.add_argument(
        "--mode", default="all",
        choices=MODES + ["all"],
        help="Evaluation mode to analyse (default: all)",
    )
    parser.add_argument(
        "--n-bg", type=int, default=50,
        help="Background samples for GradientExplainer (default: 50)",
    )
    parser.add_argument(
        "--n-explain", type=int, default=None,
        help="Test samples to explain (default: all cached)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu; use cuda for GPU)",
    )
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else [args.mode]
    all_results = {}

    for mode in modes:
        try:
            result = run_shap(
                mode,
                n_bg=args.n_bg,
                n_explain=args.n_explain,
                device_str=args.device,
            )
            all_results[mode] = result
        except FileNotFoundError as e:
            print(f"[skip] {mode}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[error] {mode}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    if len(modes) > 1 and all_results:
        _print_cross_mode_summary(all_results)


def _print_cross_mode_summary(results: dict) -> None:
    all_keys = sorted({k for r in results.values() for k in r["mean_abs_shap"]})
    mode_list = list(results.keys())

    header = f"\n{'Feature':<45}" + "".join(f"  {m[:8]:>10}" for m in mode_list)
    print("\n" + "="*80)
    print("Cross-mode SHAP comparison (mean |SHAP|)")
    print("="*80)
    print(header)
    print("-"*80)
    for k in all_keys:
        row = f"{_short_name(k):<45}"
        for m in mode_list:
            val = results[m]["mean_abs_shap"].get(k)
            row += f"  {val:>10.5f}" if val is not None else f"  {'—':>10}"
        print(row)


if __name__ == "__main__":
    main()
