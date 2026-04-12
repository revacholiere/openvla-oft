"""
Self-speculative decoding: layer skipping utilities.

This module provides tools to test the effects of skipping decoder layers on action output.
Useful for measuring which decoder layers are critical for task performance and identifying
opportunities for inference-time speedups via speculative decoding.
"""

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_vla_action


@contextmanager
def patch_layer_skipping(vla: torch.nn.Module, skip_layer_indices: set):
    """
    Temporarily patch the LLM decoder layers to skip computation for specified layers.

    When a layer is skipped, its forward pass returns the input unchanged (residual connection).
    This is done via forward hooks, which are temporary and fully reversible.

    Args:
        vla: The OpenVLA model
        skip_layer_indices: Set of layer indices to skip (e.g., {31} to skip last layer)

    Yields:
        The same vla model with hooks installed
    """
    handles = []

    try:
        # Get the decoder layers from the LLaMA model
        decoder_layers = vla.language_model.model.layers

        # Install hooks on each layer to be skipped
        for layer_idx in skip_layer_indices:
            layer = decoder_layers[layer_idx]

            # Create a hook that returns the input unchanged (residual skip)
            def make_skip_hook():
                def skip_forward(module, inputs, output):
                    # LlamaDecoderLayer output is a tuple: (hidden_states, next_cache, [attentions])
                    # We return input hidden_states but keep the cache structure from output
                    # This implements residual: skip the layer computation, pass hidden states through
                    return (inputs[0], *output[1:])

                return skip_forward

            # Register forward hook (runs after forward pass but replaces the output)
            handle = layer.register_forward_hook(make_skip_hook())
            handles.append(handle)

        yield vla

    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()


def compute_action_metrics(
    original_actions: List[np.ndarray],
    skipped_actions: List[np.ndarray],
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    n_action_bins: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute metrics comparing original and layer-skipped actions.

    Args:
        original_actions: List of action arrays from original model
        skipped_actions: List of action arrays from layer-skipped model

    Returns:
        Dictionary with metrics: l2 stats, value-diff stats, anomaly flags, and optional bin-index distance stats.
    """
    # Stack actions for easier computation
    orig_arr = np.array(original_actions)  # Shape: (num_actions, action_dim)
    skip_arr = np.array(skipped_actions)

    # Compute L2 distance between each action pair
    action_diffs = orig_arr - skip_arr
    l2_distances = np.linalg.norm(action_diffs, axis=1)  # L2 norm per action

    # Aggregate metrics
    metrics = {
        "l2_mean": float(np.mean(l2_distances)),
        "l2_max": float(np.max(l2_distances)),
        "l2_std": float(np.std(l2_distances)),
        "value_diff_mean": float(np.mean(np.abs(action_diffs))),
        "value_diff_max": float(np.max(np.abs(action_diffs))),
    }

    # Check for anomalies (NaNs, Infs, extreme values)
    has_nan = bool(np.isnan(skip_arr).any())
    has_inf = bool(np.isinf(skip_arr).any())
    extremes = bool(np.abs(skip_arr).max() > 100)  # Sanity check for reasonable action values

    metrics["anomalies"] = has_nan or has_inf or extremes
    metrics["has_nan"] = has_nan
    metrics["has_inf"] = has_inf
    metrics["has_extremes"] = extremes

    # Optional token-bin index distance metrics.
    # We map unnormalized actions -> normalized [-1, 1] -> nearest bin index in [0, n_action_bins-1].
    if action_low is not None and action_high is not None and n_action_bins is not None and n_action_bins > 1:
        denom = np.maximum(action_high - action_low, 1e-8)

        orig_norm = 2.0 * (orig_arr - action_low) / denom - 1.0
        skip_norm = 2.0 * (skip_arr - action_low) / denom - 1.0

        orig_norm = np.clip(orig_norm, -1.0, 1.0)
        skip_norm = np.clip(skip_norm, -1.0, 1.0)

        scale = 0.5 * (n_action_bins - 1)
        orig_bin_idx = np.rint((orig_norm + 1.0) * scale).astype(np.int32)
        skip_bin_idx = np.rint((skip_norm + 1.0) * scale).astype(np.int32)

        orig_bin_idx = np.clip(orig_bin_idx, 0, n_action_bins - 1)
        skip_bin_idx = np.clip(skip_bin_idx, 0, n_action_bins - 1)

        bin_idx_dist = np.abs(orig_bin_idx - skip_bin_idx)
        metrics["bin_idx_dist_mean"] = float(np.mean(bin_idx_dist))
        metrics["bin_idx_dist_max"] = float(np.max(bin_idx_dist))
        metrics["bin_idx_dist_std"] = float(np.std(bin_idx_dist))

    return metrics


def _build_contiguous_segments(layer_indices: List[int]) -> List[List[int]]:
    """Split sorted indices into contiguous segments in index space."""
    if not layer_indices:
        return []

    segments: List[List[int]] = []
    current_segment: List[int] = [layer_indices[0]]
    for idx in layer_indices[1:]:
        if idx == current_segment[-1] + 1:
            current_segment.append(idx)
        else:
            segments.append(current_segment)
            current_segment = [idx]
    segments.append(current_segment)
    return segments


def build_contiguous_skip_ranges(searchable_layers: List[int], max_skip_layers: int) -> List[Tuple[int, int]]:
    """
    Build all contiguous layer ranges [start, end] inside searchable layers.

    Contiguity is defined in model index space (e.g. 5-7 is contiguous; 5-7 and 9-10 are separate segments).
    """
    ranges: List[Tuple[int, int]] = []
    for segment in _build_contiguous_segments(searchable_layers):
        seg_len = len(segment)
        for start_pos in range(seg_len):
            max_end_pos = min(seg_len - 1, start_pos + max_skip_layers - 1)
            for end_pos in range(start_pos, max_end_pos + 1):
                ranges.append((segment[start_pos], segment[end_pos]))
    return ranges


def test_layer_skipping(
    vla: torch.nn.Module,
    cfg: GenerateConfig,
    processor: Any,
    observation: Dict,
    proprio_projector: Any,
    max_skip_layers: Optional[int] = None,
    always_exclude_layers: Optional[Set[int]] = None,
    always_include_layers: Optional[Set[int]] = None,
    csv_output_path: Optional[str] = "layer_skip_results.csv",
) -> None:
    """
    Test the effects of skipping consecutive decoder layers on action output.

    Tests all contiguous layer ranges [start, end], i.e., skip layers start..end.
    Example ranges include [0, 0], [0, 1], [3, 7], ..., [last, last].
    Reports metrics comparing each skipping pattern to the original model.

    Args:
        vla: The OpenVLA model
        cfg: Configuration object
        processor: Image processor
        observation: Sample observation dict
        proprio_projector: Proprioception projector
        max_skip_layers: Maximum number of contiguous layers to skip (default: all layers)
        always_exclude_layers: Layers that are never skipped by the search
        always_include_layers: Layers that are always skipped for every candidate
        csv_output_path: CSV output path. Set to None to disable CSV writing.
    """
    # Get number of decoder layers
    num_decoder_layers = len(vla.language_model.model.layers)
    if max_skip_layers is None:
        max_skip_layers = num_decoder_layers

    always_exclude_layers = set(always_exclude_layers or set())
    always_include_layers = set(always_include_layers or set())

    # Validate layer indices and sanitize overlap: include takes precedence over exclude
    valid_layer_set = set(range(num_decoder_layers))
    invalid_excludes = sorted(always_exclude_layers - valid_layer_set)
    invalid_includes = sorted(always_include_layers - valid_layer_set)
    if invalid_excludes or invalid_includes:
        raise ValueError(
            f"Invalid layer indices. excludes={invalid_excludes}, includes={invalid_includes}, "
            f"valid=0..{num_decoder_layers - 1}"
        )

    overlap = always_exclude_layers & always_include_layers
    if overlap:
        always_exclude_layers -= overlap

    searchable_layers = sorted(valid_layer_set - always_exclude_layers - always_include_layers)

    print("\n" + "=" * 80)
    print("SELF-SPECULATIVE DECODING: LAYER SKIPPING TEST")
    print("=" * 80)
    print(f"Model: {cfg.pretrained_checkpoint}")
    print(f"Total decoder layers: {num_decoder_layers}")
    print(f"Testing contiguous skip lengths: 1 to {max_skip_layers} layers")
    print(f"Always-excluded layers: {sorted(always_exclude_layers) if always_exclude_layers else 'None'}")
    print(f"Always-included layers: {sorted(always_include_layers) if always_include_layers else 'None'}")
    print(f"Searchable layers: {searchable_layers if searchable_layers else 'None'}")

    skip_ranges = build_contiguous_skip_ranges(searchable_layers, max_skip_layers)
    print(f"Total skip ranges to evaluate: {len(skip_ranges)}")
    print("=" * 80 + "\n")

    # Generate reference actions with full model
    print("Generating reference actions (full model)...")
    with torch.inference_mode():
        original_actions = get_vla_action(
            cfg, vla, processor, observation, observation["task_description"], proprio_projector=proprio_projector
        )
    print(f"✓ Reference actions generated: {len(original_actions)} action steps\n")

    # Prepare action binning metadata for bin-index distance metrics.
    action_low = None
    action_high = None
    n_action_bins = None
    try:
        action_stats = vla.get_action_stats(cfg.unnorm_key)
        if "q01" in action_stats and "q99" in action_stats:
            action_low = np.array(action_stats["q01"])
            action_high = np.array(action_stats["q99"])
        elif "min" in action_stats and "max" in action_stats:
            action_low = np.array(action_stats["min"])
            action_high = np.array(action_stats["max"])

        n_action_bins = int(getattr(vla.config, "n_action_bins", 0))
        if action_low is None or action_high is None or n_action_bins <= 1:
            action_low, action_high, n_action_bins = None, None, None
            print("Bin-index distance disabled: missing action stats or n_action_bins <= 1")
        else:
            print(f"Bin-index distance enabled: n_action_bins={n_action_bins}")
    except Exception as exc:
        print(f"Bin-index distance disabled: failed to read action bin metadata ({exc})")

    # Test each contiguous layer range; always-included layers are added to every candidate.
    results = []
    if not skip_ranges:
        skip_ranges = [(-1, -1)]

    for range_idx, (start_idx, end_idx) in enumerate(skip_ranges, start=1):
        range_layers = set() if start_idx == -1 else set(range(start_idx, end_idx + 1))
        skip_layers = always_include_layers | range_layers
        num_skip = len(skip_layers)

        if start_idx == -1:
            range_desc = "none"
            search_skip_count = 0
        else:
            range_desc = f"{start_idx}-{end_idx}"
            search_skip_count = end_idx - start_idx + 1

        print(
            f"Testing [{range_idx}/{len(skip_ranges)}]: "
            f"Search range {range_desc} ({search_skip_count} layer(s)); total skipped {num_skip}...",
            end=" ",
        )

        with torch.inference_mode():
            with patch_layer_skipping(vla, skip_layers):
                skipped_actions = get_vla_action(
                    cfg,
                    vla,
                    processor,
                    observation,
                    observation["task_description"],
                    proprio_projector=proprio_projector,
                )

        # Compute metrics
        metrics = compute_action_metrics(
            original_actions,
            skipped_actions,
            action_low=action_low,
            action_high=action_high,
            n_action_bins=n_action_bins,
        )

        result = {
            "num_skip_layers": num_skip,
            "start_layer": start_idx,
            "end_layer": end_idx,
            "skip_layer_range": range_desc,
            "search_skip_layers": search_skip_count,
            **metrics,
        }
        results.append(result)

        status = "✓" if not metrics["anomalies"] else "⚠ ANOMALY"
        print(f"{status}")
        print(f"  L2 distance: mean={metrics['l2_mean']:.6f}, max={metrics['l2_max']:.6f}, std={metrics['l2_std']:.6f}")
        print(f"  Value diff:  mean={metrics['value_diff_mean']:.6f}, max={metrics['value_diff_max']:.6f}")
        if "bin_idx_dist_mean" in metrics:
            print(
                f"  Bin distance: mean={metrics['bin_idx_dist_mean']:.6f}, "
                f"max={metrics['bin_idx_dist_max']:.6f}, std={metrics['bin_idx_dist_std']:.6f}"
            )
        if metrics["anomalies"]:
            print(
                f"  ⚠ Anomalies: NaN={metrics['has_nan']}, Inf={metrics['has_inf']}, Extremes={metrics['has_extremes']}"
            )
        print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (sorted by L2 distance)")
    print("=" * 80)
    print(
        f"{'Range':<15} {'Search#':<8} {'Total#':<8} {'L2 Mean':<12} {'L2 Max':<12} "
        f"{'Bin Mean':<10} {'Bin Max':<10} {'Anomalies':<10}"
    )
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x["l2_mean"])
    for result in sorted_results:
        range_str = result["skip_layer_range"]
        anom_str = "YES" if result["anomalies"] else "no"
        bin_mean_str = f"{result['bin_idx_dist_mean']:.6f}" if "bin_idx_dist_mean" in result else "n/a"
        bin_max_str = f"{result['bin_idx_dist_max']:.6f}" if "bin_idx_dist_max" in result else "n/a"
        print(
            f"{range_str:<15} "
            f"{result['search_skip_layers']:<8} "
            f"{result['num_skip_layers']:<8} "
            f"{result['l2_mean']:<12.6f} "
            f"{result['l2_max']:<12.6f} "
            f"{bin_mean_str:<10} "
            f"{bin_max_str:<10} "
            f"{anom_str:<10}"
        )

    print("=" * 80 + "\n")

    if csv_output_path:
        csv_path = Path(csv_output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "run_idx",
                    "start_idx",
                    "end_idx",
                    "skip_range",
                    "search_skip_layers",
                    "total_skip_layers",
                    "l2_mean",
                    "l2_max",
                    "l2_std",
                    "value_diff_mean",
                    "value_diff_max",
                    "bin_idx_dist_mean",
                    "bin_idx_dist_max",
                    "bin_idx_dist_std",
                    "anomalies",
                    "has_nan",
                    "has_inf",
                    "has_extremes",
                    "always_exclude_layers",
                    "always_include_layers",
                ]
            )

    # Best pattern analysis
    best_result = sorted_results[0]
    print(f"Best search range: {best_result['skip_layer_range']}")
    print(f"  Search-skipped layers: {best_result['search_skip_layers']}")
    print(f"  Total skipped layers (including always-included): {best_result['num_skip_layers']}")
    print(f"  L2 distance: {best_result['l2_mean']:.6f} (mean), {best_result['l2_max']:.6f} (max)")
    print(
        f"  Speed improvement potential: ~{100 * best_result['num_skip_layers'] / num_decoder_layers:.1f}% of decoder compute"
    )

    if csv_output_path:
        with Path(csv_output_path).open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for idx, result in enumerate(sorted_results, start=1):
                writer.writerow(
                    [
                        idx,
                        result["start_layer"],
                        result["end_layer"],
                        result["skip_layer_range"],
                        result["search_skip_layers"],
                        result["num_skip_layers"],
                        result["l2_mean"],
                        result["l2_max"],
                        result["l2_std"],
                        result["value_diff_mean"],
                        result["value_diff_max"],
                        result.get("bin_idx_dist_mean", ""),
                        result.get("bin_idx_dist_max", ""),
                        result.get("bin_idx_dist_std", ""),
                        result["anomalies"],
                        result["has_nan"],
                        result["has_inf"],
                        result["has_extremes"],
                        "|".join(map(str, sorted(always_exclude_layers))),
                        "|".join(map(str, sorted(always_include_layers))),
                    ]
                )
        print(f"Saved CSV results: {csv_output_path}")
