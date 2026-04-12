"""
Self-speculative decoding: layer skipping utilities.

This module provides tools to test the effects of skipping decoder layers on action output.
Useful for measuring which decoder layers are critical for task performance and identifying
opportunities for inference-time speedups via speculative decoding.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

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
                    # inputs is a tuple, first element is hidden states
                    # We want to return it unchanged to skip computation
                    return inputs[0]  # Return hidden states directly (residual)

                return skip_forward

            # Register forward hook (runs after forward pass)
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
) -> Dict[str, float]:
    """
    Compute metrics comparing original and layer-skipped actions.

    Args:
        original_actions: List of action arrays from original model
        skipped_actions: List of action arrays from layer-skipped model

    Returns:
        Dictionary with metrics: l2_mean, l2_max, l2_std, value_diff_mean, value_diff_max, anomalies, etc.
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

    return metrics


def test_layer_skipping(
    vla: torch.nn.Module,
    cfg: GenerateConfig,
    processor: Any,
    observation: Dict,
    proprio_projector: Any,
    max_skip_layers: Optional[int] = None,
) -> None:
    """
    Test the effects of skipping consecutive decoder layers on action output.

    Tests skipping the last N layers for N in [1, 2, 3, ..., max_skip_layers].
    Reports metrics comparing each skipping pattern to the original model.

    Args:
        vla: The OpenVLA model
        cfg: Configuration object
        processor: Image processor
        observation: Sample observation dict
        proprio_projector: Proprioception projector
        max_skip_layers: Maximum number of layers to skip (default: all layers)
    """
    # Get number of decoder layers
    num_decoder_layers = len(vla.language_model.model.layers)
    if max_skip_layers is None:
        max_skip_layers = num_decoder_layers

    print("\n" + "=" * 80)
    print("SELF-SPECULATIVE DECODING: LAYER SKIPPING TEST")
    print("=" * 80)
    print(f"Model: {cfg.pretrained_checkpoint}")
    print(f"Total decoder layers: {num_decoder_layers}")
    print(f"Testing skip patterns: last 1 to last {max_skip_layers} layers")
    print("=" * 80 + "\n")

    # Generate reference actions with full model
    print("Generating reference actions (full model)...")
    with torch.inference_mode():
        original_actions = get_vla_action(
            cfg, vla, processor, observation, observation["task_description"], proprio_projector=proprio_projector
        )
    print(f"✓ Reference actions generated: {len(original_actions)} action steps\n")

    # Test each consecutive layer skip pattern
    results = []
    for num_skip in range(1, max_skip_layers + 1):
        skip_layers = set(range(num_decoder_layers - num_skip, num_decoder_layers))

        print(f"Testing: Skip last {num_skip} layer(s) (indices {min(skip_layers)}-{max(skip_layers)})...", end=" ")

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
        metrics = compute_action_metrics(original_actions, skipped_actions)

        result = {"num_skip_layers": num_skip, "skip_layer_range": f"{min(skip_layers)}-{max(skip_layers)}", **metrics}
        results.append(result)

        status = "✓" if not metrics["anomalies"] else "⚠ ANOMALY"
        print(f"{status}")
        print(f"  L2 distance: mean={metrics['l2_mean']:.6f}, max={metrics['l2_max']:.6f}, std={metrics['l2_std']:.6f}")
        print(f"  Value diff:  mean={metrics['value_diff_mean']:.6f}, max={metrics['value_diff_max']:.6f}")
        if metrics["anomalies"]:
            print(
                f"  ⚠ Anomalies: NaN={metrics['has_nan']}, Inf={metrics['has_inf']}, Extremes={metrics['has_extremes']}"
            )
        print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (sorted by L2 distance)")
    print("=" * 80)
    print(f"{'Skip Layers':<15} {'L2 Mean':<15} {'L2 Max':<15} {'L2 Std':<15} {'Mean Diff':<15} {'Anomalies':<10}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x["l2_mean"])
    for result in sorted_results:
        skip_str = f"last {result['num_skip_layers']}"
        anom_str = "YES" if result["anomalies"] else "no"
        print(
            f"{skip_str:<15} "
            f"{result['l2_mean']:<15.6f} "
            f"{result['l2_max']:<15.6f} "
            f"{result['l2_std']:<15.6f} "
            f"{result['value_diff_mean']:<15.6f} "
            f"{anom_str:<10}"
        )

    print("=" * 80 + "\n")

    # Best pattern analysis
    best_result = sorted_results[0]
    print(f"Best pattern: Skip last {best_result['num_skip_layers']} layers")
    print(f"  L2 distance: {best_result['l2_mean']:.6f} (mean), {best_result['l2_max']:.6f} (max)")
    print(
        f"  Speed improvement potential: ~{100 * best_result['num_skip_layers'] / num_decoder_layers:.1f}% of decoder compute"
    )
