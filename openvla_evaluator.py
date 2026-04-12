"""OpenVLA evaluator used by optimization and greedy-search demos."""

from __future__ import annotations

from typing import Dict, Set

import numpy as np
import torch

from experiments.robot.openvla_utils import get_vla_action
from optimization import LayerSkipEvaluator
from skip import compute_action_metrics, patch_layer_skipping


class OpenVLAEvaluator(LayerSkipEvaluator):
    """Evaluate a skip set by running OpenVLA and computing action metrics."""

    def __init__(self, cfg, vla, processor, observation, proprio_projector):
        self.cfg = cfg
        self.vla = vla
        self.processor = processor
        self.observation = observation
        self.proprio_projector = proprio_projector

        with torch.inference_mode():
            self.reference_actions = get_vla_action(
                cfg,
                vla,
                processor,
                observation,
                observation["task_description"],
                proprio_projector=proprio_projector,
            )

        action_stats = vla.get_action_stats(cfg.unnorm_key)
        if "q01" in action_stats and "q99" in action_stats:
            self.action_low = np.array(action_stats["q01"])
            self.action_high = np.array(action_stats["q99"])
        elif "min" in action_stats and "max" in action_stats:
            self.action_low = np.array(action_stats["min"])
            self.action_high = np.array(action_stats["max"])
        else:
            self.action_low = None
            self.action_high = None

        self.n_action_bins = int(getattr(vla.config, "n_action_bins", 0))
        self.metric_name = (
            "bin_idx_dist_mean"
            if self.action_low is not None and self.action_high is not None and self.n_action_bins > 1
            else "l2_mean"
        )

    def evaluate(self, skip_layers: Set[int]) -> Dict[str, float]:
        with torch.inference_mode():
            with patch_layer_skipping(self.vla, skip_layers):
                skipped_actions = get_vla_action(
                    self.cfg,
                    self.vla,
                    self.processor,
                    self.observation,
                    self.observation["task_description"],
                    proprio_projector=self.proprio_projector,
                )

        return compute_action_metrics(
            self.reference_actions,
            skipped_actions,
            action_low=self.action_low,
            action_high=self.action_high,
            n_action_bins=self.n_action_bins,
        )
