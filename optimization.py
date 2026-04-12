"""Layer-skip optimization framework.

This module provides the abstractions and a generic optimization loop required to
search over layer skip sets. It also includes a lightweight ParEGO implementation
that can be used immediately without extra dependencies.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ObjectiveConfig:
    """Configuration for scalarizing the optimization result.

    The optimization loop will compute a scalar objective from:
    1) active layer count (compute proxy)
    2) one chosen metric from the evaluator (e.g. l2_mean or bin_idx_dist_mean)

    The scalar is then converted into a reward via sign flip when `minimize=True`.
    """

    metric_name: str = "l2_mean"
    active_layers_weight: float = 1.0
    metric_weight: float = 1.0
    minimize: bool = True


@dataclass
class SearchSpaceConfig:
    """Layer-skip search constraints."""

    num_layers: int
    always_exclude_layers: Set[int] = field(default_factory=set)
    always_include_layers: Set[int] = field(default_factory=set)
    min_skip_layers: int = 0
    max_skip_layers: Optional[int] = None
    contiguous_only: bool = False


@dataclass
class TrialRecord:
    """Single optimization trial result."""

    iteration: int
    skip_layers: Set[int]
    active_layers: int
    objectives: Dict[str, float]
    metrics: Dict[str, float]
    objective_value: float
    reward: float


@dataclass
class OptimizationResult:
    """Aggregate optimization run output."""

    best_trial: TrialRecord
    history: List[TrialRecord]


class LayerSkipEvaluator(ABC):
    """Abstract evaluator for a proposed layer skip set."""

    @abstractmethod
    def evaluate(self, skip_layers: Set[int]) -> Dict[str, float]:
        raise NotImplementedError


class BaseLayerSkipOptimizer(ABC):
    """Abstract optimizer interface for proposing layer skip sets."""

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        objective: Optional[ObjectiveConfig] = None,
        seed: int = 0,
    ) -> None:
        self.search_space = self._normalize_and_validate_search_space(search_space)
        self.objective = objective
        self.rng = random.Random(seed)

    @staticmethod
    def _normalize_and_validate_search_space(search_space: SearchSpaceConfig) -> SearchSpaceConfig:
        if search_space.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        valid_layers = set(range(search_space.num_layers))
        invalid_excluded = search_space.always_exclude_layers - valid_layers
        invalid_included = search_space.always_include_layers - valid_layers
        if invalid_excluded or invalid_included:
            raise ValueError(
                f"Invalid layer ids. excluded={sorted(invalid_excluded)}, included={sorted(invalid_included)}"
            )

        overlap = search_space.always_exclude_layers & search_space.always_include_layers
        if overlap:
            search_space.always_exclude_layers = set(search_space.always_exclude_layers) - overlap

        if search_space.max_skip_layers is None:
            search_space.max_skip_layers = search_space.num_layers

        if search_space.min_skip_layers < 0:
            raise ValueError("min_skip_layers must be >= 0")

        if search_space.max_skip_layers < search_space.min_skip_layers:
            raise ValueError("max_skip_layers must be >= min_skip_layers")

        return search_space

    @property
    def searchable_layers(self) -> List[int]:
        all_layers = set(range(self.search_space.num_layers))
        layers = all_layers - self.search_space.always_exclude_layers - self.search_space.always_include_layers
        return sorted(layers)

    def finalize_skip_layers(self, proposed_layers: Set[int]) -> Set[int]:
        skip_layers = set(proposed_layers)
        skip_layers -= self.search_space.always_exclude_layers
        skip_layers |= self.search_space.always_include_layers

        max_skip = self.search_space.max_skip_layers
        min_skip = self.search_space.min_skip_layers

        if len(skip_layers) > max_skip:
            fixed = sorted(self.search_space.always_include_layers)
            optional = sorted(skip_layers - self.search_space.always_include_layers)
            keep_optional = max(0, max_skip - len(fixed))
            skip_layers = set(fixed + optional[:keep_optional])

        if len(skip_layers) < min_skip:
            raise ValueError(
                "Proposed skip set violates min_skip_layers after constraints; subclass ask() should ensure feasibility."
            )

        if self.search_space.contiguous_only and skip_layers:
            sorted_layers = sorted(skip_layers)
            if sorted_layers[-1] - sorted_layers[0] + 1 != len(sorted_layers):
                raise ValueError("Skip set must be contiguous but received a non-contiguous proposal")

        return skip_layers

    @abstractmethod
    def ask(self) -> Set[int]:
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        skip_layers: Set[int],
        objectives: Dict[str, float],
        metrics: Dict[str, float],
        reward: float,
    ) -> None:
        raise NotImplementedError


def _contiguous_segments(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []

    segments: List[List[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]
    segments.append(current)
    return segments


def _sample_feasible_skip_set(space: SearchSpaceConfig, rng: random.Random, searchable_layers: List[int]) -> Set[int]:
    if not searchable_layers:
        return set(space.always_include_layers)

    max_skip = min(space.max_skip_layers or len(searchable_layers), len(searchable_layers))
    min_skip = min(space.min_skip_layers, max_skip)
    skip_count = rng.randint(min_skip, max_skip) if max_skip >= min_skip else len(searchable_layers)

    if space.contiguous_only:
        segments = _contiguous_segments(searchable_layers)
        candidate_segments = [seg for seg in segments if len(seg) >= skip_count]
        if not candidate_segments:
            candidate_segments = segments
        chosen_segment = rng.choice(candidate_segments)
        start_max = max(0, len(chosen_segment) - skip_count)
        start_idx = rng.randint(0, start_max) if start_max > 0 else 0
        sampled = set(chosen_segment[start_idx : start_idx + skip_count])
    else:
        sampled = set(rng.sample(searchable_layers, skip_count))

    sampled |= set(space.always_include_layers)
    sampled -= set(space.always_exclude_layers)
    return sampled


def _skip_set_to_binary(skip_layers: Set[int], searchable_layers: List[int]) -> np.ndarray:
    return np.array([1.0 if layer in skip_layers else 0.0 for layer in searchable_layers], dtype=np.float64)


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    diff = x1[:, None, :] - x2[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    return np.exp(-0.5 * sq_dist / max(length_scale, 1e-8) ** 2)


def _fit_gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    length_scale: float = 1.5,
    noise: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x_train) == 0:
        return np.zeros(len(x_test), dtype=np.float64), np.ones(len(x_test), dtype=np.float64)

    k_xx = _rbf_kernel(x_train, x_train, length_scale)
    k_xx.flat[:: len(x_train) + 1] += noise
    k_xs = _rbf_kernel(x_train, x_test, length_scale)
    k_ss = _rbf_kernel(x_test, x_test, length_scale)

    try:
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
        mu = k_xs.T @ alpha
        v = np.linalg.solve(chol, k_xs)
        var = np.clip(np.diag(k_ss) - np.sum(v * v, axis=0), 1e-12, None)
        return mu, np.sqrt(var)
    except np.linalg.LinAlgError:
        # Fallback to a stable least-squares fit if kernel matrix is singular.
        ridge = 1e-6
        coeff = np.linalg.solve(k_xx + ridge * np.eye(len(k_xx)), y_train)
        mu = k_xs.T @ coeff
        var = np.full(len(x_test), np.var(y_train) if len(y_train) > 1 else 1.0, dtype=np.float64)
        return mu, np.sqrt(np.maximum(var, 1e-12))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    # Minimization EI.
    sigma = np.maximum(sigma, 1e-12)
    improvement = best - mu - xi
    z = improvement / sigma
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    normal_pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return improvement * normal_cdf + sigma * normal_pdf


class ParEGOOptimizer(BaseLayerSkipOptimizer):
    """Lightweight ParEGO optimizer for layer skipping.

    This implementation uses a random-weight scalarization of two objectives:
    1) active_layers
    2) selected metric from ObjectiveConfig.metric_name

    A simple numpy GP surrogate is fit on the binary skip-vector representation and
    candidates are chosen via expected improvement.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        objective: ObjectiveConfig,
        seed: int = 0,
        candidate_pool_size: int = 256,
        length_scale: float = 1.5,
        noise: float = 1e-6,
        rho: float = 0.05,
        xi: float = 0.01,
    ) -> None:
        super().__init__(search_space=search_space, objective=objective, seed=seed)
        self.candidate_pool_size = candidate_pool_size
        self.length_scale = length_scale
        self.noise = noise
        self.rho = rho
        self.xi = xi

        self._skip_history: List[Set[int]] = []
        self._objective_history: List[Dict[str, float]] = []
        self._current_weights: Optional[np.ndarray] = None
        self._objective_names = ("active_layers", self.objective.metric_name)

    def _sample_weights(self) -> np.ndarray:
        weights = self.rng.random(), self.rng.random()
        total = max(sum(weights), 1e-12)
        return np.array([weights[0] / total, weights[1] / total], dtype=np.float64)

    def _scalarize_history(self, weights: np.ndarray) -> np.ndarray:
        values = np.array(
            [[obj["active_layers"], obj[self.objective.metric_name]] for obj in self._objective_history],
            dtype=np.float64,
        )
        if len(values) == 0:
            return np.array([], dtype=np.float64)

        ideal = np.min(values, axis=0)
        scale = np.maximum(np.max(values, axis=0) - ideal, 1e-12)
        normalized = (values - ideal) / scale

        weighted = weights[None, :] * normalized
        tchebycheff = np.max(weighted, axis=1) + self.rho * np.sum(weighted, axis=1)
        return tchebycheff

    def _generate_candidate_pool(self) -> List[Set[int]]:
        candidates: List[Set[int]] = []
        seen = set()
        max_attempts = max(4 * self.candidate_pool_size, 32)
        attempts = 0
        while len(candidates) < self.candidate_pool_size and attempts < max_attempts:
            attempts += 1
            candidate = self.finalize_skip_layers(
                _sample_feasible_skip_set(self.search_space, self.rng, self.searchable_layers)
            )
            key = tuple(sorted(candidate))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def ask(self) -> Set[int]:
        if len(self._skip_history) < 2:
            return self.finalize_skip_layers(
                _sample_feasible_skip_set(self.search_space, self.rng, self.searchable_layers)
            )

        weights = self._sample_weights()
        self._current_weights = weights
        y_train = self._scalarize_history(weights)
        x_train = np.array(
            [_skip_set_to_binary(skip_layers, self.searchable_layers) for skip_layers in self._skip_history],
            dtype=np.float64,
        )

        candidates = self._generate_candidate_pool()
        if not candidates:
            return self.finalize_skip_layers(
                _sample_feasible_skip_set(self.search_space, self.rng, self.searchable_layers)
            )

        x_test = np.array([_skip_set_to_binary(candidate, self.searchable_layers) for candidate in candidates])
        mu, sigma = _fit_gp_predict(x_train, y_train, x_test, length_scale=self.length_scale, noise=self.noise)
        acquisition = _expected_improvement(mu, sigma, best=float(np.min(y_train)), xi=self.xi)

        best_idx = int(np.argmax(acquisition))
        return candidates[best_idx]

    def tell(
        self,
        skip_layers: Set[int],
        objectives: Dict[str, float],
        metrics: Dict[str, float],
        reward: float,
    ) -> None:
        self._skip_history.append(set(skip_layers))
        self._objective_history.append(dict(objectives))


class GreedyLayerSkipOptimizer:
    """Greedy search that adds one layer at a time.

    At each iteration, the optimizer evaluates all still-unskipped candidate layers
    and picks the one that yields the lowest value for `metric_name`.

    This is intentionally simple and deterministic. It is useful for building a
    timeline of how performance changes as more layers are removed.
    """

    def __init__(
        self,
        search_space: SearchSpaceConfig,
        metric_name: str = "l2_mean",
    ) -> None:
        self.search_space = BaseLayerSkipOptimizer._normalize_and_validate_search_space(search_space)
        self.metric_name = metric_name

    def _objective_and_reward(self, skip_layers: Set[int], metrics: Dict[str, float]) -> Tuple[Dict[str, float], float, float, int]:
        if self.metric_name not in metrics:
            raise KeyError(
                f"Metric '{self.metric_name}' missing from evaluator output. "
                f"Available metrics: {sorted(metrics.keys())}"
            )

        active_layers = self.search_space.num_layers - len(skip_layers)
        metric_value = float(metrics[self.metric_name])
        objectives = {"active_layers": float(active_layers), self.metric_name: metric_value}
        objective_value = metric_value
        reward = -objective_value
        return objectives, objective_value, reward, active_layers

    def run(self, evaluator: LayerSkipEvaluator) -> OptimizationResult:
        current_skip_layers = set(self.search_space.always_include_layers)
        searchable_layers = [
            layer
            for layer in range(self.search_space.num_layers)
            if layer not in self.search_space.always_exclude_layers and layer not in self.search_space.always_include_layers
        ]

        history: List[TrialRecord] = []

        baseline_metrics = evaluator.evaluate(current_skip_layers)
        objectives, objective_value, reward, active_layers = self._objective_and_reward(
            current_skip_layers,
            baseline_metrics,
        )
        history.append(
            TrialRecord(
                iteration=0,
                skip_layers=set(current_skip_layers),
                active_layers=active_layers,
                objectives=objectives,
                metrics=dict(baseline_metrics),
                objective_value=objective_value,
                reward=reward,
            )
        )

        iteration = 1
        remaining_layers = [layer for layer in searchable_layers if layer not in current_skip_layers]
        while remaining_layers:
            best_layer = None
            best_metrics: Optional[Dict[str, float]] = None
            best_objective_value = float("inf")
            best_reward = float("-inf")
            best_active_layers = 0
            best_objectives: Dict[str, float] = {}

            for candidate_layer in remaining_layers:
                proposal = set(current_skip_layers)
                proposal.add(candidate_layer)
                proposal = proposal - self.search_space.always_exclude_layers
                proposal = proposal | self.search_space.always_include_layers

                metrics = evaluator.evaluate(proposal)
                objectives, objective_value, reward, active_layers = self._objective_and_reward(proposal, metrics)

                if objective_value < best_objective_value:
                    best_layer = candidate_layer
                    best_metrics = metrics
                    best_objective_value = objective_value
                    best_reward = reward
                    best_active_layers = active_layers
                    best_objectives = objectives

            if best_layer is None or best_metrics is None:
                break

            current_skip_layers.add(best_layer)
            remaining_layers = [layer for layer in remaining_layers if layer != best_layer]

            history.append(
                TrialRecord(
                    iteration=iteration,
                    skip_layers=set(current_skip_layers),
                    active_layers=best_active_layers,
                    objectives=best_objectives,
                    metrics=dict(best_metrics),
                    objective_value=best_objective_value,
                    reward=best_reward,
                )
            )
            iteration += 1

        best_trial = min(history, key=lambda trial: trial.objective_value)
        return OptimizationResult(best_trial=best_trial, history=history)


class OptimizationLoop:
    """Generic optimization loop that is independent of concrete optimizer choice."""

    def __init__(
        self,
        evaluator: LayerSkipEvaluator,
        optimizer: BaseLayerSkipOptimizer,
        objective: ObjectiveConfig,
    ) -> None:
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.objective = objective
        self.history: List[TrialRecord] = []

    def _compute_objectives_and_reward(
        self,
        skip_layers: Set[int],
        metrics: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, float, int]:
        if self.objective.metric_name not in metrics:
            raise KeyError(
                f"Metric '{self.objective.metric_name}' missing from evaluator output. "
                f"Available metrics: {sorted(metrics.keys())}"
            )

        active_layers = self.optimizer.search_space.num_layers - len(skip_layers)
        metric_value = float(metrics[self.objective.metric_name])

        objective_value = (
            self.objective.active_layers_weight * float(active_layers)
            + self.objective.metric_weight * metric_value
        )
        reward = -objective_value if self.objective.minimize else objective_value
        objectives = {"active_layers": float(active_layers), self.objective.metric_name: metric_value}
        return objectives, objective_value, reward, active_layers

    def run(self, num_iterations: int) -> OptimizationResult:
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        self.history = []

        for iteration in range(1, num_iterations + 1):
            proposed = self.optimizer.ask()
            skip_layers = self.optimizer.finalize_skip_layers(proposed)

            metrics = self.evaluator.evaluate(skip_layers)
            objectives, objective_value, reward, active_layers = self._compute_objectives_and_reward(
                skip_layers,
                metrics,
            )

            self.optimizer.tell(skip_layers=skip_layers, objectives=objectives, metrics=metrics, reward=reward)

            self.history.append(
                TrialRecord(
                    iteration=iteration,
                    skip_layers=set(skip_layers),
                    active_layers=active_layers,
                    objectives=objectives,
                    metrics=dict(metrics),
                    objective_value=objective_value,
                    reward=reward,
                )
            )

        best_trial = max(self.history, key=lambda x: x.reward)
        return OptimizationResult(best_trial=best_trial, history=self.history)
