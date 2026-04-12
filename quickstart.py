import pickle

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_processor, get_proprio_projector, get_vla, get_vla_action
from openvla_evaluator import OpenVLAEvaluator
from optimization import GreedyLayerSkipOptimizer, ObjectiveConfig, OptimizationLoop, ParEGOOptimizer, SearchSpaceConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# ===== Main Script =====

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=False,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
# action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(
    cfg, vla, processor, observation, observation["task_description"], proprio_projector=proprio_projector
)
print("Generated action chunk:")
for act in actions:
    print(act)

# ===== Run Layer Skipping Test =====
# Uncomment the line below to run the self-speculative decoding layer skipping test
#test_layer_skipping(vla, cfg, processor, observation, proprio_projector, always_exclude_layers={0, 31},
#                    always_include_layers={25})


# ===== Run ParEGO Optimization =====
print("\n" + "=" * 80)
print("PAREGO LAYER-SKIP OPTIMIZATION")
print("=" * 80)

evaluator = OpenVLAEvaluator(cfg, vla, processor, observation, proprio_projector)
search_space = SearchSpaceConfig(
    num_layers=len(vla.language_model.model.layers),
    min_skip_layers=1,
    max_skip_layers=8,
    contiguous_only=False,
)
objective = ObjectiveConfig(
    metric_name=evaluator.metric_name,
    active_layers_weight=0.05,
    metric_weight=1.0,
    minimize=True,
)
optimizer = ParEGOOptimizer(
    search_space=search_space,
    objective=objective,
    seed=0,
    candidate_pool_size=64,
)
loop = OptimizationLoop(evaluator=evaluator, optimizer=optimizer, objective=objective)
result = loop.run(num_iterations=6)

print(f"Best reward: {result.best_trial.reward:.6f}")
print(f"Best objective value: {result.best_trial.objective_value:.6f}")
print(f"Best active layers: {result.best_trial.active_layers}")
print(f"Best skip layers: {sorted(result.best_trial.skip_layers)}")
print(f"Best metric ({evaluator.metric_name}): {result.best_trial.metrics[evaluator.metric_name]:.6f}")

print("\nTop trials:")
for trial in sorted(result.history, key=lambda item: item.reward, reverse=True)[:5]:
    metric_value = trial.metrics[evaluator.metric_name]
    print(
        f"iter={trial.iteration} reward={trial.reward:.6f} objective={trial.objective_value:.6f} "
        f"active={trial.active_layers} skip={sorted(trial.skip_layers)} metric={metric_value:.6f}"
    )


# ===== Run Greedy Layer-Skip Search =====
print("\n" + "=" * 80)
print("GREEDY LAYER-SKIP SEARCH")
print("=" * 80)

greedy_search_space = SearchSpaceConfig(
    num_layers=len(vla.language_model.model.layers),
    min_skip_layers=0,
    max_skip_layers=None,
    contiguous_only=False,
)
greedy = GreedyLayerSkipOptimizer(search_space=greedy_search_space, metric_name="l2_mean")
greedy_result = greedy.run(evaluator=evaluator)

print(f"Best greedy metric (l2_mean): {greedy_result.best_trial.objective_value:.6f}")
print(f"Best greedy skip layers: {sorted(greedy_result.best_trial.skip_layers)}")

print("\nGreedy timeline:")
for trial in greedy_result.history:
    print(
        f"step={trial.iteration:02d} active={trial.active_layers:02d} "
        f"skip={sorted(trial.skip_layers)} l2_mean={trial.metrics['l2_mean']:.6f}"
    )
