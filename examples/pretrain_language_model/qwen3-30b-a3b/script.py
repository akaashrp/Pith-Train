"""
Pretrain Qwen3-30B-A3B with 2-way pipeline parallelism and 8-way expert parallelism.
"""

from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.tasks.pretrain_language_model import PretrainLanguageModelCfg, launch

cfg = PretrainLanguageModelCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_language_model/qwen3-30b-a3b/config.json")
training.optimizer = "Adam"
training.scheduler = "CosineAnnealing"
training.max_lr = 3.0e-4
training.min_lr = 1.0e-5
training.warmup_steps = 128
training.max_steps = 4096
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
# Optional weighted multi-source mixture (hot-reload enabled).
# Set ENABLE_DATA_MIXTURE=True and update paths as needed.
ENABLE_DATA_MIXTURE = False
if ENABLE_DATA_MIXTURE:
    training.dataset_sources = {
        "dclm": Path("workspace/datasets/dclm-baseline/toktxt/qwen3"),
        "math": Path("workspace/datasets/math/toktxt/qwen3"),
        "code": Path("workspace/datasets/code/toktxt/qwen3"),
    }
    training.dataset_mixture = {"dclm": 0.30, "math": 0.10, "code": 0.60}
    training.dataset_mixture_hot_reload_path = Path(
        "examples/pretrain_language_model/qwen3-30b-a3b/mixture.sample.json"
    )
    training.dataset_mixture_poll_interval_steps = 1
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = "disabled"
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/qwen3-30b-a3b")

# Wandb logging configuration. Comment out to disable.
logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = ""  # your wandb entity
logging.wandb.project = ""  # your wandb project
logging.wandb.name = "qwen3-30b-a3b"

if __name__ == "__main__":
    launch(cfg)
