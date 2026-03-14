from pathlib import Path

import numpy as np
import pytest

from pithtrain.modules.dataset import MemmapDataset, SourceDataset, WeightedMixtureDataset
from pithtrain.tasks.pretrain_language_model import (
    PretrainLanguageModelCfg,
    PretrainLanguageModelCtx,
    maybe_reload_dataset_mixture,
)


def _build_weighted_dataset(tmp_path: Path) -> WeightedMixtureDataset:
    sequence_length = 8
    tokens_a = np.arange(0, 8 * 64 + 1, dtype=np.int64)
    tokens_b = np.arange(1_000_000, 1_000_000 + 8 * 64 + 1, dtype=np.int64)
    path_a = Path(tmp_path, "a.bin")
    path_b = Path(tmp_path, "b.bin")
    with open(path_a, "wb") as fa:
        np.save(fa, tokens_a)
    with open(path_b, "wb") as fb:
        np.save(fb, tokens_b)
    source_a = SourceDataset("a", [MemmapDataset(path_a, sequence_length)])
    source_b = SourceDataset("b", [MemmapDataset(path_b, sequence_length)])
    return WeightedMixtureDataset({"a": source_a, "b": source_b}, seed=1, weights={"a": 0.5, "b": 0.5})


def test_reload_rejects_invalid_poll_interval(tmp_path: Path):
    cfg = PretrainLanguageModelCfg()
    cfg.training.dataset_mixture_poll_interval_steps = 0
    ctx = PretrainLanguageModelCtx()
    ctx.training.dataset = _build_weighted_dataset(tmp_path)
    ctx.training.step = 0

    with pytest.raises(ValueError):
        maybe_reload_dataset_mixture(cfg, ctx)


def test_reload_skips_when_not_polling_step(tmp_path: Path):
    cfg = PretrainLanguageModelCfg()
    cfg.training.dataset_mixture_poll_interval_steps = 8
    cfg.training.dataset_mixture_hot_reload_path = Path(tmp_path, "unused.json")
    ctx = PretrainLanguageModelCtx()
    ctx.training.dataset = _build_weighted_dataset(tmp_path)
    ctx.training.step = 3  # does not match poll cadence
    ctx.training.dataset_mixture_version = 0

    # Should return before any distributed communication path.
    maybe_reload_dataset_mixture(cfg, ctx)
    assert ctx.training.dataset_mixture_version == 0


def test_reload_skips_when_hot_reload_path_not_configured(tmp_path: Path):
    cfg = PretrainLanguageModelCfg()
    cfg.training.dataset_mixture_poll_interval_steps = 1
    cfg.training.dataset_mixture_hot_reload_path = None
    ctx = PretrainLanguageModelCtx()
    ctx.training.dataset = _build_weighted_dataset(tmp_path)
    ctx.training.step = 0
    ctx.training.dataset_mixture_version = 0

    # Path is None, so the helper is a no-op and should not touch version.
    maybe_reload_dataset_mixture(cfg, ctx)
    assert ctx.training.dataset_mixture_version == 0
