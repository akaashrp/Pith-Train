import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pithtrain.modules.dataset import MemmapDataset, SourceDataset, WeightedMixtureDataset
from pithtrain.tasks.pretrain_language_model import load_dataset_mixture_file


def _write_tokens(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, tokens)


def _build_source(root: Path, name: str, start: int, sequence_length: int, samples: int) -> SourceDataset:
    # len(tokens) = samples * sequence_length + 1 => MemmapDataset.__len__ == samples
    tokens = np.arange(start, start + samples * sequence_length + 1, dtype=np.int64)
    shard_path = Path(root, name, "shard0.bin")
    _write_tokens(shard_path, tokens)
    return SourceDataset(name, [MemmapDataset(shard_path, sequence_length)])


def _estimate_source_share(dataset: WeightedMixtureDataset, begin: int, end: int) -> float:
    # Source "a" starts at token range < 1_000_000 and source "b" >= 1_000_000.
    count_a = 0
    for idx in range(begin, end):
        tokens, _ = dataset[idx]
        assert isinstance(tokens, torch.Tensor)
        if int(tokens[0]) < 1_000_000:
            count_a += 1
    return count_a / max(1, end - begin)


def test_weighted_mixture_basic_and_validation(tmp_path: Path):
    sequence_length = 8
    source_a = _build_source(tmp_path, "a", start=0, sequence_length=sequence_length, samples=128)
    source_b = _build_source(
        tmp_path, "b", start=1_000_000, sequence_length=sequence_length, samples=128
    )
    dataset = WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=1234, weights={"a": 0.7, "b": 0.3}
    )

    assert dataset.current_weights() == {"a": 0.7, "b": 0.3}
    assert dataset.source_lengths() == {"a": 128, "b": 128}
    assert len(dataset) == 256

    with pytest.raises(ValueError):
        dataset.update_weights({"a": 0.5, "c": 0.5})
    with pytest.raises(ValueError):
        dataset.update_weights({"a": 0.5, "b": 0.49})
    with pytest.raises(ValueError):
        dataset.update_weights({"a": -0.1, "b": 1.1})


def test_weighted_mixture_is_deterministic_for_fixed_seed(tmp_path: Path):
    sequence_length = 8
    source_a = _build_source(tmp_path, "a", start=0, sequence_length=sequence_length, samples=64)
    source_b = _build_source(tmp_path, "b", start=1_000_000, sequence_length=sequence_length, samples=64)
    weights = {"a": 0.6, "b": 0.4}

    dataset1 = WeightedMixtureDataset({"a": source_a, "b": source_b}, seed=7, weights=weights)
    dataset2 = WeightedMixtureDataset({"a": source_a, "b": source_b}, seed=7, weights=weights)

    for idx in [0, 1, 7, 31, 100, 777]:
        tokens1, labels1 = dataset1[idx]
        tokens2, labels2 = dataset2[idx]
        assert torch.equal(tokens1, tokens2)
        assert torch.equal(labels1, labels2)


def test_weighted_mixture_respects_weights_and_updates(tmp_path: Path):
    sequence_length = 8
    source_a = _build_source(tmp_path, "a", start=0, sequence_length=sequence_length, samples=256)
    source_b = _build_source(
        tmp_path, "b", start=1_000_000, sequence_length=sequence_length, samples=256
    )
    dataset = WeightedMixtureDataset(
        {"a": source_a, "b": source_b}, seed=2026, weights={"a": 0.8, "b": 0.2}
    )

    share_a_before = _estimate_source_share(dataset, begin=0, end=20000)
    assert abs(share_a_before - 0.8) < 0.03

    dataset.update_weights({"a": 0.2, "b": 0.8})
    share_a_after = _estimate_source_share(dataset, begin=20000, end=40000)
    assert abs(share_a_after - 0.2) < 0.03


def test_load_dataset_mixture_file_validation(tmp_path: Path):
    valid = Path(tmp_path, "mixture.json")
    valid.write_text(json.dumps({"dclm": 0.3, "math": 0.7}))
    parsed = load_dataset_mixture_file(valid)
    assert parsed == {"dclm": 0.3, "math": 0.7}

    not_object = Path(tmp_path, "bad-list.json")
    not_object.write_text(json.dumps([["dclm", 1.0]]))
    with pytest.raises(ValueError):
        load_dataset_mixture_file(not_object)

    non_numeric = Path(tmp_path, "bad-value.json")
    non_numeric.write_text(json.dumps({"dclm": "abc"}))
    with pytest.raises(ValueError):
        load_dataset_mixture_file(non_numeric)
