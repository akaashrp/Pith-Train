import importlib.util
import sys
import types

import pytest
import torch

HAS_TRITON = importlib.util.find_spec("triton") is not None
pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")

if HAS_TRITON:
    from pithtrain.layers.factory import ModelImplMode, get_group_linear_cls
    from pithtrain.layers.group_linear import GroupLinear
    from pithtrain.layers.sonic_moe_group_linear import (
        SonicMoEGroupLinear,
        _reset_sonicmoe_cache_for_testing,
    )
    from pithtrain.modules.training import TrainingCfg, validate_backend_selection


def _build_fake_sonicmoe_module() -> types.ModuleType:
    mod = types.ModuleType("sonicmoe")
    functional = types.ModuleType("sonicmoe.functional")

    def gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        cu_seqlens_m: torch.Tensor | None = None,
        dynamic_scheduler: bool = False,
    ) -> torch.Tensor:
        del dynamic_scheduler
        assert cu_seqlens_m is not None
        out = torch.empty((a.shape[0], b.shape[-1]), dtype=a.dtype, device=a.device)
        for g in range(cu_seqlens_m.numel() - 1):
            start = int(cu_seqlens_m[g].item())
            end = int(cu_seqlens_m[g + 1].item())
            if end > start:
                out[start:end] = a[start:end] @ b[g]
        return out

    functional.gemm = gemm
    mod.functional = functional
    return mod


def test_factory_functions_bf16_dualpipe_mode():
    prev_fp8 = ModelImplMode.fp8_training
    prev_moe = ModelImplMode.moe_backend
    try:
        ModelImplMode.fp8_training = "disabled"
        ModelImplMode.moe_backend = "dualpipe"
        assert get_group_linear_cls() is GroupLinear
    finally:
        ModelImplMode.fp8_training = prev_fp8
        ModelImplMode.moe_backend = prev_moe


def test_factory_functions_sonic_mode(monkeypatch):
    monkeypatch.setitem(sys.modules, "sonicmoe", _build_fake_sonicmoe_module())
    _reset_sonicmoe_cache_for_testing()

    prev_fp8 = ModelImplMode.fp8_training
    prev_moe = ModelImplMode.moe_backend
    try:
        ModelImplMode.fp8_training = "disabled"
        ModelImplMode.moe_backend = "sonic-moe"
        assert get_group_linear_cls() is SonicMoEGroupLinear
    finally:
        ModelImplMode.fp8_training = prev_fp8
        ModelImplMode.moe_backend = prev_moe
        _reset_sonicmoe_cache_for_testing()


def test_sonic_group_linear_forward_and_empty_input(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setitem(sys.modules, "sonicmoe", _build_fake_sonicmoe_module())
    _reset_sonicmoe_cache_for_testing()

    layer = SonicMoEGroupLinear(3, 4, 5)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))

    x = torch.randn(6, 4)
    grouped_mm_offs = torch.tensor([2, 4, 6], dtype=torch.int32)
    y = layer(x, grouped_mm_offs)

    y_ref_parts = []
    start = 0
    for g, end in enumerate(grouped_mm_offs.tolist()):
        y_ref_parts.append(x[start:end] @ layer.weight[g].T)
        start = end
    y_ref = torch.cat(y_ref_parts, dim=0)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)

    x0 = torch.randn(0, 4)
    y0 = layer(x0, torch.tensor([0, 0, 0], dtype=torch.int32))
    assert y0.shape == (0, 5)


def test_validate_backend_selection_rejects_sonic_with_fp8():
    cfg = TrainingCfg()
    cfg.fp8_training = "deep-gemm"
    cfg.moe_backend = "sonic-moe"
    with pytest.raises(ValueError, match="only supports fp8_training='disabled'"):
        validate_backend_selection(cfg)


def test_validate_backend_selection_requires_sonic_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sonicmoe", types.ModuleType("sonicmoe"))
    _reset_sonicmoe_cache_for_testing()

    cfg = TrainingCfg()
    cfg.fp8_training = "disabled"
    cfg.moe_backend = "sonic-moe"
    with pytest.raises(ImportError, match="sonicmoe.functional.gemm"):
        validate_backend_selection(cfg)
