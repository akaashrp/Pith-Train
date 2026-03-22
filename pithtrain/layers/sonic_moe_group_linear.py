import importlib
from typing import Callable, Optional

import torch
import torch.nn as nn

_SONIC_GEMM: Optional[Callable[..., torch.Tensor]] = None


def _make_import_error(detail: str) -> ImportError:
    msg = (
        "moe_backend='sonic-moe' requires the 'sonic-moe' package with "
        "'sonicmoe.functional.gemm'. Install it by running: uv sync --extra sonic"
    )
    if detail:
        msg = f"{msg}. Detail: {detail}"
    return ImportError(msg)


def _load_sonic_gemm() -> Callable[..., torch.Tensor]:
    global _SONIC_GEMM
    gemm = _SONIC_GEMM
    if gemm is None:
        try:
            gemm = importlib.import_module("sonicmoe.functional").gemm
        except ImportError as exc:
            raise _make_import_error("failed to import module 'sonicmoe.functional'") from exc
        _SONIC_GEMM = gemm
    return gemm


def ensure_sonicmoe_available() -> None:
    _load_sonic_gemm()


def _reset_sonicmoe_cache_for_testing() -> None:
    global _SONIC_GEMM
    _SONIC_GEMM = None


def _make_cu_seqlens(
    grouped_mm_offs: torch.Tensor, num_groups: int, device: torch.device
) -> torch.Tensor:
    grouped_mm_offs = grouped_mm_offs.to(device=device, dtype=torch.int32)
    if grouped_mm_offs.numel() == num_groups + 1:
        return grouped_mm_offs
    zero = torch.zeros((1,), dtype=torch.int32, device=grouped_mm_offs.device)
    return torch.cat((zero, grouped_mm_offs), dim=0)


class SonicMoEGroupLinear(nn.Module):
    """
    Grouped linear layer backed by SonicMoE's grouped GEMM (`sonicmoe.functional.gemm`).
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))
        self._gemm = _load_sonic_gemm()

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: Optional[list] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ks, ks_tensor, group_indices
        if input.shape[0] == 0:
            # Preserve autograd graph on empty inputs (same contract as GroupLinear).
            return input @ self.weight[0].T

        cu_seqlens_m = _make_cu_seqlens(grouped_mm_offs, self.num_groups, input.device)
        weight_kt = self.weight.transpose(1, 2)
        return self._gemm(
            input,
            weight_kt,
            cu_seqlens_m=cu_seqlens_m,
            dynamic_scheduler=False,
        )
