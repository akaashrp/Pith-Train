from typing import Literal

import torch.nn as nn


class ModelImplMode:
    """
    Model Implementation Mode. Turn on the reference implementation to validate
    the correctness of the optimized, possibly distributed implementation.
    """

    use_reference_fwd = False
    fp8_training: Literal["deep-gemm", "disabled"] = "disabled"
    moe_backend: Literal["dualpipe", "sonic-moe"] = "dualpipe"


def get_linear_cls():
    """Return the appropriate Linear class based on ModelImplMode.fp8_training."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8Linear

        return FP8Linear
    return nn.Linear


def get_group_linear_cls():
    """Return the GroupLinear class selected by the active backend mode."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8GroupLinear

        return FP8GroupLinear
    if ModelImplMode.moe_backend == "sonic-moe":
        from pithtrain.layers.sonic_moe_group_linear import SonicMoEGroupLinear

        return SonicMoEGroupLinear
    from pithtrain.layers.group_linear import GroupLinear

    return GroupLinear
