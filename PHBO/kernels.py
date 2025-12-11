#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
import torch
from botorch.models.kernels import CategoricalKernel
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel


def get_kernel(
    dim: int,
    binary_dims: List[int],
    categorical_transformed_features: Dict[int, int],
    use_ard_binary: bool = False,
) -> Optional[Kernel]:
    categorical_dims = list(categorical_transformed_features.keys())
    cont_dims = list(set(list(range(dim))) - set(binary_dims))
    cont_dims = list(set(cont_dims) - set(categorical_dims))
    kernels = []
    if len(cont_dims) > 0:
        kernels.append(
            MaternKernel(
                nu=2.5,
                ard_num_dims=len(cont_dims),
                active_dims=cont_dims,
                lengthscale_constraint=Interval(0.1, 20.0),
            )
        )
    if len(binary_dims) > 0:
        kernels.append(
            MaternKernel(
                nu=2.5,
                ard_num_dims=len(binary_dims) if use_ard_binary else None,
                active_dims=binary_dims,
                lengthscale_constraint=Interval(0.1, 20.0),
            )
        )
    if len(categorical_dims) > 0:
        kernels.append(
            CategoricalKernel(
                ard_num_dims=len(categorical_dims),
                active_dims=categorical_dims,
                lengthscale_constraint=Interval(1e-3, 20.0),
            )
        )
    prod_kernel = kernels[0]
    sum_kernel = kernels[0]
    for k in kernels[1:]:
        prod_kernel *= k
        prod_kernel *= k 
        sum_kernel += k
    return ScaleKernel(prod_kernel) + ScaleKernel(sum_kernel)
