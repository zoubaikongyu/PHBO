from copy import deepcopy
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from PHBO.problem_base import DiscreteTestProblem

class Enzyme(DiscreteTestProblem):
    dim = 5
    _bounds = [[0, 6],[0, 16], [0, 1],[0, 1], [0, 1],]

    def __init__(self,noise_std: Optional[float] = None,
        negate: bool = False,continuous: bool = False,) -> None:
        integer_indices = []
        if not continuous:
            self._orig_cont_bounds_list = deepcopy(self._bounds)
            for i in range(2, 5):
                n_ordinal = int((self._bounds[i][1] - self._bounds[i][0]) / 0.005)
                self._bounds[i][0] = 0
                self._bounds[i][1] = n_ordinal - 1
                integer_indices.append(i)
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=integer_indices,
            categorical_indices=[0,1],
        )
        self.continuous = continuous
        if not continuous:
            self.register_buffer(
                "_orig_cont_bounds_tensor",
                torch.tensor(self._orig_cont_bounds_list).t(),
            )

    def evaluate_true(self, X: Tensor,) -> Tensor:
        raise NotImplementedError("need experiment")

