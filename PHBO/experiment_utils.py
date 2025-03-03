#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from math import log
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.test_functions.synthetic import Hartmann
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from statsmodels.distributions.empirical_distribution import ECDF

from PHBO.input import (
    OneHotToNumeric,
    Round,
)
from PHBO.kernels import get_kernel
from PHBO.probabilistic_reparameterization import (
    MCProbabilisticReparameterization,
)
from PHBO.problem_base import (
    DiscreteTestProblem,
    DiscretizedBotorchTestProblem
)
from enzyme.problem_enzyme import Enzyme


def eval_problem(X: Tensor, old_x:Tensor, old_y:Tensor, base_function: DiscreteTestProblem, fantasy_point=False) -> Tensor:
    if fantasy_point:
        distances = F.pairwise_distance(old_x, X)
        _, indices = torch.topk(distances, 3, largest=False)
        selected_y = old_y[indices]
        Y = torch.mean(selected_y)
        Y = Y.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        return Y
    else:
        X_numeric = torch.zeros(
            *X.shape[:-1], 
            base_function.bounds.shape[-1],
            dtype=X.dtype,
            device=X.device,
        )
        X_numeric[..., base_function.integer_indices] = X[
            ..., base_function.integer_indices
        ]
        X_numeric[..., base_function.cont_indices] = X[..., base_function.cont_indices]
        start_idx = None
        for i, cardinality in base_function.categorical_features.items():
            if start_idx is None:
                start_idx = i
            end_idx = start_idx + cardinality
            X_numeric[..., i] = (
                X[..., start_idx:end_idx].argmax(dim=-1).to(dtype=X_numeric.dtype)
            )
            start_idx = end_idx
        if len(base_function.categorical_features) > 0:
            X_numeric[..., base_function.categorical_indices] = normalize(
                X_numeric[..., base_function.categorical_indices],
                base_function.categorical_bounds,
            )
        X_numeric = unnormalize(X_numeric, base_function.bounds)
        Y = base_function(X_numeric)
        if Y.ndim == X_numeric.ndim - 1:
            Y = Y.unsqueeze(-1)
        return Y

def get_exact_rounding_func(
    bounds: Tensor,
    integer_indices: Optional[List[int]] = None,
    categorical_features: Optional[Dict[int, int]] = None,
    initialization: bool = False,
) -> ChainedInputTransform:
    if initialization:
        init_bounds = bounds.clone()
        init_bounds[0, integer_indices] -= 0.4999
        init_bounds[1, integer_indices] += 0.4999
    else:
        init_bounds = bounds

    tfs = OrderedDict()
    if integer_indices is not None and len(integer_indices) > 0:
        tfs["unnormalize_tf"] = Normalize(
            d=bounds.shape[1],
            bounds=init_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=True,
        )
        
    tfs["round"] = Round(
        approximate=False,
        transform_on_train=False,
        transform_on_fantasize=False,
        integer_indices=integer_indices,
        categorical_features=categorical_features,
    )

    if integer_indices is not None and len(integer_indices) > 0:
        tfs["normalize_tf"] = Normalize(
            d=bounds.shape[1],
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=False,
        )
    tf = ChainedInputTransform(**tfs)
    tf.to(dtype=bounds.dtype, device=bounds.device)
    tf.eval()
    return tf

def generate_initial_data(
    n: int,
    base_function: DiscreteTestProblem,
    bounds: Tensor,
    tkwargs: dict,
    init_exact_rounding_func: ChainedInputTransform,
) -> Tuple[Tensor, Tensor]:
    raw_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    train_x = init_exact_rounding_func(raw_x)
    train_obj = eval_problem(X=train_x, old_x = None, old_y = None, base_function=base_function)
    return train_x, train_obj

def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
    binary_dims: List[int],
    categorical_features: Optional[List[int]] = None,
    use_fixed_noise: bool = True,
    kernel_type: str = "mixed",
    use_ard_binary: bool = False,
) -> Tuple[
    Union[ExactMarginalLogLikelihood, SumMarginalLogLikelihood],
    Union[FixedNoiseGP, SingleTaskGP, ModelListGP],
    Optional[List[ECDF]],
]:
    base_model_class = FixedNoiseGP if use_fixed_noise else SingleTaskGP
    if use_fixed_noise: 
        train_Yvar = torch.full_like(train_y, 1e-7) * train_y.std(dim=0).pow(2)
    if kernel_type == "mixed_categorical":
        input_transform = OneHotToNumeric(categorical_features=categorical_features)
        input_transform.eval()
        train_x = input_transform(train_x)
    if categorical_features is None:
        categorical_dims = []
    else:
        categorical_dims = list(categorical_features.keys())
    categorical_transformed_features = categorical_features
    model_kwargs = []
    for i in range(train_y.shape[-1]): 
        transformed_x = train_x
        input_transform = None
        model_kwargs.append(
            {
                "train_X": train_x,
                "train_Y": train_y[..., i : i + 1],
                "covar_module": get_kernel(
                    dim=transformed_x.shape[-1],
                    binary_dims=binary_dims,
                    categorical_transformed_features=categorical_transformed_features,
                    use_ard_binary=use_ard_binary,
                ),
                "input_transform": input_transform,
            }
        )
        if use_fixed_noise:
            model_kwargs[i]["train_Yvar"] = train_Yvar[..., i : i + 1]
        else:
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3),
            )
    models = [base_model_class(**model_kwargs[i]) for i in range(train_y.shape[-1])]
    if len(models) > 1: 
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        model = models[0]
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def get_acqf(
    label: str,
    model: GPyTorchModel,
    X_baseline: Tensor,
    train_Y: Tensor,
    iteration: int,
    tkwargs: dict,
    base_function: DiscreteTestProblem,
    **kwargs,
) -> Union[AcquisitionFunction, List[AcquisitionFunction]]:
    
    if label[-2:] == "ei":
        obj = train_Y[..., 0]
        acq_func = ExpectedImprovement(
            model=model, 
            best_f=obj.max()
        )
    elif label[-3:] == "ucb":
        beta = 0.2 * X_baseline.shape[-1] * log(2 * iteration)
        acq_func = UpperConfidenceBound(
            model=model,
            beta=beta,
        )
    else: 
        raise NotImplementedError
    
    if "pr" in label: 
        acq_func = MCProbabilisticReparameterization( 
            acq_function=acq_func,  
            integer_indices=base_function.integer_indices.cpu().tolist(),
            integer_bounds=base_function.integer_bounds,
            categorical_features=base_function.categorical_features,
            dim=X_baseline.shape[-1],
            batch_limit=kwargs.get("pr_batch_limit", 32),
            mc_samples=kwargs.get("pr_mc_samples", 1024),
            apply_numeric=kwargs.get("apply_numeric", False),
            tau=kwargs.get("pr_tau", 0.1),
            grad_estimator=kwargs.get("pr_grad_estimator", "reinforce"),
        )
    return acq_func

def get_problem(name: str, dim: Optional[int] = None, **kwargs) -> DiscreteTestProblem:
    if name == "enzyme":
        return Enzyme(
            negate=True,
            continuous=kwargs.get("continuous", False),
        )
    else:
        raise ValueError(f"Unknown function name: {name}!")
