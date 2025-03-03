#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchWarning,
    SamplingWarning,
)
from botorch.optim.utils import fix_features, get_X_baseline
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import (
    batched_multinomial,
    draw_sobol_samples,
    get_polytope_samples,
    manual_seed,
)
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
from torch.distributions import Normal
from torch.quasirandom import SobolEngine


def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    seed: int = None,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
) -> Tensor:

    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds_cpu = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if inequality_constraints is None and equality_constraints is None:
                if effective_dim <= SobolEngine.MAXDIM:
                    X_rnd = draw_sobol_samples(bounds=bounds_cpu, n=n, q=q, seed=seed)
                else:
                    with manual_seed(seed):
                        # load on cpu
                        X_rnd_nlzd = torch.rand(
                            n, q, bounds_cpu.shape[-1], dtype=bounds.dtype
                        )
                    X_rnd = bounds_cpu[0] + (bounds_cpu[1] - bounds_cpu[0]) * X_rnd_nlzd
            else:
                X_rnd = (
                    get_polytope_samples(
                        n=n * q,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        seed=seed,
                        n_burnin=options.get("n_burnin", 10000),
                        thinning=options.get("thinning", 32),
                    )
                    .view(n, q, -1)
                    .cpu()
                )
            if options.get("sample_around_best", False):
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                    prob_perturb=options.get("sample_around_best_prob_perturb"),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )
            X_rnd = fix_features(X_rnd, fixed_features=fixed_features)
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()  
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions
    # endregion


def initialize_q_batch(X: Tensor, Y: Tensor, n: int, eta: float = 1.0) -> Tensor:
    n_samples = X.shape[0]
    batch_shape = X.shape[1:-2] or torch.Size()
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X

    Ystd = Y.std(dim=0)
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    max_val, max_idx = torch.max(Y, dim=0)
    Z = (Y - Y.mean(dim=0)) / Ystd
    etaZ = eta * Z
    weights = torch.exp(etaZ)
    while torch.isinf(weights).any():
        etaZ *= 0.5
        weights = torch.exp(etaZ)
    if batch_shape == torch.Size():
        idcs = torch.multinomial(weights, n)
    else:
        idcs = batched_multinomial(
            weights=weights.permute(*range(1, len(batch_shape) + 1), 0), num_samples=n
        ).permute(-1, *range(len(batch_shape)))
    # make sure we get the maximum
    if max_idx not in idcs:
        idcs[-1] = max_idx
    if batch_shape == torch.Size():
        return X[idcs]
    else:
        return X.gather(
            dim=0, index=idcs.view(*idcs.shape, 1, 1).expand(n, *X.shape[1:])
        )


def initialize_q_batch_nonneg(
    X: Tensor, Y: Tensor, n: int, eta: float = 1.0, alpha: float = 1e-4
) -> Tensor:
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X

    max_val, max_idx = torch.max(Y, dim=0)
    if torch.any(max_val <= 0):
        warnings.warn(
            "All acquisition values for raw sampled points are nonpositive, so "
            "initial conditions are being selected randomly.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    pos = Y > 0
    num_pos = pos.sum().item()
    if num_pos < n:
        remaining_indices = (~pos).nonzero(as_tuple=False).view(-1)
        rand_indices = torch.randperm(remaining_indices.shape[0], device=Y.device)
        sampled_remaining_indices = remaining_indices[rand_indices[: n - num_pos]]
        pos[sampled_remaining_indices] = 1
        return X[pos]
    alpha_pos = Y >= alpha * max_val
    while alpha_pos.sum() < n:
        alpha = 0.1 * alpha
        alpha_pos = Y >= alpha * max_val
    alpha_pos_idcs = torch.arange(len(Y), device=Y.device)[alpha_pos]
    weights = torch.exp(eta * (Y[alpha_pos] / max_val - 1))
    idcs = alpha_pos_idcs[torch.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]


def sample_points_around_best(
    acq_function: AcquisitionFunction,
    n_discrete_points: int,
    sigma: Tensor,
    bounds: Tensor,
    best_pct: float = 5.0,
    subset_sigma: float = 1e-1,
    prob_perturb: Optional[float] = None,
) -> Optional[Tensor]:
    X = get_X_baseline(acq_function=acq_function)
    if X is None:
        return
    with torch.no_grad():
        try:
            posterior = acq_function.model.posterior(X)
        except AttributeError:
            warnings.warn(
                "Failed to sample around previous best points.",
                BotorchWarning,
            )
            return
        mean = posterior.mean
        while mean.ndim > 2:
            mean = mean.mean(dim=0)
        try:
            f_pred = acq_function.objective(mean)
        except (AttributeError, TypeError):
            f_pred = mean
        if hasattr(acq_function, "maximize"):
            if not acq_function.maximize:
                f_pred = -f_pred
        try:
            constraints = acq_function.constraints
            if constraints is not None:
                neg_violation = -torch.stack(
                    [c(mean).clamp_min(0.0) for c in constraints], dim=-1
                ).sum(dim=-1)
                feas = neg_violation == 0
                if feas.any():
                    f_pred[~feas] = float("-inf")
                else:
                    f_pred = neg_violation
        except AttributeError:
            pass
        if f_pred.ndim == mean.ndim and f_pred.shape[-1] > 1:
            is_pareto = is_non_dominated(f_pred)
            best_X = X[is_pareto]
        else:
            if f_pred.shape[-1] == 1:
                f_pred = f_pred.squeeze(-1)
            n_best = max(1, round(X.shape[0] * best_pct / 100))
            best_idcs = torch.topk(f_pred, n_best).indices.view(-1)
            best_X = X[best_idcs]
    n_trunc_normal_points = (
        n_discrete_points // 2 if best_X.shape[-1] > 20 else n_discrete_points
    )
    perturbed_X = sample_truncated_normal_perturbations(
        X=best_X,
        n_discrete_points=n_trunc_normal_points,
        sigma=sigma,
        bounds=bounds,
    )
    if best_X.shape[-1] > 20 or prob_perturb is not None:
        perturbed_subset_dims_X = sample_perturbed_subset_dims(
            X=best_X,
            bounds=bounds,
            n_discrete_points=n_discrete_points - n_trunc_normal_points,
            sigma=sigma,
            prob_perturb=prob_perturb,
        )
        perturbed_X = torch.cat([perturbed_X, perturbed_subset_dims_X], dim=0)
        perm = torch.randperm(perturbed_X.shape[0], device=X.device)
        perturbed_X = perturbed_X[perm]
    return perturbed_X


def sample_truncated_normal_perturbations(
    X: Tensor,
    n_discrete_points: int,
    sigma: Tensor,
    bounds: Tensor,
    qmc: bool = True,
) -> Tensor:
    X = normalize(X, bounds=bounds)
    d = X.shape[1]
    if X.shape[0] > 1:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X = X[rand_indices]
    if qmc:
        std_bounds = torch.zeros(2, d, dtype=X.dtype, device=X.device)
        std_bounds[1] = 1
        u = draw_sobol_samples(bounds=std_bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand((n_discrete_points, d), dtype=X.dtype, device=X.device)
    a = -X
    b = 1 - X
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    perturbed_X = (X + perturbation).clamp(0.0, 1.0)
    return unnormalize(perturbed_X, bounds=bounds)


def sample_perturbed_subset_dims(
    X: Tensor,
    bounds: Tensor,
    n_discrete_points: int,
    sigma: Union[Tensor, float] = 1e-1,
    qmc: bool = True,
    prob_perturb: Optional[float] = None,
) -> Tensor:
    if bounds.ndim != 2:
        raise BotorchTensorDimensionError("bounds must be a `2 x d`-dim tensor.")
    elif X.ndim != 2:
        raise BotorchTensorDimensionError("X must be a `n x d`-dim tensor.")
    d = bounds.shape[-1]
    if prob_perturb is None:
        prob_perturb = min(20.0 / d, 1.0)

    if X.shape[0] == 1:
        X_cand = X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X_cand = X[rand_indices]
    pert = sample_truncated_normal_perturbations(
        X=X_cand,
        n_discrete_points=n_discrete_points,
        sigma=sigma,
        bounds=bounds,
        qmc=qmc,
    )
    mask = (
        torch.rand(
            n_discrete_points,
            d,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=bounds.device)]
    X_cand[mask] = pert[mask]
    return X_cand
