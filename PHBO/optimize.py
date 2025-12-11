#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.logging import logger
from torch import Tensor

from PHBO.gen import gen_candidates_scipy, gen_candidates_torch
from PHBO.initializers import gen_batch_initial_conditions


INIT_OPTION_KEYS = {
    "alpha",
    "batch_limit",
    "eta",
    "init_batch_limit",
    "nonnegative",
    "n_burnin",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_prob_perturb",
    "sample_around_best_prob_perturb",
    "seed",
    "thinning",
}


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    stochastic: bool = False, 
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    if not (bounds.ndim == 2 and bounds.shape[0] == 2):
        raise ValueError(
            f"bounds should be a `2 x d` tensor, current shape: {list(bounds.shape)}."
        )
    
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "`return_best_only=False` only supported for joint optimization."
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
                stochastic=stochastic, 
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    if batch_initial_conditions is None:
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "`batch_initial_conditions` must be given if there are non-linear "
                "inequality constraints."
            )
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

        ic_gen = gen_batch_initial_conditions
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    batch_limit: int = options.get(
        "batch_limit", num_restarts if not nonlinear_inequality_constraints else 1
    )

    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batched_ics = batch_initial_conditions.split(batch_limit)
    gen_kwargs = {}
    if stochastic:
        gen_candidates = gen_candidates_torch
    else:
        gen_candidates = gen_candidates_scipy
        gen_kwargs.update(
            {
                "inequality_constraints": inequality_constraints,
                "equality_constraints": equality_constraints,
                "nonlinear_inequality_constraints": nonlinear_inequality_constraints,
            }
        )
    for i, batched_ics_ in enumerate(batched_ics):
        batch_candidates_curr, batch_acq_values_curr = gen_candidates(
            initial_conditions=batched_ics_,
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
            fixed_features=fixed_features,
            **gen_kwargs,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
        
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values

