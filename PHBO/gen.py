#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation.utils import _remove_fixed_features_from_optimization
from botorch.optim.parameter_constraints import (
    NLC_TOL,
    _arrayify,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs, columnwise_clamp, fix_features
from scipy.optimize import minimize
from torch import Tensor

def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    options: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    options = options or {}
    reduced_domain = False
    if fixed_features:
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "Fixed features are not supported when non-linear inequality "
                "constraints are given."
            )
        if not (inequality_constraints or equality_constraints):
            reduced_domain = True
        else:
            reduced_domain = None not in fixed_features.values()
    if reduced_domain:
        _no_fixed_features = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )
        clamped_candidates, batch_acquisition = gen_candidates_scipy(
            initial_conditions=_no_fixed_features.initial_conditions,
            acquisition_function=_no_fixed_features.acquisition_function,
            lower_bounds=_no_fixed_features.lower_bounds,
            upper_bounds=_no_fixed_features.upper_bounds,
            inequality_constraints=_no_fixed_features.inequality_constraints,
            equality_constraints=_no_fixed_features.equality_constraints,
            options=options,
            fixed_features=None,
        )
        clamped_candidates = _no_fixed_features.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition

    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )
    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=clamped_candidates.shape,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )
    with_grad = options.get("with_grad", True)
    if with_grad:
        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            X_fix = fix_features(X, fixed_features=fixed_features)
            loss = f(X_fix).sum()
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. This often indicates numerical issues."
                )
                if initial_conditions.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf
    else:
        def f_np_wrapper(x: np.ndarray, f: Callable):
            X = torch.from_numpy(x).to(initial_conditions).view(shapeX).contiguous()
            with torch.no_grad():
                X_fix = fix_features(X=X, fixed_features=fixed_features)
                loss = f(X_fix).sum()
            fval = loss.item()
            return fval

    if nonlinear_inequality_constraints:
        if not (len(shapeX) == 3 and shapeX[:2] == torch.Size([1, 1])):
            raise ValueError(
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given."
            )
        constraints += make_scipy_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            f_np_wrapper=f_np_wrapper,
            x0=x0,
        )
    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x)

    res = minimize(
        fun=f_np_wrapper,
        args=(f,),
        x0=x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=with_grad,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={
            k: v
            for k, v in options.items()
            if k
            not in [
                "method",
                "callback",
                "with_grad",
                "sample_around_best_sigma",
                "sample_around_best_subset_sigma",
            ]
        },
    )
    candidates = fix_features(
        X=torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX),
        fixed_features=fixed_features,
    )

    if nonlinear_inequality_constraints and any(
        nlc(candidates.view(-1)) < NLC_TOL for nlc in nonlinear_inequality_constraints
    ):
        candidates = torch.from_numpy(x0).to(candidates).reshape(shapeX)
        warnings.warn(
            "SLSQP failed to converge to a solution the satisfies the non-linear "
            "constraints. Returning the feasible starting point."
        )

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)
    return clamped_candidates, batch_acquisition

def gen_candidates_torch(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    options: Optional[Dict[str, Union[float, str]]] = None,
    callback: Optional[Callable[[int, Tensor, Tensor], NoReturn]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
    optimizer=torch.optim.Adam,
) -> Tuple[Tensor, Tensor]:
    
    options = options or {}

    if fixed_features:
        subproblem = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=None,
            equality_constraints=None,
        )
        clamped_candidates, batch_acquisition = gen_candidates_torch(
            initial_conditions=subproblem.initial_conditions,
            acquisition_function=subproblem.acquisition_function,
            lower_bounds=subproblem.lower_bounds,
            upper_bounds=subproblem.upper_bounds,
            optimizer=optimizer,
            options=options,
            callback=callback,
            fixed_features=None,
        )
        clamped_candidates = subproblem.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition

    _clamp = partial(columnwise_clamp, lower=lower_bounds, upper=upper_bounds)
    clamped_candidates = _clamp(initial_conditions).requires_grad_(True)
    lr = options.get("lr", 0.025)
    _optimizer = optimizer(params=[clamped_candidates], lr=lr)

    i = 0
    stop = False

    if options.get("rel_tol") is None:
        options["rel_tol"] = float("-inf")
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **options)
    )
    decay = options.get("decay", False)
    if callback is None:
        callback = options.get("callback")
    while not stop:   
        i += 1
        with torch.no_grad():
            X = _clamp(clamped_candidates).requires_grad_(True)
        loss = -acquisition_function(X).sum()
        grad = torch.autograd.grad(loss, X)[0]
        if callback:
            callback(i, loss, grad, X)

        def assign_grad():
            _optimizer.zero_grad()
            clamped_candidates.grad = grad
            return loss

        _optimizer.step(assign_grad)
        if decay:
            _optimizer = optimizer(params=[clamped_candidates], lr=lr / i**0.7)
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    clamped_candidates = clamped_candidates.detach()
    clamped_candidates = _clamp(clamped_candidates)
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)
    return clamped_candidates, batch_acquisition

