#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
from time import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.utils import is_nonnegative
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from PHBO.experiment_utils import (
    eval_problem,
    generate_initial_data,
    get_acqf,
    get_exact_rounding_func,
    get_problem,
    initialize_model,
)
from PHBO.optimize import optimize_acqf
from data_record.feedback import feedback
from PHBO.probabilistic_reparameterization import (
    AbstractProbabilisticReparameterization,
)

def run_one_replication(
    seed: int,
    label: str,
    iterations: int,
    function_name: str,
    batch_size: int,
    one_batch: bool = True,
    save_position: str = None,
    n_initial_points: Optional[int] = None,
    optimization_kwargs: Optional[dict] = None,
    dim: Optional[int] = None,
    acqf_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
    problem_kwargs: Optional[Dict[str, np.ndarray]] = None,
    X_init: Optional[Tensor] = None,
    Y_init: Optional[Tensor] = None,
) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'
    tkwargs = {"dtype": dtype, "device": device}
    acqf_kwargs = acqf_kwargs or {}
    model_kwargs = model_kwargs or {}
    problem_kwargs = problem_kwargs or {}
    optimization_kwargs = optimization_kwargs or {}

    base_function = get_problem(name=function_name, dim=dim, **problem_kwargs)
    base_function.to(**tkwargs)

    binary_dims = base_function.integer_indices
    binary_mask = base_function.integer_bounds[1] - base_function.integer_bounds[0] == 1
    if binary_mask.any():
        binary_dims = base_function.integer_indices.clone().detach().to(dtype=torch.int32)[binary_mask].cpu().tolist()
    else:
        binary_dims = []

    init_exact_rounding_func = get_exact_rounding_func(
        bounds=base_function.one_hot_bounds,
        integer_indices=base_function.integer_indices.tolist(),
        categorical_features=base_function.categorical_features,
        initialization=True,
    )

    standard_bounds = torch.ones(2, base_function.effective_dim, **tkwargs)
    standard_bounds[0] = 0
    if n_initial_points is None:
        n_initial_points = min(20, 2 * base_function.effective_dim)
    if X_init is not None:
        assert Y_init is not None
        assert X_init.shape[-1] == base_function.effective_dim
        X = X_init.to(**tkwargs)
        Y = Y_init.to(**tkwargs)
    else:
        X, Y = generate_initial_data(
            n=n_initial_points,
            base_function=base_function,
            bounds=standard_bounds,
            tkwargs=tkwargs,  
            init_exact_rounding_func=init_exact_rounding_func,
        ) 

    standardize_tf = Standardize(m=Y.shape[-1])
    stdized_Y, _ = standardize_tf(Y)
    standardize_tf.eval()
    best_obj = Y.max().view(-1).cpu()
    existing_iterations = 0
    batch_count = 0
    start_time = time()
    print(f"time: {time()-start_time}, current best obj: {best_obj}.")

    for i in range(existing_iterations, iterations):
        mll, model = initialize_model(
            train_x=X,
            train_y=stdized_Y,
            binary_dims=binary_dims,
            categorical_features=base_function.categorical_features,
            **model_kwargs,
        )
        fit_gpytorch_model(mll)

        if label == "sobol":
            raw_candidates = draw_sobol_samples(bounds=standard_bounds,n=1,q=1).squeeze(0).to(**tkwargs)
            candidates = init_exact_rounding_func(raw_candidates)
        else:
            acqf_kwargs["apply_numeric"] = True
            acqf_kwargs.setdefault("pr_mc_samples", 128)
            acqf_kwargs.setdefault("pr_grad_estimator", "reinforce_ma")
            acqf_kwargs.setdefault("pr_resample", True)
            optimization_kwargs.setdefault("stochastic", True)
            optimization_kwargs.setdefault("num_restarts", 20)
            optimization_kwargs.setdefault("raw_samples", 1024)
            options = optimization_kwargs.get("options")
            if options is None:
                options = {}
            optimization_kwargs["options"] = options
            options.setdefault("batch_limit", 5)
            options.setdefault("init_batch_limit", 32)
            options.setdefault("maxiter", 200)
            options.setdefault("seed", seed)

            acq_func = get_acqf(
                label=label,
                model=model,
                X_baseline=X,
                iteration=i+1,
                tkwargs=tkwargs,
                base_function=base_function,
                train_Y=stdized_Y,
                **acqf_kwargs,
            )
            options["nonnegative"] = is_nonnegative(acq_func.acq_function)
            bounds = standard_bounds
            torch.cuda.empty_cache()

            if isinstance(acq_func, AbstractProbabilisticReparameterization):
                true_acq_func = acq_func.acq_function 

            raw_candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                **optimization_kwargs,
                return_best_only=False,
            )

            with torch.no_grad():
                candidates = acq_func.sample_candidates(raw_candidates)
            candidates_numeric = acq_func.one_hot_to_numeric(candidates)
            with torch.no_grad():
                max_af = true_acq_func(candidates_numeric).max(dim=0)
                best_idx = max_af.indices.item()
            if candidates.ndim > 2:
                candidates = candidates[best_idx]

            torch.cuda.empty_cache()
            del acq_func, mll, model
            gc.collect()


        if one_batch:
            if batch_count < batch_size:
                new_y = eval_problem(X=candidates, old_x=X, old_y=Y, base_function=base_function, fantasy_point=True)
                batch_count += 1
                new_x = candidates if batch_count == 1 else torch.cat([new_x,candidates], dim=0)
        else:
            assert batch_size == 1
            new_y = eval_problem(X=candidates, old_x = None, old_y = None, base_function=base_function)


        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        standardize_tf.train()
        stdized_Y, _ = standardize_tf(Y)
        standardize_tf.eval()
        best_obj = Y.max().view(-1).cpu()
        print(f"iteration {i}, time: {time()-start_time}, current best obj: {best_obj}.")
        print(X,Y)

        if (one_batch and (batch_count == batch_size)):
            print(f"Finish with {batch_size} points")
            df = pd.DataFrame(new_x.cpu().numpy())
            feedback(df, save_position)
            break
