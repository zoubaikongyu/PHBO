#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod  
from collections import OrderedDict
from contextlib import ExitStack
from typing import Dict, List, Optional
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torch.nn.functional import one_hot

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
)
from botorch.models.model import Model

from PHBO.input import (
    MCProbabilisticReparameterizationInputTransform,
    OneHotToNumeric,
)

def get_probabilistic_reparameterization_input_transform(
    dim: int,
    integer_indices: List[int],
    integer_bounds: Tensor,
    categorical_features: Optional[Dict[int, int]] = None,
    mc_samples: int = 1024,
    resample: bool = False,
    flip: bool = False,
    tau: float = 0.1,
) -> ChainedInputTransform:
    bounds = torch.zeros(
        2, dim, dtype=integer_bounds.dtype, device=integer_bounds.device
    )
    bounds[1] = 1
    bounds[:, integer_indices] = integer_bounds
    tfs = OrderedDict()
    if integer_indices is not None and len(integer_indices) > 0:
        tfs["unnormalize"] = Normalize(
            d=dim,
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=True,
        )
    tfs["round"] = MCProbabilisticReparameterizationInputTransform(
        integer_indices=integer_indices,
        integer_bounds=integer_bounds,
        categorical_features=categorical_features,
        resample=resample,
        mc_samples=mc_samples,
        flip=flip,
        tau=tau,
    )
    if integer_indices is not None and len(integer_indices) > 0:
        tfs["normalize"] = Normalize(
            d=dim,
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=False,
        )
    tf = ChainedInputTransform(**tfs)
    tf.eval()
    return tf


class AcquisitionFunctionWrapper(AcquisitionFunction):
    def __init__(self, acq_function: AcquisitionFunction) -> None:
        Module.__init__(self)
        self.acq_function = acq_function

    @property
    def X_baseline(self) -> Optional[Tensor]:
        return self.acq_function.X_baseline

    @property
    def model(self) -> Model:
        return self.acq_function.model


class AbstractProbabilisticReparameterization(AcquisitionFunctionWrapper, ABC):
    def __init__(
        self,
        acq_function: AcquisitionFunction,
        dim: int,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        batch_limit: Optional[int] = None,
        apply_numeric: bool = False,
        **kwargs
    ) -> None:
        if categorical_features is None and (
            integer_indices is None or integer_bounds is None
        ):
            raise NotImplementedError(
                "categorical_features or integer indices and integer_bounds must be provided."
            )
        super().__init__(acq_function=acq_function)
        self.batch_limit = batch_limit

        if apply_numeric:
            self.one_hot_to_numeric = OneHotToNumeric(
                categorical_features=categorical_features,
                transform_on_train=False,
                transform_on_eval=True,
                transform_on_fantasize=False,
            )
            self.one_hot_to_numeric.eval()
        else:
            self.one_hot_to_numeric = None
        discrete_indices = []
        if integer_indices is not None:
            self.register_buffer(
                "integer_indices",
                torch.tensor(
                    integer_indices, dtype=torch.long, device=integer_bounds.device
                ),
            )
            self.register_buffer("integer_bounds", integer_bounds)
            discrete_indices.extend(integer_indices)
        else:
            self.register_buffer(
                "integer_indices",
                torch.tensor([], dtype=torch.long, device=integer_bounds.device),
            )
            self.register_buffer(
                "integer_bounds",
                torch.tensor(
                    [], dtype=integer_bounds.dtype, device=integer_bounds.device
                ),
            )
        if categorical_features is not None and len(categorical_features) > 0:
            categorical_indices = list(range(min(categorical_features.keys()), dim))
            discrete_indices.extend(categorical_indices)
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    categorical_indices,
                    dtype=torch.long,
                    device=integer_bounds.device,
                ),
            )
            self.categorical_features = categorical_features
        else:
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    [],
                    dtype=torch.long,
                    device=integer_bounds.device,
                ),
            )

        self.register_buffer(
            "cont_indices",
            torch.tensor(
                sorted(list(set(range(dim)) - set(discrete_indices))),
                dtype=torch.long,
                device=integer_bounds.device,
            ),
        )
        self.model = acq_function.model 
        self.register_buffer(
            "ma_counter",
            torch.zeros(1, dtype=torch.double, device=integer_bounds.device),
        )
        self.register_buffer(
            "ma_hidden",
            torch.zeros(1, dtype=torch.double, device=integer_bounds.device),
        )
        self.register_buffer(
            "ma_baseline",
            torch.zeros(1, dtype=torch.double, device=integer_bounds.device),
        )

    def sample_candidates(self, X: Tensor) -> Tensor:
        if "unnormalize" in self.input_transform:
            unnormalized_X = self.input_transform["unnormalize"](X)
        else:
            unnormalized_X = X.clone()
        prob = self.input_transform["round"].get_rounding_prob(X=unnormalized_X)
        discrete_idx = 0
        for i in self.integer_indices:
            p = prob[..., discrete_idx]
            rounding_component = torch.distributions.Bernoulli(probs=p).sample()
            unnormalized_X[..., i] = unnormalized_X[..., i].floor() + rounding_component
            discrete_idx += 1
        unnormalized_X[..., self.integer_indices] = torch.minimum(
            torch.maximum(
                unnormalized_X[..., self.integer_indices], self.integer_bounds[0]
            ),
            self.integer_bounds[1],
        )
        raw_idx = self.cont_indices.shape[0] + discrete_idx
        if self.categorical_indices.shape[0] > 0:
            for i, cardinality in self.categorical_features.items():
                discrete_end = discrete_idx + cardinality
                p = prob[..., discrete_idx:discrete_end]
                z = one_hot(
                    torch.distributions.Categorical(probs=p).sample(),
                    num_classes=cardinality,
                )
                raw_end = raw_idx + cardinality
                unnormalized_X[..., raw_idx:raw_end] = z
                discrete_idx = discrete_end
                raw_idx = raw_end
        if "normalize" in self.input_transform:
            return self.input_transform["normalize"](unnormalized_X)
        return unnormalized_X

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Compute PR."""
        pass


class MCProbabilisticReparameterization(AbstractProbabilisticReparameterization):
    def __init__(
        self,
        acq_function: AcquisitionFunction,
        dim: int,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        batch_limit: Optional[int] = None,
        apply_numeric: bool = False,
        mc_samples: int = 1024,
        grad_estimator: str = "reinforce",
        tau: float = 0.1,
    ) -> None:
        super().__init__(
            acq_function=acq_function,
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            batch_limit=batch_limit,
            apply_numeric=apply_numeric,
        )
        if self.batch_limit is None:
            self.batch_limit = mc_samples
        self.grad_estimator = grad_estimator 
        self._pr_acq_function = _MCProbabilisticReparameterization()
        self.input_transform = get_probabilistic_reparameterization_input_transform(
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            mc_samples=mc_samples,
            tau=tau,
        )
        self.input_transform_flip = None

    def forward(self, X: Tensor) -> Tensor:
        return self._pr_acq_function.apply(
            X,
            self.acq_function,
            self.input_transform,
            self.input_transform_flip,
            self.batch_limit,
            self.integer_indices,
            self.cont_indices,
            self.categorical_indices,
            self.grad_estimator,
            self.one_hot_to_numeric,
            self.ma_counter,
            self.ma_hidden,
        )


class _MCProbabilisticReparameterization(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        acq_function: AcquisitionFunction,
        input_tf: InputTransform,
        input_tf_flip: Optional[InputTransform],
        batch_limit: Optional[int],
        integer_indices: Tensor,
        cont_indices: Tensor,
        categorical_indices: Tensor,
        grad_estimator: str,
        one_hot_to_numeric: Optional[OneHotToNumeric],
        ma_counter: Optional[Tensor],
        ma_hidden: Optional[Tensor],
    ):
        with ExitStack() as es:
            if ctx.needs_input_grad[0]:
                es.enter_context(torch.enable_grad())
            if cont_indices.shape[0] > 0:
                ctx.cont_input = input[..., cont_indices].detach().requires_grad_(True)
                cont_idx = 0
                cols = []
                for col in range(input.shape[-1]):
                    if (
                        cont_idx < cont_indices.shape[0]
                        and col == cont_indices[cont_idx]
                    ):
                        cols.append(ctx.cont_input[..., cont_idx])
                        cont_idx += 1
                    else:
                        cols.append(input[..., col])
                input = torch.stack(cols, dim=-1)
            else:
                ctx.cont_input = None
            ctx.input = input
            ctx.integer_indices = integer_indices
            ctx.discrete_indices = input_tf["round"].discrete_indices
            ctx.cont_indices = cont_indices
            ctx.categorical_indices = categorical_indices
            ctx.ma_counter = ma_counter
            ctx.ma_hidden = ma_hidden
            tilde_x_samples = input_tf(input.unsqueeze(-3))
            rounding_component = tilde_x_samples.clone()
            if integer_indices.shape[0] > 0:
                input_integer_params = input[..., integer_indices].unsqueeze(-3)
                rounding_component[..., integer_indices] = (
                    (tilde_x_samples[..., integer_indices] - input_integer_params > 0)
                    | (input_integer_params == 1)
                ).to(tilde_x_samples)
            if categorical_indices.shape[0] > 0:
                rounding_component[..., categorical_indices] = tilde_x_samples[
                    ..., categorical_indices
                ]
            ctx.rounding_component = rounding_component[..., ctx.discrete_indices]
            ctx.tau = input_tf["round"].tau
            if hasattr(input_tf["round"], "base_samples"):
                ctx.base_samples = input_tf["round"].base_samples.detach()
            if "unnormalize" in input_tf:
                unnormalized_input = input_tf["unnormalize"](input)
            else:
                unnormalized_input = input
            ctx.prob = input_tf["round"].get_rounding_prob(unnormalized_input)

            if categorical_indices.shape[0] > 0:
                ctx.base_samples_categorical = input_tf[
                    "round"
                ].base_samples_categorical.clone()
            ctx.tilde_x_samples = tilde_x_samples
            ctx.grad_estimator = grad_estimator
            acq_values_list = []
            start_idx = 0
            if one_hot_to_numeric is not None:
                tilde_x_samples = one_hot_to_numeric(tilde_x_samples)

            while start_idx < tilde_x_samples.shape[-3]:
                end_idx = min(start_idx + batch_limit, tilde_x_samples.shape[-3])
                acq_values = acq_function(tilde_x_samples[..., start_idx:end_idx, :, :])
                acq_values_list.append(acq_values)
                start_idx += batch_limit
            acq_values = torch.cat(acq_values_list, dim=-1)
            ctx.mean_acq_values = acq_values.mean(dim=-1) 
            ctx.acq_values = acq_values
            ctx.ma_hidden = ma_hidden.clone()
            ctx.ma_counter = ctx.ma_counter.clone()
            decay = 0.7
            ma_counter.add_(1)
            ma_hidden.sub_((ma_hidden - acq_values.detach().mean()) * (1 - decay))
            if ctx.needs_input_grad[0]:
                if grad_estimator in ("arm", "u2g"):
                    if input_tf["round"].categorical_starts.shape[0] > 0:
                        raise NotImplementedError
                    if (
                        not hasattr(input_tf_flip, "base_samples")
                        or input_tf_flip.base_samples.shape[-2] != input.shape[-2]
                    ):
                        input_tf_flip["round"].base_samples = (
                            input_tf["round"].base_samples.detach().clone()
                        )

                    ctx.base_samples = input_tf_flip["round"].base_samples.detach()
                    with torch.no_grad():
                        tilde_x_samples_flip = input_tf_flip(input.unsqueeze(-3))
                        ctx.rounding_component_flip = (
                            tilde_x_samples_flip[..., integer_indices]
                            - input[..., integer_indices].unsqueeze(-3)
                            > 0
                        ).to(tilde_x_samples_flip)
                        if one_hot_to_numeric is not None:
                            tilde_x_samples_flip = one_hot_to_numeric(
                                tilde_x_samples_flip
                            )
                        acq_values_flip_list = []
                        start_idx = 0
                        while start_idx < tilde_x_samples_flip.shape[-3]:
                            end_idx = min(
                                start_idx + batch_limit, tilde_x_samples_flip.shape[-3]
                            )
                            acq_values_flip = acq_function(
                                tilde_x_samples_flip[..., start_idx:end_idx, :, :]
                            )
                            acq_values_flip_list.append(acq_values_flip)
                            start_idx += batch_limit
                        acq_values_flip = torch.cat(acq_values_flip_list, dim=-1)
                        ctx.mean_acq_values_flip = acq_values_flip.mean(dim=-1) 
                        ctx.acq_values_flip = acq_values_flip
            return ctx.mean_acq_values.detach()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            acq_values = ctx.acq_values
            mean_acq_values = ctx.mean_acq_values

            tilde_x_samples = ctx.tilde_x_samples
            integer_indices = ctx.integer_indices
            cont_indices = ctx.cont_indices
            discrete_indices = ctx.discrete_indices
            rounding_component = ctx.rounding_component
            input = ctx.input
            expanded_acq_values = acq_values.view(*acq_values.shape, 1, 1).expand(
                acq_values.shape + rounding_component.shape[-2:]
            )
            prob = ctx.prob.unsqueeze(-3)
            if ctx.grad_estimator in ("arm", "u2g"):
                rounding_component_flip = ctx.rounding_component_flip
                acq_values_flip = ctx.acq_values_flip
                mean_acq_values_flip = ctx.mean_acq_values_flip
                expanded_acq_values_flip = acq_values_flip.view(
                    *acq_values_flip.shape, 1, 1
                ).expand(acq_values_flip.shape + rounding_component_flip.shape[-2:])
                if ctx.grad_estimator == "arm":
                    sample_level = (
                        (expanded_acq_values_flip - expanded_acq_values)
                        * (ctx.base_samples - 0.5)
                        * torch.abs(rounding_component_flip - rounding_component)
                    )
                elif ctx.grad_estimator == "u2g":
                    prob_abs = prob.clone()
                    prob_abs[prob_abs < 0.5] = 1 - prob_abs[prob_abs < 0.5]
                    sample_level = (
                        0.5
                        * (expanded_acq_values_flip - expanded_acq_values)
                        * prob_abs
                        * (rounding_component_flip - rounding_component)
                    )
            elif ctx.grad_estimator == "reinforce":
                sample_level = expanded_acq_values * (rounding_component - prob)
            elif ctx.grad_estimator == "reinforce_ma":
                if ctx.ma_counter == 0:
                    baseline = 0.0
                else:
                    decay = 0.7
                    baseline = ctx.ma_hidden / (1.0 - torch.pow(decay, ctx.ma_counter))
                sample_level = (expanded_acq_values - baseline) * (
                    rounding_component - prob
                )

            grads = (sample_level / ctx.tau).mean(dim=-3)

            new_grads = (
                grad_output.view(
                    *grad_output.shape,
                    *[1 for _ in range(grads.ndim - grad_output.ndim)]
                )
                .expand(*grad_output.shape, *input.shape[-2:])
                .clone()
            )
            new_grads[..., discrete_indices] *= grads
            if ctx.cont_input is not None:
                auto_grad = torch.autograd.grad(
                    mean_acq_values,
                    ctx.cont_input,
                    grad_outputs=grad_output,
                )[0]
                new_grads[..., cont_indices] = auto_grad
            return (
                new_grads,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return None, None, None, None, None, None, None, None, None, None, None, None
