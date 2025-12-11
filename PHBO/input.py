#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from botorch.models.transforms.input import InputTransform
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot

class OneHotToNumeric(InputTransform, Module):
    def __init__(
        self,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.categorical_starts = []
        self.categorical_ends = []
        self.categorical_features = (
            None
            if ((categorical_features is None) or (len(categorical_features) == 0))
            else categorical_features
        )

        if self.categorical_features is not None:
            start_idx = None
            for i in sorted(categorical_features.keys()):
                if start_idx is None:
                    start_idx = i
                self.categorical_starts.append(start_idx)
                end_idx = start_idx + categorical_features[i]
                self.categorical_ends.append(end_idx)
                start_idx = end_idx
            self.numeric_dim = min(self.categorical_starts) + len(categorical_features)

    def transform(self, X: Tensor) -> Tensor:
        if self.categorical_features is not None:
            X_numeric = X[..., : self.numeric_dim].clone()
            idx = self.categorical_starts[0]
            for start, end in zip(self.categorical_starts, self.categorical_ends):
                X_numeric[..., idx] = X[..., start:end].argmax(dim=-1)
                idx += 1
            return X_numeric
        return X

    def untransform(self, X: Tensor) -> Tensor:
        if X.requires_grad:
            raise NotImplementedError
        if self.categorical_features is not None:
            one_hot_categoricals = [
                one_hot(X[..., idx].long(), num_classes=cardinality)
                for idx, cardinality in sorted(
                    self.categorical_features.items(), key=lambda x: x[0]
                )
            ]
            X = torch.cat(
                [
                    X[..., : min(self.categorical_features.keys())],
                    *one_hot_categoricals,
                ],
                dim=-1,
            )
        return X

class Round(InputTransform, Module):
    def __init__(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        approximate: bool = True,
        tau: float = 1e-3,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        integer_indices = integer_indices or []
        self.register_buffer(
            "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
        )
        self.categorical_starts = []
        self.categorical_ends = []
        if categorical_features is not None:
            start_idx = None
            for i in sorted(categorical_features.keys()):
                if start_idx is None:
                    start_idx = i

                self.categorical_starts.append(start_idx)
                end_idx = start_idx + categorical_features[i]
                self.categorical_ends.append(end_idx)
                start_idx = end_idx
        self.approximate = approximate
        self.tau = tau

    def transform(self, X: Tensor) -> Tensor:
        X_rounded = X.clone()
        X_int = X_rounded[..., self.integer_indices]
        X_int = X_int.round()
        X_rounded[..., self.integer_indices] = X_int
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            cardinality = end - start
            if self.approximate:
                raise NotImplementedError
            else:
                X_rounded[..., start:end] = one_hot(
                    X[..., start:end].argmax(dim=-1), num_classes=cardinality
                )
        return X_rounded

class MCProbabilisticReparameterizationInputTransform(InputTransform, Module):
    def __init__(
        self,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        flip: bool = False,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        if integer_indices is None and categorical_features is None:
            raise ValueError(
                "integer_indices and/or categorical_features must be provided."
            )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        discrete_indices = []
        if integer_indices is not None and len(integer_indices) > 0:
            self.register_buffer(
                "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
            )
            discrete_indices += integer_indices
        else:
            self.integer_indices = None
        self.categorical_features = categorical_features
        categorical_starts = []
        categorical_ends = []
        if self.categorical_features is not None:
            start = None
            for i, n_categories in categorical_features.items():
                if start is None:
                    start = i
                end = start + n_categories
                categorical_starts.append(start)
                categorical_ends.append(end)
                discrete_indices += list(range(start, end))
                start = end
        self.register_buffer(
            "discrete_indices", torch.tensor(discrete_indices, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_starts", torch.tensor(categorical_starts, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_ends", torch.tensor(categorical_ends, dtype=torch.long)
        )
        if integer_indices is None:
            self.register_buffer("integer_bounds", torch.tensor([], dtype=torch.long))
        else:
            self.register_buffer("integer_bounds", integer_bounds)
        self.mc_samples = mc_samples
        self.resample = resample
        self.flip = flip
        self.tau = tau

    def get_rounding_prob(self, X: Tensor) -> Tensor:
        X_prob = X.detach().clone()
        if self.integer_indices is not None:
            X_int = X_prob[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            if self.tau is not None:
                X_prob[..., self.integer_indices] = torch.sigmoid(
                    (X_int_abs - offset - 0.5) / self.tau
                )
            else:
                X_prob[..., self.integer_indices] = X_int_abs - offset
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X_prob[..., start:end]
            if self.tau is not None:
                X_prob[..., start:end] = torch.softmax(
                    (X_categ - 0.5) / self.tau, dim=-1
                )
            else:
                X_prob[..., start:end] = X_categ / X_categ.sum(dim=-1)
        return X_prob[..., self.discrete_indices]

    def transform(self, X: Tensor) -> Tensor:
        X_expanded = X.expand(*X.shape[:-3], self.mc_samples, *X.shape[-2:]).clone()
        X_prob = self.get_rounding_prob(X=X)
        if self.integer_indices is not None:
            X_int = X[..., self.integer_indices].detach()
            assert X.ndim > 1
            if X.ndim == 2:
                X.unsqueeze(-1)
            if (
                not hasattr(self, "base_samples")
                or self.base_samples.shape[-2:] != X_int.shape[-2:]
                or self.resample
            ):
                bounds = torch.zeros(
                    2, X_int.shape[-1], dtype=X_int.dtype, device=X_int.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X_int.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )
            X_int_abs = X_int.abs()
            is_negative = X_int < 0
            offset = X_int_abs.floor()
            prob = X_prob[..., : self.integer_indices.shape[0]]
            if self.flip:
                rounding_component = (1 - prob < self.base_samples).to(
                    dtype=X.dtype,
                )
            else:
                rounding_component = (prob >= self.base_samples).to(
                    dtype=X.dtype,
                )
            X_abs_rounded = offset + rounding_component
            X_int_new = (-1) ** is_negative.to(offset) * X_abs_rounded
            X_expanded[..., self.integer_indices] = torch.minimum(
                torch.maximum(X_int_new, self.integer_bounds[0]), self.integer_bounds[1]
            )

        if self.categorical_features is not None and len(self.categorical_features) > 0:
            if (
                not hasattr(self, "base_samples_categorical")
                or self.base_samples_categorical.shape[-2] != X.shape[-2]
                or self.resample
            ):
                bounds = torch.zeros(
                    2, len(self.categorical_features), dtype=X.dtype, device=X.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples_categorical",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )

            sample_d_start_idx = 0
            X_categ_prob = X_prob
            if self.integer_indices is not None:
                n_ints = self.integer_indices.shape[0]
                if n_ints > 0:
                    X_categ_prob = X_prob[..., n_ints:]

            for i, (idx, cardinality) in enumerate(self.categorical_features.items()):
                sample_d_end_idx = sample_d_start_idx + cardinality
                start = self.categorical_starts[i]
                end = self.categorical_ends[i]
                cum_prob = X_categ_prob[
                    ..., sample_d_start_idx:sample_d_end_idx
                ].cumsum(dim=-1)
                categories = (
                    (
                        (cum_prob > self.base_samples_categorical[..., i : i + 1])
                        .long()
                        .cumsum(dim=-1)
                        == 1
                    )
                    .long()
                    .argmax(dim=-1)
                )
                X_expanded[..., start:end] = one_hot(
                    categories, num_classes=cardinality
                ).to(X)
                sample_d_start_idx = sample_d_end_idx

        return X_expanded

    def equals(self, other: InputTransform) -> bool:
        return (
            super().equals(other=other)
            and (self.resample == other.resample)
            and torch.equal(self.base_samples, other.base_samples)
            and (self.flip == other.flip)
            and torch.equal(self.integer_indices, other.integer_indices)
        )
