#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from PHBO.run_one_replication import run_one_replication
from data_record.data_read import data_read

supported_labels = ["sobol","pr_ei","pr_ucb",]
supported_function_name = ["enzyme"]

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    function_name = "enzyme"
    label = "pr_ei"
    seed = 1
    data_path = os.path.join(current_dir, "data_record", "Case.xlsx")
    output_path = os.path.join(current_dir, function_name, f"{seed}_{label}.xlsx")
    
    kwargs = {
        "one_batch": False,
        "iterations": 100,
        "function_name": function_name,
        "batch_size": 1,
        "model_kwargs": {"kernel_type": "mixed_categorical"},
        "problem_kwargs":{"continuous": False},
        }

    if function_name == "enzyme":
        X_init,Y_init = data_read(data_path)
        kwargs["one_batch"] = True
        kwargs["batch_size"] = 8
    else:
        X_init,Y_init = None,None

    run_one_replication(
        seed=seed,
        label=label,
        save_position=output_path,
        X_init = X_init,
        Y_init = Y_init,
        **kwargs,
    )
     