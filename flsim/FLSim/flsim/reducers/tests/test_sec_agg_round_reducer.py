#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from dataclasses import dataclass
from tempfile import mkstemp
from typing import List, Type
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from flsim.channels.message import Message
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEqual,
    assertFalse,
    assertIsInstance,
    assertNotEqual,
    assertTrue,
)
from flsim.reducers.base_round_reducer import (
    ReductionType,
    RoundReducer,
    RoundReducerConfig,
)
from flsim.reducers.sec_agg_round_reducer import SecAggRoundReducer, SecAggRoundReducerConfig
from flsim.utils import test_utils as utils

from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flsim.secure_aggregation.secure_aggregator import FixedPointConfig



class TestRoundReducer:
    def get_round_reducer(
        self,
        model=None,
        reduction_type=ReductionType.WEIGHTED_AVERAGE,
        fixed_point_config = FixedPointConfig(num_bytes=1, scaling_factor=10),
        reset: bool = True,
    ):
        model = model or utils.SampleNet(utils.TwoFC())
        round_reducer = SecAggRoundReducer(
            **OmegaConf.structured(SecAggRoundReducerConfig(reduction_type=reduction_type,fixedpoint=fixed_point_config)),
            global_model=model,
        )
        return round_reducer

    def test_update_reduced_module(self) -> None:
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(model)
        model.fl_get_module().fill_all(0.2)
        rr.update_reduced_module(model.fl_get_module(), 3.0)
        model.fl_get_module().fill_all(0.3)
        rr.update_reduced_module(model.fl_get_module(), 2.0)
        rr.reduce()
        mismatched = utils.model_parameters_equal_to_value(
            rr.reduced_module, (3 * 0.2 + 2 * 0.3) / 5
        )
        assertEqual(mismatched, "", mismatched)

class FLRoundReducerTest:
    @pytest.mark.parametrize(
        "config, expected_type",
        [
            (RoundReducerConfig(), RoundReducer),
        ],
    )
    def test_reducer_creation_from_config(
        self, config: Type, expected_type: Type
    ) -> None:
        ref_model = utils.SampleNet(utils.TwoFC())
        reducer = instantiate(config, global_model=ref_model)
        assertIsInstance(reducer, expected_type)
