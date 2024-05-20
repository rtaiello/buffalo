#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple

from flsim.channels.base_channel import IdentityChannel
from flsim.interfaces.model import IFLModel
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.privacy.privacy_engine import IPrivacyEngine
from flsim.privacy.privacy_engine_factory import NoiseType, PrivacyEngineFactory
from flsim.privacy.user_update_clip import UserUpdateClipper
from flsim.reducers.base_round_reducer import RoundReducer, RoundReducerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils
from torch import nn
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    SecureAggregator,
    utility_config_flatter,
)


class SecAggRoundReducer(RoundReducer):
    """
    Base Class for an aggregator which gets parameters
    from different clients and aggregates them together.
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        num_users_per_round: Optional[int] = None,
        total_number_of_users: Optional[int] = None,
        channel: Optional[IdentityChannel] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SecAggRoundReducerConfig,
            **kwargs,
        )
        super().__init__(
            global_model=global_model,
            num_users_per_round=num_users_per_round,
            total_number_of_users=total_number_of_users,
            channel=channel,
            name=name,
            **kwargs,
        )
        self._secure_aggregator = SecureAggregator(
            utility_config_flatter(
                global_model.fl_get_module(),
                self.cfg.fixedpoint,
            )
        )

    def apply_weight_to_update(self, delta: nn.Module, weight: float):
        """Add the weights (parameters) of a model delta to the buffer module.

        Args:
            delta: Module whose parameters are the deltas for updating
                `self._buffer_module`'s parameters.
            weight: Weight to apply to `delta`'s parameters.

        Modifies parameters of `delta` in-place.
        """
        weight = weight if self.is_weighted else 1.0
        FLModelParamUtils.multiply_model_by_weight(
            model=delta,
            weight=weight,
            model_to_save=delta,
        )

    def add_update(self, delta: nn.Module, weight: float):
        """Update buffer module by adding the weights of a model delta to it.

        Args:
            delta: Module that contains the model delta in its weights.
            weight: Aggregation weight to apply to this model delta.
        """
        weight = weight if self.is_weighted else 1.0
        FLModelParamUtils.add_model(delta, self.reduced_module, self.reduced_module)
        self.sum_weights += weight

    def update_reduced_module(self, delta_module: nn.Module, weight: float) -> None:
        # TODO num_samples is used as the default weight, this needs revisit
        if not self.is_weighted:
            weight = 1.0
        self.apply_weight_to_update(delta_module, weight)
        self._secure_aggregator.params_to_fixedpoint(delta_module)
        self.add_update(delta_module, weight)
        self._secure_aggregator.update_aggr_overflow_and_model(
            model=self.reduced_module
        )

    def reduce(self) -> Tuple[nn.Module | float]:
        reduced_module, sum_weights = super().reduce()
        self._secure_aggregator.params_to_float(reduced_module)
        return reduced_module, sum_weights


@dataclass
class SecAggRoundReducerConfig(RoundReducerConfig):
    _target_: str = fullclassname(SecAggRoundReducer)
    fixedpoint: Optional[FixedPointConfig] = None
