# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone action-conditioned residual SAC components."""

from .gr00t_adapter import GR00TDecoratorProposal, GR00TPolicyDecoratorAdapter
from .networks import ResidualActor, TwinQCritic
from .replay_buffer import DecoratorReplayBuffer, DecoratorTransitionBatch
from .sac import PolicyDecoratorSAC, PolicyDecoratorSACConfig
from .schedule import ProgressiveResidualSchedule

__all__ = [
    "DecoratorReplayBuffer",
    "DecoratorTransitionBatch",
    "GR00TDecoratorProposal",
    "GR00TPolicyDecoratorAdapter",
    "PolicyDecoratorSAC",
    "PolicyDecoratorSACConfig",
    "ProgressiveResidualSchedule",
    "ResidualActor",
    "TwinQCritic",
]
