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

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "rlinf" / "envs" / "chunk_utils.py"
SPEC = importlib.util.spec_from_file_location("chunk_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
mask_after_first_done = MODULE.mask_after_first_done


def test_mask_after_first_done_keeps_terminal_step_only() -> None:
    rewards = torch.tensor([[0.0, 1.0, -1.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    terminations = torch.tensor(
        [[False, True, False, True], [False, False, False, False]]
    )
    truncations = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )

    masked_rewards, masked_terminations, masked_truncations, valid = (
        mask_after_first_done(rewards, terminations, truncations)
    )

    assert torch.equal(masked_rewards[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(masked_rewards[1], torch.tensor([1.0, 2.0, 3.0, 0.0]))
    assert torch.equal(
        masked_terminations,
        torch.tensor([[False, True, False, False], [False, False, False, False]]),
    )
    assert torch.equal(
        masked_truncations,
        torch.tensor([[False, False, False, False], [False, False, True, False]]),
    )
    assert torch.equal(
        valid,
        torch.tensor([[True, True, False, False], [True, True, True, False]]),
    )


def test_mask_after_first_done_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share"):
        mask_after_first_done(
            torch.zeros(2, 3),
            torch.zeros(2, 2, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool),
        )
