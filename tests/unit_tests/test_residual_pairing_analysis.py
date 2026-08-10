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

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "flow_credit"
    / "analysis"
    / "analyze_residual_pairing.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_residual_pairing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mcnemar_exact_pvalue_is_one_for_no_discordance() -> None:
    assert MODULE.mcnemar_exact_pvalue(0, 0) == 1.0


def test_mcnemar_exact_pvalue_matches_small_binomial_case() -> None:
    assert MODULE.mcnemar_exact_pvalue(4, 0) == pytest.approx(0.125)


def test_pairing_counts_rescue_and_harm() -> None:
    def row(success):
        return {
            "success": str(success),
            "reward": "0.0",
            "episode_length": "10",
            "reset_id": "0",
        }

    base = {(0, 0): row(0), (0, 1): row(1), (0, 2): row(1)}
    candidate = {(0, 0): row(1), (0, 1): row(0), (0, 2): row(1)}

    _, summary, _ = MODULE.analyze(base, candidate)

    assert summary["rescue"] == 1
    assert summary["harm"] == 1
    assert summary["preserve"] == 1
    assert summary["net_rescue"] == 0
