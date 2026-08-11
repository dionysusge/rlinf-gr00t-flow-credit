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

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
EVALUATOR = REPOSITORY_ROOT / "examples" / "embodiment" / "eval_gr00t_fixed_no_ray.py"
SCRIPT = (
    REPOSITORY_ROOT
    / "experiments"
    / "flow_credit"
    / "scripts"
    / "run_n17_residual_e0_fixed500.sh"
)


def test_e0_evaluator_has_no_ray_dependency() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert "import ray" not in source
    assert "ray.init" not in source
    assert "EmbodiedFSDPActor" not in source
    assert "MultiStepRolloutWorker" not in source


def test_e0_script_uses_non_overlapping_fixed_slices() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "declare -a OFFSETS=(0 100 200 300 400)" in source
    assert "eval_gr00t_fixed_no_ray.py" in source
    assert "eval_embodied_agent_fixed.py" not in source


def test_e0_does_not_stop_other_ray_jobs_by_default() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "${E0_CLEAN_STALE_RAY:-0}" in source
