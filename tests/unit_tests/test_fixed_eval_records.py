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

import csv
import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[2]
    / "experiments"
    / "flow_credit"
    / "analysis"
    / "fixed_eval_records.py"
)
SPEC = importlib.util.spec_from_file_location("fixed_eval_records", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

episode_records = MODULE.episode_records
to_finite_tensor = MODULE.to_finite_tensor
write_trial_records = MODULE.write_trial_records


def _infos() -> list[dict]:
    episode = {
        "task_id": torch.tensor([0, 1, 2]),
        "trial_id": torch.tensor([4, 5, 6]),
        "reset_id": torch.tensor([40, 50, 60]),
        "success_once": torch.tensor([True, False, True]),
        "return": torch.tensor([1.0, 0.0, 1.0]),
        "reward": torch.tensor([0.1, 0.0, 0.2]),
        "episode_len": torch.tensor([10, 256, 5]),
    }
    return [
        {},
        {
            "_final_info": torch.tensor([True, False, True]),
            "final_info": {"episode": episode},
        },
    ]


def test_episode_records_obeys_unique_completion_mask() -> None:
    records = episode_records(_infos(), label="base-setA", checkpoint="sft_base")

    assert len(records) == 2
    assert [(row["task_id"], row["trial_id"]) for row in records] == [
        (0, 4),
        (2, 6),
    ]
    assert records[0]["success"] == 1
    assert records[1]["episode_length"] == 5
    assert records[0]["model"] == "base-setA"


def test_episode_records_rejects_misaligned_metrics() -> None:
    infos = _infos()
    infos[-1]["final_info"]["episode"]["reward"] = torch.tensor([0.1])

    with pytest.raises(ValueError, match="Misaligned"):
        episode_records(infos, label="x", checkpoint="sft_base")


def test_finite_evidence_check_rejects_nan() -> None:
    with pytest.raises(FloatingPointError, match="Non-finite"):
        to_finite_tensor(torch.tensor([torch.nan]))


def test_trial_writer_preserves_existing_e0_schema(tmp_path: Path) -> None:
    records = episode_records(_infos(), label="base-setA", checkpoint="sft_base")

    write_trial_records(tmp_path, records)

    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "task_id",
        "trial_id",
        "reset_id",
        "success",
        "return",
        "reward",
        "episode_length",
        "model",
        "checkpoint",
    }


def test_trial_writer_rejects_duplicate_pair(tmp_path: Path) -> None:
    records = episode_records(_infos(), label="base-setA", checkpoint="sft_base")
    records.append(dict(records[0]))

    with pytest.raises(ValueError, match="duplicate"):
        write_trial_records(tmp_path, records)
