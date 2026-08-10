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

MODULE_PATH = (
    Path(__file__).parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "gr00t"
    / "residual_policy.py"
)
SPEC = importlib.util.spec_from_file_location("gr00t_residual_policy", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
GaussianResidualActor = MODULE.GaussianResidualActor
summarize_residual_actions = MODULE.summarize_residual_actions
summarize_residual_log_std = MODULE.summarize_residual_log_std


def make_actor() -> GaussianResidualActor:
    return GaussianResidualActor(
        vlm_width=8,
        state_width=6,
        hidden_width=16,
        hidden_layers=2,
        action_horizon=4,
        action_dim=3,
        residual_bound=0.1,
        initial_log_std=-2.5,
    )


def test_zero_initialized_residual_is_exactly_zero_in_eval() -> None:
    actor = make_actor()
    vlm = torch.randn(2, 5, 8)
    state = torch.randn(2, 1, 6)

    output = actor.sample(vlm, state, deterministic=True)

    assert output.raw_action.shape == (2, 4, 3)
    assert torch.equal(output.mean, torch.zeros_like(output.mean))
    assert torch.equal(output.action, torch.zeros_like(output.action))
    assert torch.equal(output.mean_action, torch.zeros_like(output.mean_action))
    assert torch.equal(
        output.exploration_action,
        torch.zeros_like(output.exploration_action),
    )
    assert torch.allclose(output.log_std, torch.full_like(output.log_std, -2.5))


def test_residual_sample_and_recomputed_logprob_match() -> None:
    torch.manual_seed(7)
    actor = make_actor()
    vlm = torch.randn(2, 5, 8)
    state = torch.randn(2, 1, 6)

    output = actor.sample(vlm, state, deterministic=False)
    recomputed, entropy = actor.evaluate_actions(vlm, state, output.raw_action)

    assert output.action.abs().max() <= 0.1
    assert torch.allclose(recomputed, output.logprobs)
    assert torch.allclose(entropy, output.entropy)


def test_force_zero_does_not_consume_sampling_rng() -> None:
    actor = make_actor()
    vlm = torch.randn(2, 5, 8)
    state = torch.randn(2, 1, 6)
    state_before = torch.random.get_rng_state()

    output = actor.sample(vlm, state, deterministic=False, force_zero=True)

    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert torch.equal(output.raw_action, torch.zeros_like(output.raw_action))
    assert torch.equal(output.action, torch.zeros_like(output.action))


def test_residual_summary_has_dimension_and_horizon_metrics() -> None:
    actions = torch.zeros(2, 3, 4, 3)
    actions[..., 0] = 0.02

    metrics = summarize_residual_actions(
        actions,
        dimension_names=("x", "y", "z"),
    )

    assert metrics["residual/dimension/x_abs_mean"] == pytest.approx(0.02)
    assert metrics["residual/dimension/y_abs_mean"] == 0.0
    assert metrics["residual/active_fraction_gt_0.01"] == 1.0
    assert "residual/horizon/03_l2_mean" in metrics


def test_residual_summary_separates_mean_noise_and_saturation() -> None:
    actions = torch.full((2, 4, 3), 0.095)
    summary = summarize_residual_actions(
        actions,
        dimension_names=("x", "y", "z"),
        metric_prefix="residual/mean",
        saturation_threshold=0.09,
    )
    log_std = torch.full((2, 4, 3), -2.5)
    scale_summary = summarize_residual_log_std(
        log_std,
        dimension_names=("x", "y", "z"),
    )

    assert summary["residual/mean/saturation_fraction_gt_0.09"] == 1.0
    assert summary["residual/mean/dimension/x_saturation_fraction_gt_0.09"] == 1.0
    assert scale_summary["residual/log_std_mean"] == pytest.approx(-2.5)
    assert scale_summary["residual/std_mean"] == pytest.approx(
        torch.exp(torch.tensor(-2.5)).item()
    )


def test_residual_actor_rejects_mismatched_feature_width() -> None:
    actor = make_actor()
    with pytest.raises(ValueError, match="Expected VLM width"):
        actor.sample(
            torch.randn(2, 5, 7),
            torch.randn(2, 1, 6),
            deterministic=True,
        )
