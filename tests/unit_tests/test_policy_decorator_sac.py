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
import sys
from pathlib import Path

import pytest
import torch

PACKAGE_DIR = Path(__file__).parents[2] / "rlinf" / "algorithms" / "policy_decorator"
SPEC = importlib.util.spec_from_file_location(
    "policy_decorator_test_package",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DecoratorReplayBuffer = MODULE.DecoratorReplayBuffer
GR00TPolicyDecoratorAdapter = MODULE.GR00TPolicyDecoratorAdapter
PolicyDecoratorSAC = MODULE.PolicyDecoratorSAC
PolicyDecoratorSACConfig = MODULE.PolicyDecoratorSACConfig
ProgressiveResidualSchedule = MODULE.ProgressiveResidualSchedule
ResidualActor = MODULE.ResidualActor


def _make_actor() -> ResidualActor:
    return ResidualActor(
        context_dim=8,
        action_shape=(4, 3),
        hidden_dim=16,
        hidden_layers=2,
        initial_log_std=-2.5,
    )


def _transition(batch_size: int = 16) -> dict[str, torch.Tensor]:
    base_action = torch.randn(batch_size, 4, 3) * 0.2
    residual = torch.tanh(torch.randn(batch_size, 4, 3) * 0.1)
    return {
        "context": torch.randn(batch_size, 8),
        "base_action": base_action,
        "residual_action": residual,
        "executed_action": base_action + 0.1 * residual,
        "reward": torch.randn(batch_size, 1),
        "done": torch.zeros(batch_size, 1),
        "next_context": torch.randn(batch_size, 8),
        "next_base_action": torch.randn(batch_size, 4, 3) * 0.2,
    }


def test_deterministic_actor_starts_as_exact_base_policy() -> None:
    actor = _make_actor()
    output = actor.sample(
        torch.randn(2, 8),
        torch.randn(2, 4, 3),
        deterministic=True,
    )

    assert torch.equal(output.action, torch.zeros_like(output.action))
    assert torch.equal(output.mean_action, torch.zeros_like(output.mean_action))
    assert torch.allclose(output.log_std, torch.full_like(output.log_std, -2.5))
    assert torch.isfinite(output.log_prob).all()


def test_actor_is_conditioned_on_base_action() -> None:
    torch.manual_seed(7)
    actor = _make_actor()
    torch.nn.init.normal_(actor.mean_head.weight)
    context = torch.randn(2, 8)

    zero_mean, _ = actor.distribution_parameters(context, torch.zeros(2, 4, 3))
    shifted_mean, _ = actor.distribution_parameters(context, torch.ones(2, 4, 3))

    assert not torch.allclose(zero_mean, shifted_mean)


def test_replay_has_no_padding_and_rejects_nonfinite_fields() -> None:
    replay = DecoratorReplayBuffer(
        capacity=32,
        context_dim=8,
        action_shape=(4, 3),
        seed=3,
    )
    transition = _transition(batch_size=8)
    replay.add_batch(**transition)
    sample = replay.sample(8)

    assert len(replay) == 8
    assert sample.context.shape == (8, 8)
    assert sample.done.shape == (8, 1)
    assert torch.isfinite(sample.executed_action).all()

    transition["context"][0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="context"):
        replay.add_batch(**transition)


def test_replay_checkpoint_round_trip() -> None:
    replay = DecoratorReplayBuffer(
        capacity=32,
        context_dim=8,
        action_shape=(4, 3),
        seed=3,
    )
    replay.add_batch(**_transition(batch_size=8))
    restored = DecoratorReplayBuffer(
        capacity=32,
        context_dim=8,
        action_shape=(4, 3),
        seed=99,
    )

    restored.load_state_dict(replay.state_dict())

    assert len(restored) == len(replay)
    assert torch.equal(restored.context[:8], replay.context[:8])
    assert torch.equal(restored.generator.get_state(), replay.generator.get_state())


def test_progressive_exploration_schedule_has_exact_endpoints() -> None:
    schedule = ProgressiveResidualSchedule(100)

    assert schedule.ratio(0) == 0.0
    assert schedule.ratio(50) == 0.5
    assert schedule.ratio(100) == 1.0
    assert not schedule.mask(step=0, batch_size=4, device="cpu").any()
    assert schedule.mask(step=100, batch_size=4, device="cpu").all()


def test_sac_update_is_finite_and_changes_actor() -> None:
    torch.manual_seed(11)
    config = PolicyDecoratorSACConfig(
        context_dim=8,
        action_horizon=4,
        action_dim=3,
        hidden_dim=32,
        hidden_layers=2,
    )
    learner = PolicyDecoratorSAC(config, device="cpu")
    replay = DecoratorReplayBuffer(
        capacity=64,
        context_dim=8,
        action_shape=(4, 3),
    )
    replay.add_batch(**_transition(batch_size=32))
    before = learner.actor.mean_head.weight.detach().clone()

    metrics = learner.update(replay.sample(16))

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(before, learner.actor.mean_head.weight)
    assert learner.update_step == 1


def test_sac_checkpoint_round_trip() -> None:
    config = PolicyDecoratorSACConfig(
        context_dim=8,
        action_horizon=4,
        action_dim=3,
        hidden_dim=16,
        hidden_layers=1,
    )
    learner = PolicyDecoratorSAC(config, device="cpu")
    replay = DecoratorReplayBuffer(
        capacity=32,
        context_dim=8,
        action_shape=(4, 3),
    )
    replay.add_batch(**_transition(batch_size=16))
    learner.update(replay.sample(8))
    restored = PolicyDecoratorSAC(config, device="cpu")

    restored.load_state_dict(learner.state_dict())

    assert restored.update_step == learner.update_step
    assert torch.equal(restored.log_alpha, learner.log_alpha)
    for restored_parameter, parameter in zip(
        restored.actor.parameters(), learner.actor.parameters(), strict=True
    ):
        assert torch.equal(restored_parameter, parameter)


def test_composition_keeps_scale_outside_policy_distribution() -> None:
    learner = PolicyDecoratorSAC(
        PolicyDecoratorSACConfig(
            context_dim=8,
            action_horizon=4,
            action_dim=3,
            residual_scale=0.1,
            hidden_dim=16,
            hidden_layers=1,
        ),
        device="cpu",
    )
    base = torch.full((2, 4, 3), 0.3)
    residual = torch.full((2, 4, 3), 0.5)

    executed = learner.compose_action(base, residual)

    assert torch.allclose(executed, torch.full_like(executed, 0.35))
    with pytest.raises(ValueError, match="inside"):
        learner.compose_action(base, torch.full_like(residual, 1.1))


class _FakeGR00T(torch.nn.Module):
    def __init__(self, *, embedded_residual: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.residual_policy_enabled = embedded_residual

    def get_policy_decorator_proposal(self, env_obs):
        del env_obs
        return {
            "context": torch.ones(2, 8),
            "base_action": torch.zeros(2, 4, 3),
            "base_action_full": torch.zeros(2, 4, 3),
            "observations": {},
            "is_batch": True,
        }

    def decode_policy_decorator_action(self, proposal, action):
        del proposal
        return action.numpy()


def test_gr00t_adapter_freezes_model_and_copies_plain_tensors() -> None:
    model = _FakeGR00T()
    adapter = GR00TPolicyDecoratorAdapter(model, learner_device="cpu")

    proposal = adapter.propose({})
    decoded = adapter.decode(proposal, proposal.base_action)

    assert not model.weight.requires_grad
    assert proposal.context.dtype == torch.float32
    assert decoded.shape == (2, 4, 3)


def test_gr00t_adapter_rejects_embedded_residual_module() -> None:
    with pytest.raises(RuntimeError, match="embedded"):
        GR00TPolicyDecoratorAdapter(
            _FakeGR00T(embedded_residual=True), learner_device="cpu"
        )
