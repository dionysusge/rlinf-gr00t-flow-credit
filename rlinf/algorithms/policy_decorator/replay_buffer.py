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

"""Fixed-schema replay for chunk-level Policy Decorator transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecoratorTransitionBatch:
    """One sampled batch with no padding or trajectory masking."""

    context: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    executed_action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_context: torch.Tensor
    next_base_action: torch.Tensor

    def to(self, device: torch.device | str) -> "DecoratorTransitionBatch":
        """Move and promote a sampled batch to float32 learner tensors."""
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device=device, dtype=torch.float32)
        return DecoratorTransitionBatch(**values)


class DecoratorReplayBuffer:
    """Preallocated CPU ring buffer with strict shape and finite checks."""

    _TENSOR_NAMES = (
        "context",
        "base_action",
        "residual_action",
        "executed_action",
        "reward",
        "done",
        "next_context",
        "next_base_action",
    )

    def __init__(
        self,
        *,
        capacity: int,
        context_dim: int,
        action_shape: tuple[int, int],
        storage_dtype: torch.dtype = torch.float16,
        seed: int = 0,
    ) -> None:
        if capacity < 1 or context_dim < 1:
            raise ValueError("capacity and context_dim must be positive")
        if storage_dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("storage_dtype must be a floating-point torch dtype")
        self.capacity = int(capacity)
        self.context_dim = int(context_dim)
        self.action_shape = (int(action_shape[0]), int(action_shape[1]))
        self.storage_dtype = storage_dtype
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

        self.context = torch.empty(
            (capacity, context_dim), dtype=storage_dtype, device="cpu"
        )
        self.next_context = torch.empty_like(self.context)
        action_buffer_shape = (capacity, *self.action_shape)
        self.base_action = torch.empty(action_buffer_shape, dtype=torch.float32)
        self.residual_action = torch.empty_like(self.base_action)
        self.executed_action = torch.empty_like(self.base_action)
        self.next_base_action = torch.empty_like(self.base_action)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32)
        self.done = torch.empty((capacity, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _cpu_tensor(value: torch.Tensor | object, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value).detach().to(device="cpu")
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Non-finite replay field: {name}")
        return tensor

    def add_batch(
        self,
        *,
        context: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        executed_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_context: torch.Tensor,
        next_base_action: torch.Tensor,
    ) -> None:
        """Add a complete vectorized transition batch without padding."""
        fields = {
            name: self._cpu_tensor(value, name=name)
            for name, value in {
                "context": context,
                "base_action": base_action,
                "residual_action": residual_action,
                "executed_action": executed_action,
                "reward": reward,
                "done": done,
                "next_context": next_context,
                "next_base_action": next_base_action,
            }.items()
        }
        batch_size = int(fields["context"].shape[0])
        expected = {
            "context": (batch_size, self.context_dim),
            "next_context": (batch_size, self.context_dim),
            "base_action": (batch_size, *self.action_shape),
            "residual_action": (batch_size, *self.action_shape),
            "executed_action": (batch_size, *self.action_shape),
            "next_base_action": (batch_size, *self.action_shape),
            "reward": (batch_size, 1),
            "done": (batch_size, 1),
        }
        for name, shape in expected.items():
            if tuple(fields[name].shape) != shape:
                raise ValueError(
                    f"Replay field {name} must have shape {shape}, "
                    f"got {tuple(fields[name].shape)}"
                )
        if not torch.logical_or(fields["done"] == 0, fields["done"] == 1).all():
            raise ValueError("done must contain only 0 or 1")

        indices = (torch.arange(batch_size) + self.position) % self.capacity
        if batch_size > self.capacity:
            # Keep the most recent capacity entries with deterministic semantics.
            keep = torch.arange(batch_size - self.capacity, batch_size)
            fields = {name: value[keep] for name, value in fields.items()}
            batch_size = self.capacity
            indices = (torch.arange(batch_size) + self.position) % self.capacity
        for name, value in fields.items():
            target = getattr(self, name)
            target[indices] = value.to(dtype=target.dtype)
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> DecoratorTransitionBatch:
        """Uniformly sample transitions from initialized slots only."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(
                f"Replay has {self.size} transitions, cannot sample {batch_size}"
            )
        indices = torch.randint(
            self.size,
            (batch_size,),
            generator=self.generator,
            device="cpu",
        )
        return DecoratorTransitionBatch(
            **{name: getattr(self, name)[indices] for name in self._TENSOR_NAMES}
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize initialized transitions and sampler state."""
        return {
            "capacity": self.capacity,
            "context_dim": self.context_dim,
            "action_shape": self.action_shape,
            "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            "position": self.position,
            "size": self.size,
            "generator_state": self.generator.get_state(),
            "tensors": {
                name: getattr(self, name)[: self.size].clone()
                for name in self._TENSOR_NAMES
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a buffer written by :meth:`state_dict`."""
        if int(state["context_dim"]) != self.context_dim:
            raise ValueError("Replay context_dim mismatch")
        if tuple(state["action_shape"]) != self.action_shape:
            raise ValueError("Replay action_shape mismatch")
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("Checkpoint replay is larger than configured capacity")
        tensors = state["tensors"]
        if not isinstance(tensors, dict):
            raise TypeError("Invalid replay tensor state")
        for name in self._TENSOR_NAMES:
            target = getattr(self, name)
            value = torch.as_tensor(tensors[name])
            target[:size].copy_(value.to(dtype=target.dtype))
        self.size = size
        self.position = int(state["position"]) % self.capacity
        self.generator.set_state(torch.as_tensor(state["generator_state"]))
