# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Openenv Jayesh Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TaskManagerAction, TaskManagerObservation
except ImportError:
    from models import TaskManagerAction, TaskManagerObservation


class OpenenvJayeshEnv(
    EnvClient[TaskManagerAction, TaskManagerObservation, State]
):
    """Client for the Task Manager Environment."""

    def _step_payload(self, action: TaskManagerAction) -> Dict:
        """Convert TaskManagerAction to JSON payload."""
        return {
            "command": action.command,
            "title": action.title,
            "priority": action.priority,
            "deadline": action.deadline
        }

    def _parse_result(self, payload: Dict) -> StepResult[TaskManagerObservation]:
        """Parse server response."""
        obs_data = payload.get("observation", {})
        observation = TaskManagerObservation(
            success=obs_data.get("success", True),
            message=obs_data.get("message", ""),
            tasks=obs_data.get("tasks", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
