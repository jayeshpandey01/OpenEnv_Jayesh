# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Smart Personal Task Manager OpenEnv Environment.

Supports three difficulty levels:
  Easy   – add tasks & list them
  Medium – priorities + deadlines; complete High-priority tasks before deadline
  Hard   – priorities + deadlines + dependencies; complete in valid topological order
"""

from typing import Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TaskManagerAction(Action):
    """Action for the Smart Personal Task Manager."""

    command: str = Field(
        ...,
        description=(
            "Command to execute. One of: 'add', 'complete', 'list'. "
            "'add' – create a new task. "
            "'complete' – mark an existing task as done. "
            "'list' – display all current tasks."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable task title. Required for 'add' and 'complete'.",
    )
    priority: Optional[str] = Field(
        default="Normal",
        description="Task priority. One of: 'Low', 'Normal', 'High'. Used for 'add'.",
    )
    deadline: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 date string (YYYY-MM-DD) indicating when the task "
            "must be completed. Used for 'add'. Relevant in Medium and Hard modes."
        ),
    )
    depends_on: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of task titles that must be completed BEFORE this task can be "
            "completed. Used for 'add'. Relevant in Hard mode only."
        ),
    )


class TaskManagerObservation(Observation):
    """Observation returned after each step in the Task Manager."""

    success: bool = Field(default=True, description="Whether the last action succeeded.")
    message: str = Field(default="", description="System status message or error description.")
    tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Snapshot of all tasks. Each entry contains: "
            "title, priority, deadline, depends_on, completed, "
            "deadline_missed (bool), dependency_violation (bool)."
        ),
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of rule violations encountered this episode (deadline misses, dependency order errors).",
    )
