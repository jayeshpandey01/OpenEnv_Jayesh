# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Openenv Jayesh Task Manager Environment.
"""

from typing import Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class TaskManagerAction(Action):
    """Action for the Task Manager."""
    command: str = Field(..., description="The command to execute: 'add', 'complete', 'list'")
    title: Optional[str] = Field(default=None, description="Task title (for 'add' or 'complete')")
    priority: Optional[str] = Field(default="Normal", description="Task priority (for 'add')")
    deadline: Optional[str] = Field(default=None, description="Task deadline (for 'add')")

class TaskManagerObservation(Observation):
    """Observation from the Task Manager."""
    success: bool = Field(default=True, description="Whether the action succeeded")
    message: str = Field(default="", description="System message or error")
    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="List of tasks in the system")
