# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Openenv Jayesh Environment Implementation.
A Simple Task Manager Environment.
"""

from typing import Optional, Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TaskManagerAction, TaskManagerObservation
except (ModuleNotFoundError, ImportError):
    from models import TaskManagerAction, TaskManagerObservation


class OpenenvJayeshEnvironment(Environment):
    """A Task Manager environment with 3 difficulty modes."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._scenarios = ["Easy", "Medium", "Hard"]
        
        # Internal state
        self.tasks = []
        self.difficulty = "Easy"
        self.goal_completed = False
        self.tasks_added = 0
        self.high_priority_added = 0
        self.high_priority_completed = 0
        self.list_called = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:
        """Reset the environment."""
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        
        idx = self._reset_count % len(self._scenarios)
        self.difficulty = self._scenarios[idx]
        self._reset_count += 1
        
        self.tasks = []
        self.goal_completed = False
        self.tasks_added = 0
        self.high_priority_added = 0
        self.high_priority_completed = 0
        self.list_called = False
        
        message = f"Task Manager started in {self.difficulty} mode. "
        if self.difficulty == "Easy":
            message += "Goal: Add 2 tasks and list them."
        elif self.difficulty == "Medium":
            message += "Goal: Add 3 tasks (mixed priorities) and complete all High priority ones."
        elif self.difficulty == "Hard":
            message += "Goal: Add 4 tasks (at least 2 High) and complete at least 2 High priority tasks."
            
        return TaskManagerObservation(
            success=True,
            message=message,
            tasks=self.tasks,
            done=False,
            reward=0.0
        )

    def step(
        self,
        action: TaskManagerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:  # type: ignore[override]
        """Execute a step."""
        self._state.step_count += 1
        
        cmd = action.command.lower()
        success = True
        message = ""
        
        if cmd == "add":
            if not action.title:
                success = False
                message = "Title is required to add a task."
            else:
                prio = action.priority or "Normal"
                new_task = {
                    "title": action.title,
                    "priority": prio,
                    "deadline": action.deadline or "N/A",
                    "completed": False
                }
                self.tasks.append(new_task)
                self.tasks_added += 1
                if prio == "High":
                    self.high_priority_added += 1
                message = f"Task '{action.title}' added successfully. (Priority: {prio})"
        elif cmd == "complete":
            if not action.title:
                success = False
                message = "Title is required to complete a task."
            else:
                found = False
                for t in self.tasks:
                    if t["title"] == action.title:
                        if not t["completed"]:
                            t["completed"] = True
                            if t["priority"] == "High":
                                self.high_priority_completed += 1
                        found = True
                        message = f"Task '{action.title}' marked as completed."
                        break
                if not found:
                    success = False
                    message = f"Task '{action.title}' not found."
        elif cmd == "list":
            self.list_called = True
            message = f"Listed {len(self.tasks)} current tasks."
        else:
            success = False
            message = f"Unknown command: '{cmd}'"
            
        reward = self._calculate_reward()
        done = self.goal_completed
        
        return TaskManagerObservation(
            success=success,
            message=message,
            tasks=self.tasks,
            done=done,
            reward=reward,
            metadata={"difficulty": self.difficulty, "step": self._state.step_count}
        )

    def _calculate_reward(self) -> float:
        reward = 0.0
        self.goal_completed = False
        
        if self.difficulty == "Easy":
            # Add 2 tasks (0.4 each) + list them (0.2)
            reward += min(0.8, self.tasks_added * 0.4)
            if self.list_called:
                reward += 0.2
                
            if self.tasks_added >= 2 and self.list_called:
                reward = 1.0
                self.goal_completed = True
                
        elif self.difficulty == "Medium":
            # Add 3 tasks (0.2 each) + Add High (0.1) + Complete High (0.3 ratio)
            reward += min(0.6, self.tasks_added * 0.2)
            if self.high_priority_added >= 1:
                reward += 0.1
                completion_ratio = self.high_priority_completed / self.high_priority_added
                reward += min(0.3, completion_ratio * 0.3)
                    
            if self.tasks_added >= 3 and self.high_priority_added >= 1:
                if self.high_priority_completed >= self.high_priority_added:
                    reward = 1.0
                    self.goal_completed = True
                    
        elif self.difficulty == "Hard":
            # Add 4 tasks (0.1 each) + Add 2 High (0.1 each) + Complete 2 High (0.2 each)
            reward += min(0.4, self.tasks_added * 0.1)
            reward += min(0.2, self.high_priority_added * 0.1)
            reward += min(0.4, self.high_priority_completed * 0.2)
            
            if self.tasks_added >= 4 and self.high_priority_added >= 2 and self.high_priority_completed >= 2:
                reward = 1.0
                self.goal_completed = True
        
        return round(min(1.0, reward), 2)

    @property
    def state(self) -> State:
        return self._state
