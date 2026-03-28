# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Smart Personal Task Manager - OpenEnv Environment Implementation.

Three difficulty tiers:

Easy (reset 0, 3, 6 ...)
  Goal: Add 2-3 tasks of any priority, then call 'list'.
  Reward: +0.15 per task added (up to 3), +0.20 for listing. Full 1.0 on completion.

Medium (reset 1, 4, 7 ...)
  Goal: Add 4 tasks with mixed priorities AND deadlines.
        Complete ALL High-priority tasks before their deadlines.
  Reward: +0.15 per task added (up to 4), +0.10 per correct priority label,
          +0.20 per High-priority task completed on time.
          -0.25 penalty per deadline miss.
          Full 1.0 on completion.

Hard (reset 2, 5, 8 ...)
  Goal: Add 5 tasks with priorities, deadlines, AND dependencies.
        Complete tasks in valid dependency order; respect all deadlines.
  Reward: +0.15 per task added (up to 5), +0.10 per correct priority,
          +0.25 per task completed without violation.
          Bonus +0.10 for achieving perfect (optimal) topological ordering.
          -0.30 per dependency violation, -0.25 per deadline miss.
          Full 1.0 on perfect completion.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TaskManagerAction, TaskManagerObservation
except (ModuleNotFoundError, ImportError):
    from models import TaskManagerAction, TaskManagerObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {"Low", "Normal", "High"}
PRIORITY_RANK = {"Low": 0, "Normal": 1, "High": 2}


def _parse_date(dt_str: Optional[str]) -> Optional[date]:
    """Parse an ISO-8601 date string; return None on failure."""
    if not dt_str:
        return None
    try:
        return date.fromisoformat(dt_str)
    except ValueError:
        return None


def _today() -> date:
    return date.today()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OpenenvJayeshEnvironment(Environment):
    """
    Smart Personal Task Manager with three distinct difficulty levels.

    Easy   -> add tasks + list
    Medium -> deadlines + priority management
    Hard   -> deadlines + priority + dependency ordering
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._scenarios = ["Easy", "Medium", "Hard"]
        self._init_episode_state()

    def _init_episode_state(self) -> None:
        """Zero all mutable per-episode state."""
        self.tasks: List[Dict[str, Any]] = []       # ordered list of task dicts
        self.difficulty: str = "Easy"
        self.goal_completed: bool = False

        # Counters
        self.tasks_added: int = 0
        self.tasks_completed: int = 0
        self.high_tasks_added: int = 0
        self.high_tasks_completed_on_time: int = 0
        self.deadline_misses: int = 0
        self.dependency_violations: int = 0
        self.completion_order: List[str] = []       # titles in completion order

        # Episode-level flag
        self.list_called: bool = False

        # Violations log (for observation)
        self.violations: List[str] = []

        # Target counts per difficulty
        self.target_tasks: int = 2
        self.target_high: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._init_episode_state()

        idx = self._reset_count % len(self._scenarios)
        self.difficulty = self._scenarios[idx]
        self._reset_count += 1

        if self.difficulty == "Easy":
            self.target_tasks = 2
            self.target_high = 0
            msg = (
                "=== EASY MODE ===\n"
                "Goal: Add 2-3 tasks of any priority, then call 'list' to review them.\n"
                "Rewards: +0.15 per task added (max 3), +0.20 for calling list.\n"
                "Complete the goal to receive full reward (1.0)."
            )
        elif self.difficulty == "Medium":
            self.target_tasks = 4
            self.target_high = 2
            msg = (
                "=== MEDIUM MODE ===\n"
                "Goal: Add 4 tasks with priorities AND deadlines.\n"
                "      Complete ALL High-priority tasks before their deadlines.\n"
                "Tips: Include at least 2 High-priority tasks.\n"
                "      Use deadline='YYYY-MM-DD' (today or future date for on-time credit).\n"
                "Rewards: +0.15/task added, +0.10/correct priority, +0.20/High completed on time.\n"
                "Penalty: -0.25 per deadline miss."
            )
        else:  # Hard
            self.target_tasks = 5
            self.target_high = 2
            msg = (
                "=== HARD MODE ===\n"
                "Goal: Add 5 tasks with priorities, deadlines, AND dependencies.\n"
                "      Complete tasks in valid dependency order (dependencies first).\n"
                "      Respect all deadlines.\n"
                "Tips: Use depends_on=['Task Title'] when adding a dependent task.\n"
                "      Complete prerequisite tasks before their dependents.\n"
                "Rewards: +0.15/task, +0.10/priority, +0.25/completion without violation.\n"
                "         +0.10 bonus for perfect topological ordering.\n"
                "Penalty: -0.30/dependency violation, -0.25/deadline miss."
            )

        return TaskManagerObservation(
            success=True,
            message=msg,
            tasks=[],
            violations=[],
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: TaskManagerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:
        self._state.step_count += 1
        cmd = (action.command or "").strip().lower()
        success = True
        message = ""

        if cmd == "add":
            success, message = self._handle_add(action)
        elif cmd == "complete":
            success, message = self._handle_complete(action)
        elif cmd == "list":
            self.list_called = True
            message = self._format_task_list()
        else:
            success = False
            message = f"Unknown command '{cmd}'. Valid commands: 'add', 'complete', 'list'."

        reward = self._calculate_reward()

        return TaskManagerObservation(
            success=success,
            message=message,
            tasks=list(self.tasks),
            violations=list(self.violations),
            done=self.goal_completed,
            reward=reward,
            metadata={
                "difficulty": self.difficulty,
                "step": self._state.step_count,
                "tasks_added": self.tasks_added,
                "tasks_completed": self.tasks_completed,
                "deadline_misses": self.deadline_misses,
                "dependency_violations": self.dependency_violations,
            },
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_add(self, action: TaskManagerAction):
        if not action.title or not action.title.strip():
            return False, "Error: 'title' is required for the 'add' command."

        title = action.title.strip()

        # Duplicate check
        if any(t["title"] == title for t in self.tasks):
            return False, f"Error: A task titled '{title}' already exists."

        priority = (action.priority or "Normal").strip()
        if priority not in VALID_PRIORITIES:
            priority = "Normal"

        deadline_str = action.deadline
        deadline_obj = _parse_date(deadline_str)

        depends_on: List[str] = []
        if action.depends_on:
            for dep in action.depends_on:
                dep = dep.strip()
                if dep and any(t["title"] == dep for t in self.tasks):
                    depends_on.append(dep)
                elif dep:
                    return (
                        False,
                        f"Error: Dependency '{dep}' does not exist yet. "
                        "Add prerequisite tasks first.",
                    )

        task: Dict[str, Any] = {
            "title": title,
            "priority": priority,
            "deadline": deadline_str or "none",
            "depends_on": depends_on,
            "completed": False,
            "deadline_missed": False,
            "dependency_violation": False,
            "added_step": self._state.step_count,
        }
        self.tasks.append(task)
        self.tasks_added += 1
        if priority == "High":
            self.high_tasks_added += 1

        msg = (
            f"Task added: '{title}' | Priority: {priority}"
            + (f" | Deadline: {deadline_str}" if deadline_str else "")
            + (f" | Depends on: {depends_on}" if depends_on else "")
        )
        return True, msg

    def _handle_complete(self, action: TaskManagerAction):
        if not action.title or not action.title.strip():
            return False, "Error: 'title' is required for the 'complete' command."

        title = action.title.strip()
        task = next((t for t in self.tasks if t["title"] == title), None)

        if task is None:
            return False, f"Error: Task '{title}' not found."
        if task["completed"]:
            return False, f"Task '{title}' is already completed."

        # ---- Dependency check ----
        dep_violation = False
        unmet = [
            dep for dep in task["depends_on"]
            if not any(t["title"] == dep and t["completed"] for t in self.tasks)
        ]
        if unmet:
            dep_violation = True
            task["dependency_violation"] = True
            self.dependency_violations += 1
            violation_msg = (
                f"DEPENDENCY VIOLATION: Completed '{title}' before prerequisites: {unmet}. "
                "Penalty applied."
            )
            self.violations.append(violation_msg)

        # ---- Deadline check ----
        deadline_missed = False
        dl = _parse_date(task["deadline"])
        if dl is not None and _today() > dl:
            deadline_missed = True
            task["deadline_missed"] = True
            self.deadline_misses += 1
            violation_msg = (
                f"DEADLINE MISSED: '{title}' was due {task['deadline']} "
                f"but completed on {_today().isoformat()}. Penalty applied."
            )
            self.violations.append(violation_msg)

        # ---- Mark completed ----
        task["completed"] = True
        self.tasks_completed += 1
        self.completion_order.append(title)

        if task["priority"] == "High" and not deadline_missed:
            self.high_tasks_completed_on_time += 1

        msg_parts = [f"Task '{title}' marked complete."]
        if dep_violation:
            msg_parts.append("(!) Dependency violation penalty applied.")
        if deadline_missed:
            msg_parts.append("(!) Deadline miss penalty applied.")
        if not dep_violation and not deadline_missed:
            msg_parts.append("Clean completion -- no penalties.")

        return True, " ".join(msg_parts)

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def _calculate_reward(self) -> float:
        self.goal_completed = False
        reward = 0.0

        if self.difficulty == "Easy":
            reward = self._reward_easy()
        elif self.difficulty == "Medium":
            reward = self._reward_medium()
        else:
            reward = self._reward_hard()

        return round(min(1.0, max(0.0, reward)), 3)

    def _reward_easy(self) -> float:
        r = 0.0
        # +0.15 per task added, up to 3 tasks
        r += min(3, self.tasks_added) * 0.15
        # +0.20 for calling list
        if self.list_called:
            r += 0.20
        # Goal: >=2 tasks added + list called
        if self.tasks_added >= 2 and self.list_called:
            r = 1.0
            self.goal_completed = True
        return r

    def _reward_medium(self) -> float:
        r = 0.0
        # +0.15 per task added, up to 4
        r += min(4, self.tasks_added) * 0.15
        # +0.10 per task that has an explicit priority label (not default "Normal" by omission)
        explicit_priority_tasks = sum(
            1 for t in self.tasks
            if t["priority"] != "Normal" or t.get("priority_explicit", False)
        )
        r += min(4, explicit_priority_tasks) * 0.10
        # +0.20 per High-priority task completed on time
        r += self.high_tasks_completed_on_time * 0.20
        # Penalties
        r -= self.deadline_misses * 0.25
        # Goal: >=4 tasks, >=2 High added, ALL High completed on time, no deadline misses
        high_tasks = [t for t in self.tasks if t["priority"] == "High"]
        all_high_done = all(t["completed"] and not t["deadline_missed"] for t in high_tasks)
        if (
            self.tasks_added >= 4
            and self.high_tasks_added >= 2
            and all_high_done
            and self.deadline_misses == 0
        ):
            r = 1.0
            self.goal_completed = True
        return r

    def _reward_hard(self) -> float:
        r = 0.0
        # +0.15 per task added, up to 5
        r += min(5, self.tasks_added) * 0.15
        # +0.10 per task with non-Normal or explicit priority
        explicit_priority_tasks = sum(
            1 for t in self.tasks if t["priority"] != "Normal"
        )
        r += min(5, explicit_priority_tasks) * 0.10
        # +0.25 per task completed without any violation
        clean_completions = sum(
            1 for t in self.tasks
            if t["completed"] and not t["deadline_missed"] and not t["dependency_violation"]
        )
        r += clean_completions * 0.25
        # Penalty
        r -= self.dependency_violations * 0.30
        r -= self.deadline_misses * 0.25
        # Bonus: optimal ordering (no violations at all + all done)
        all_done = all(t["completed"] for t in self.tasks)
        if all_done and self.dependency_violations == 0 and self.deadline_misses == 0:
            r += 0.10  # perfect-run bonus
        # Goal: >=5 tasks, >=2 High, all completed, zero violations
        high_tasks = [t for t in self.tasks if t["priority"] == "High"]
        if (
            self.tasks_added >= 5
            and self.high_tasks_added >= 2
            and len(self.tasks) == self.tasks_completed
            and self.dependency_violations == 0
            and self.deadline_misses == 0
        ):
            r = 1.0
            self.goal_completed = True
        return r

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_task_list(self) -> str:
        if not self.tasks:
            return "No tasks in the system."
        lines = [f"Current tasks ({len(self.tasks)} total):"]
        for i, t in enumerate(self.tasks, 1):
            status = "DONE" if t["completed"] else "PENDING"
            flags = []
            if t.get("deadline_missed"):
                flags.append("LATE")
            if t.get("dependency_violation"):
                flags.append("DEP-VIOLATION")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            dep_str = f" | Deps: {t['depends_on']}" if t["depends_on"] else ""
            dl_str = f" | Due: {t['deadline']}" if t["deadline"] != "none" else ""
            lines.append(
                f"  {i}. [{status}]{flag_str} {t['title']} "
                f"(Priority: {t['priority']}{dl_str}{dep_str})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state
