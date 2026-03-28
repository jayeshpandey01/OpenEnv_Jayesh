"""
Smart Personal Task Manager - OpenEnv Environment

Easy: three tasks + list; full 1.0 when at least 2 distinct priorities; else partial credit.
Medium: small +0.07 signals; deadline miss −0.15.
Hard: +0.018 per dep edge on add; rising clean completes; dep-satisfied micro-bonus; topo bonus;
      deadline miss −0.15, dependency violation −0.18.

Reward clamp [0.0, 1.0].
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

VALID_PRIORITIES = {"Low", "Normal", "High"}

# Fair penalties (requested ranges)
MEDIUM_DEADLINE_MISS_PENALTY = 0.15   # -0.12 … -0.18
HARD_DEADLINE_MISS_PENALTY = 0.15     # -0.12 … -0.18
HARD_DEP_VIOLATION_PENALTY = 0.18     # -0.15 … -0.20


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _today() -> date:
    return date.today()


class OpenenvJayeshEnvironment(Environment):
    """Task Manager with Easy / Medium / Hard difficulty levels."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._reset_count = 0
        self._scenarios = ["Easy", "Medium", "Hard"]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_state()

    def _init_state(self) -> None:
        self.tasks: List[Dict[str, Any]] = []
        self.difficulty = "Easy"
        self.goal_completed = False
        self.list_called = False
        self.violations: List[str] = []

        self.tasks_added = 0
        self.high_added = 0
        self.high_on_time = 0
        self.clean_completions = 0
        self.deadline_misses = 0
        self.dep_violations = 0

        self._r_add = 0.0
        self._r_priority = 0.0
        self._r_deadline_set = 0.0
        self._r_complete = 0.0
        self._r_hard_dep_ok = 0.0
        self._r_hard_edge = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._init_state()

        self.difficulty = self._scenarios[self._reset_count % 3]
        self._reset_count += 1

        msgs = {
            "Easy": (
                "EASY MODE\n"
                "Goal: Add 3 tasks with at least 2 different priorities, then call 'list'.\n"
                "Full score (1.0) when 3 tasks + list + 2+ priority levels; "
                "otherwise partial credit if you list without enough variety."
            ),
            "Medium": (
                "MEDIUM MODE\n"
                "Goal: Add 4 tasks with priorities and deadlines. Every High task on time.\n"
                "Rewards: small steps (~+0.07 per signal) for smooth partial credit.\n"
                f"Penalty: -{MEDIUM_DEADLINE_MISS_PENALTY:.2f} per deadline miss."
            ),
            "Hard": (
                "HARD MODE\n"
                "Goal: 5 tasks with deps + deadlines; valid order; no misses.\n"
                "Rewards: tiny edge credit when linking deps on add; rising clean completes; "
                "small bonus when deps satisfied at complete.\n"
                f"Penalties: -{HARD_DEP_VIOLATION_PENALTY:.2f} dependency, -{HARD_DEADLINE_MISS_PENALTY:.2f} deadline miss."
            ),
        }
        return TaskManagerObservation(
            success=True,
            message=msgs[self.difficulty],
            tasks=[],
            violations=[],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: TaskManagerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TaskManagerObservation:
        self._state.step_count += 1
        cmd = (action.command or "").strip().lower()

        if cmd == "add":
            success, msg = self._add(action)
        elif cmd == "complete":
            success, msg = self._complete(action)
        elif cmd == "list":
            self.list_called = True
            success, msg = True, self._fmt_list()
        else:
            success, msg = False, f"Unknown command '{cmd}'. Use: add | complete | list."

        reward = self._score()

        return TaskManagerObservation(
            success=success,
            message=msg,
            tasks=list(self.tasks),
            violations=list(self.violations),
            done=self.goal_completed,
            reward=reward,
            metadata={
                "difficulty": self.difficulty,
                "step": self._state.step_count,
                "tasks_added": self.tasks_added,
                "clean_completions": self.clean_completions,
                "dep_violations": self.dep_violations,
                "deadline_misses": self.deadline_misses,
            },
        )

    def _add(self, action: TaskManagerAction):
        title = (action.title or "").strip()
        if not title:
            return False, "Error: 'title' is required for add."
        if any(t["title"] == title for t in self.tasks):
            return False, f"Error: task '{title}' already exists."

        priority = (action.priority or "Normal").strip()
        if priority not in VALID_PRIORITIES:
            priority = "Normal"

        deadline_str = (action.deadline or "").strip() or None

        depends_on: List[str] = []
        for dep in (action.depends_on or []):
            dep = dep.strip()
            if not dep:
                continue
            if not any(t["title"] == dep for t in self.tasks):
                return False, f"Error: dependency '{dep}' not found. Add it first."
            depends_on.append(dep)

        task: Dict[str, Any] = {
            "title": title,
            "priority": priority,
            "deadline": deadline_str,
            "depends_on": depends_on,
            "completed": False,
            "deadline_missed": False,
            "dep_violation": False,
        }
        self.tasks.append(task)
        self.tasks_added += 1
        if priority == "High":
            self.high_added += 1

        if self.difficulty == "Medium":
            # Smaller per-signal steps (~0.07) so scores rise gradually (avoid large single-step jumps)
            self._r_add += 0.07
            if priority != "Normal":
                self._r_priority += 0.07
            if deadline_str:
                self._r_deadline_set += 0.07
        elif self.difficulty == "Hard":
            # Slightly smaller per-signal steps than Medium to smooth the last adds (e.g. Deploy)
            inc = 0.062
            self._r_add += inc
            if priority != "Normal":
                self._r_priority += inc
            if deadline_str:
                self._r_deadline_set += inc
            if depends_on:
                self._r_hard_edge += 0.012 * float(len(depends_on))

        parts = [f"Task '{title}' added (priority={priority}"]
        if deadline_str:
            parts[0] += f", deadline={deadline_str}"
        if depends_on:
            parts[0] += f", depends_on={depends_on}"
        parts[0] += ")."
        return True, " ".join(parts)

    def _complete(self, action: TaskManagerAction):
        title = (action.title or "").strip()
        if not title:
            return False, "Error: 'title' is required for complete."

        task = next((t for t in self.tasks if t["title"] == title), None)
        if task is None:
            return False, f"Error: task '{title}' not found."
        if task["completed"]:
            return False, f"Task '{title}' is already completed."

        msgs = []
        dep_viol = False
        dl_miss = False

        unmet = [
            d for d in task["depends_on"]
            if not any(t["title"] == d and t["completed"] for t in self.tasks)
        ]
        if unmet:
            dep_viol = True
            task["dep_violation"] = True
            self.dep_violations += 1
            v = f"DEP VIOLATION: '{title}' completed before {unmet}."
            self.violations.append(v)
            msgs.append("(!) Dependency violation — penalty applied.")

        dl = _parse_date(task["deadline"])
        if dl and _today() > dl:
            dl_miss = True
            task["deadline_missed"] = True
            self.deadline_misses += 1
            v = f"DEADLINE MISSED: '{title}' was due {task['deadline']} (completed after deadline)."
            self.violations.append(v)
            msgs.append("(!) Deadline missed — penalty applied.")
            if self.difficulty == "Medium":
                msgs.append(
                    f"Score impact (Medium): -{MEDIUM_DEADLINE_MISS_PENALTY:.2f} for this miss "
                    f"(total deadline-miss penalty: "
                    f"{self.deadline_misses * MEDIUM_DEADLINE_MISS_PENALTY:.2f})."
                )
            elif self.difficulty == "Hard":
                msgs.append(
                    f"Score impact (Hard): -{HARD_DEADLINE_MISS_PENALTY:.2f} for this miss "
                    f"(total: {self.deadline_misses * HARD_DEADLINE_MISS_PENALTY:.2f})."
                )

        task["completed"] = True

        if not dep_viol and not dl_miss:
            order_idx = self.clean_completions
            self.clean_completions += 1
            if task["priority"] == "High":
                self.high_on_time += 1
            msgs.append(f"Task '{title}' completed cleanly.")
            if self.difficulty == "Medium":
                if task["priority"] == "High":
                    self._r_complete += 0.09
            elif self.difficulty == "Hard":
                # Rising reward each clean step; tiny extra when prerequisites were satisfied
                step_r = 0.065 + 0.012 * float(order_idx)
                self._r_complete += min(0.125, step_r)
                if task["depends_on"]:
                    self._r_hard_dep_ok += 0.025
        else:
            msgs.append(f"Task '{title}' completed with violations.")

        return True, " ".join(msgs)

    def _score(self) -> float:
        if self.difficulty == "Easy":
            return self._score_easy()
        if self.difficulty == "Medium":
            return self._score_medium()
        return self._score_hard()

    def _easy_has_two_priorities(self) -> bool:
        if len(self.tasks) < 3:
            return False
        return len({t["priority"] for t in self.tasks}) >= 2

    def _score_easy(self) -> float:
        # Partial: ~+0.11/add, ~+0.12/list. Full 1.0 + done only with 2+ distinct priorities + list.
        r = min(3, self.tasks_added) * 0.11
        if self.list_called:
            r += 0.12
        if self.tasks_added >= 3 and self.list_called:
            if self._easy_has_two_priorities():
                r += 0.55
                self.goal_completed = True
            else:
                r += 0.40
        return round(min(1.0, r), 3)

    def _score_medium(self) -> float:
        # Softer caps match smaller increments (4×0.07 ≈ 0.28 per bucket)
        r = min(0.28, self._r_add)
        r += min(0.28, self._r_priority)
        r += min(0.28, self._r_deadline_set)
        r += min(0.22, self._r_complete)
        r -= self.deadline_misses * MEDIUM_DEADLINE_MISS_PENALTY

        high_tasks = [t for t in self.tasks if t["priority"] == "High"]
        all_high_on_time = (
            len(high_tasks) >= 1
            and all(t["completed"] and not t["deadline_missed"] for t in high_tasks)
        )
        if self.tasks_added >= 4 and all_high_on_time and self.deadline_misses == 0:
            self.goal_completed = True
            return 1.0
        return round(min(0.94, max(0.0, r)), 3)

    def _score_hard(self) -> float:
        # Caps aligned with 0.062 increments (~0.31 over 5 adds) to avoid one huge step on task 5
        r = min(0.34, self._r_add)
        r += min(0.26, self._r_priority)
        r += min(0.26, self._r_deadline_set)
        r += min(0.42, self._r_complete)
        r += min(0.14, self._r_hard_dep_ok)
        r += min(0.08, self._r_hard_edge)
        topo_bonus = 0.0
        if self.clean_completions == 5 and self.dep_violations == 0:
            topo_bonus = 0.055
        r += topo_bonus
        r -= self.dep_violations * HARD_DEP_VIOLATION_PENALTY
        r -= self.deadline_misses * HARD_DEADLINE_MISS_PENALTY

        all_done = self.tasks_added >= 5 and all(t["completed"] for t in self.tasks)
        if all_done and self.dep_violations == 0 and self.deadline_misses == 0:
            self.goal_completed = True
            return 1.0
        return round(min(0.99, max(0.0, r)), 3)

    def _fmt_list(self) -> str:
        if not self.tasks:
            return "No tasks yet."
        lines = [f"Tasks ({len(self.tasks)}):"]
        for i, t in enumerate(self.tasks, 1):
            status = "DONE" if t["completed"] else "PENDING"
            flags = []
            if t.get("deadline_missed"):
                flags.append("LATE")
            if t.get("dep_violation"):
                flags.append("DEP-ERR")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            dl = f" due={t['deadline']}" if t["deadline"] else ""
            dep = f" deps={t['depends_on']}" if t["depends_on"] else ""
            lines.append(f"  {i}. [{status}]{flag_str} {t['title']} ({t['priority']}{dl}{dep})")
        return "\n".join(lines)

    @property
    def state(self) -> State:
        return self._state
