"""
inference.py - Task Manager OpenEnv standalone test runner.

Runs three scenarios (no server). Easy and Hard show successful runs; Medium shows one deadline miss.
Hard includes a separate violation demo episode.

Usage:
    python inference.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.openenv_jayesh_environment import OpenenvJayeshEnvironment
from models import TaskManagerAction


def step(env, action: TaskManagerAction, label: str):
    obs = env.step(action)
    done_tag = " [DONE]" if obs.done else ""
    print(f"  {label:<50} | score={obs.reward:.3f}{done_tag}")
    msg = obs.message or ""
    if obs.violations and (
        "(!)" in msg
        or "Score impact (Medium):" in msg
        or "Score impact (Hard):" in msg
        or "Dependency violation" in msg
    ):
        print(f"    (!) {obs.violations[-1]}")
    for marker in ("Score impact (Medium):", "Score impact (Hard):"):
        if marker in msg:
            sub = msg[msg.index(marker) :]
            if "). Task" in sub:
                line = sub.split("). Task", 1)[0] + ")."
            else:
                line = sub.strip()
            print(f"    >> {line}")
            break
    return obs


def run_easy():
    print("\n" + "=" * 70)
    print("EASY MODE  (perfect: 3 tasks + list)")
    print("=" * 70)
    env = OpenenvJayeshEnvironment()
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n")

    step(env, TaskManagerAction(command="add", title="Buy groceries", priority="Low"), "add 'Buy groceries' Low")
    step(env, TaskManagerAction(command="add", title="Call dentist", priority="Normal"), "add 'Call dentist' Normal")
    step(env, TaskManagerAction(command="add", title="Review PR", priority="High"), "add 'Review PR' High")
    obs = step(env, TaskManagerAction(command="list"), "list (shows all tasks)")
    return obs.reward


def run_medium():
    print("\n" + "=" * 70)
    print("MEDIUM MODE  (one deadline miss on a High task)")
    print("=" * 70)
    env = OpenenvJayeshEnvironment()
    env._reset_count = 1
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n")

    future = "2099-12-31"
    past = "2020-01-01"

    step(env, TaskManagerAction(command="add", title="Fix critical bug", priority="High", deadline=future), "add 'Fix critical bug' High  deadline=future")
    step(env, TaskManagerAction(command="add", title="Deploy hotfix", priority="High", deadline=past), "add 'Deploy hotfix' High  deadline=PAST")
    step(env, TaskManagerAction(command="add", title="Write release notes", priority="Normal", deadline=future), "add 'Write release notes' Normal")
    step(env, TaskManagerAction(command="add", title="Standup prep", priority="Low", deadline=future), "add 'Standup prep' Low")
    step(env, TaskManagerAction(command="complete", title="Fix critical bug"), "complete 'Fix critical bug' [on-time]")
    obs = step(env, TaskManagerAction(command="complete", title="Deploy hotfix"), "complete 'Deploy hotfix' [DEADLINE MISSED]")
    return obs.reward


def run_hard():
    print("\n" + "=" * 70)
    print("HARD MODE  (perfect topological order, all deadlines met)")
    print("=" * 70)
    env = OpenenvJayeshEnvironment()
    env._reset_count = 2
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n")

    future = "2099-12-31"

    step(env, TaskManagerAction(command="add", title="Reproduce bug", priority="High", deadline=future), "add 'Reproduce bug' High")
    step(env, TaskManagerAction(command="add", title="Write tests", priority="Normal", deadline=future), "add 'Write tests' Normal")
    step(
        env,
        TaskManagerAction(command="add", title="Write fix", priority="High", deadline=future, depends_on=["Reproduce bug"]),
        "add 'Write fix' High  (dep: Reproduce bug)",
    )
    step(
        env,
        TaskManagerAction(command="add", title="Code review", priority="Normal", deadline=future, depends_on=["Write fix", "Write tests"]),
        "add 'Code review'  (dep: Write fix, Write tests)",
    )
    step(
        env,
        TaskManagerAction(command="add", title="Deploy to production", priority="Low", deadline=future, depends_on=["Code review"]),
        "add 'Deploy'  (dep: Code review)",
    )

    print()
    step(env, TaskManagerAction(command="complete", title="Reproduce bug"), "complete 'Reproduce bug'")
    step(env, TaskManagerAction(command="complete", title="Write tests"), "complete 'Write tests'")
    step(env, TaskManagerAction(command="complete", title="Write fix"), "complete 'Write fix'")
    step(env, TaskManagerAction(command="complete", title="Code review"), "complete 'Code review'")
    obs = step(env, TaskManagerAction(command="complete", title="Deploy to production"), "complete 'Deploy'  [goal]")

    print()
    print("  -- Violation demo (fresh Hard episode) --")
    env2 = OpenenvJayeshEnvironment()
    env2._reset_count = 2
    env2.reset()
    step(env2, TaskManagerAction(command="add", title="Task A", priority="High", deadline=future), "add 'Task A' High")
    step(env2, TaskManagerAction(command="add", title="Task B", priority="Normal", deadline=future, depends_on=["Task A"]), "add 'Task B'  (dep: Task A)")
    obs2 = step(env2, TaskManagerAction(command="complete", title="Task B"), "WRONG: complete 'Task B' before 'Task A'")
    print(f"  Score after dep violation: {obs2.reward:.3f}  (penalty applied)")

    return obs.reward


if __name__ == "__main__":
    print("=" * 70)
    print("  Task Manager OpenEnv - Inference Runner")
    print("=" * 70)

    easy_score = run_easy()
    medium_score = run_medium()
    hard_score = run_hard()

    avg = (easy_score + medium_score + hard_score) / 3

    print("\n" + "=" * 70)
    print(f"  EASY   final score : {easy_score:.3f}")
    print(f"  MEDIUM final score : {medium_score:.3f}  (deadline miss penalty)")
    print(f"  HARD   final score : {hard_score:.3f}")
    print(f"  AVERAGE            : {avg:.3f}")
    print("=" * 70)
