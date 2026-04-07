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


def step(env, action: TaskManagerAction, label: str, step_num: int):
    obs = env.step(action)
    done_tag = " [DONE]" if obs.done else ""
    print(f"  {label:<50} | score={obs.reward:.3f}{done_tag}", flush=True)
    print(f"[STEP] step={step_num} reward={obs.reward:.3f}", flush=True)
    msg = obs.message or ""
    if obs.violations and (
        "(!)" in msg
        or "Score impact (Medium):" in msg
        or "Score impact (Hard):" in msg
        or "Dependency violation" in msg
    ):
        print(f"    (!) {obs.violations[-1]}", flush=True)
    for marker in ("Score impact (Medium):", "Score impact (Hard):"):
        if marker in msg:
            sub = msg[msg.index(marker) :]
            if "). Task" in sub:
                line = sub.split("). Task", 1)[0] + ")."
            else:
                line = sub.strip()
            print(f"    >> {line}", flush=True)
            break
    return obs


def run_easy():
    print(f"[START] task=EasyMode", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("EASY MODE  (perfect: 3 tasks + list)", flush=True)
    print("=" * 70, flush=True)
    env = OpenenvJayeshEnvironment()
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

    step(env, TaskManagerAction(command="add", title="Buy groceries", priority="Low"), "add 'Buy groceries' Low", 1)
    step(env, TaskManagerAction(command="add", title="Call dentist", priority="Normal"), "add 'Call dentist' Normal", 2)
    step(env, TaskManagerAction(command="add", title="Review PR", priority="High"), "add 'Review PR' High", 3)
    obs = step(env, TaskManagerAction(command="list"), "list (shows all tasks)", 4)
    print(f"[END] task=EasyMode score={obs.reward:.3f} steps=4", flush=True)
    return obs.reward


def run_medium():
    print(f"[START] task=MediumMode", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("MEDIUM MODE  (one deadline miss on a High task)", flush=True)
    print("=" * 70, flush=True)
    env = OpenenvJayeshEnvironment()
    env._reset_count = 1
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

    future = "2099-12-31"
    past = "2020-01-01"

    step(env, TaskManagerAction(command="add", title="Fix critical bug", priority="High", deadline=future), "add 'Fix critical bug' High  deadline=future", 1)
    step(env, TaskManagerAction(command="add", title="Deploy hotfix", priority="High", deadline=past), "add 'Deploy hotfix' High  deadline=PAST", 2)
    step(env, TaskManagerAction(command="add", title="Write release notes", priority="Normal", deadline=future), "add 'Write release notes' Normal", 3)
    step(env, TaskManagerAction(command="add", title="Standup prep", priority="Low", deadline=future), "add 'Standup prep' Low", 4)
    step(env, TaskManagerAction(command="complete", title="Fix critical bug"), "complete 'Fix critical bug' [on-time]", 5)
    obs = step(env, TaskManagerAction(command="complete", title="Deploy hotfix"), "complete 'Deploy hotfix' [DEADLINE MISSED]", 6)
    print(f"[END] task=MediumMode score={obs.reward:.3f} steps=6", flush=True)
    return obs.reward


def run_hard():
    print(f"[START] task=HardMode", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("HARD MODE  (perfect topological order, all deadlines met)", flush=True)
    print("=" * 70, flush=True)
    env = OpenenvJayeshEnvironment()
    env._reset_count = 2
    obs = env.reset()
    print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

    future = "2099-12-31"

    step(env, TaskManagerAction(command="add", title="Reproduce bug", priority="High", deadline=future), "add 'Reproduce bug' High", 1)
    step(env, TaskManagerAction(command="add", title="Write tests", priority="Normal", deadline=future), "add 'Write tests' Normal", 2)
    step(
        env,
        TaskManagerAction(command="add", title="Write fix", priority="High", deadline=future, depends_on=["Reproduce bug"]),
        "add 'Write fix' High  (dep: Reproduce bug)",
        3,
    )
    step(
        env,
        TaskManagerAction(command="add", title="Code review", priority="Normal", deadline=future, depends_on=["Write fix", "Write tests"]),
        "add 'Code review'  (dep: Write fix, Write tests)",
        4,
    )
    step(
        env,
        TaskManagerAction(command="add", title="Deploy to production", priority="Low", deadline=future, depends_on=["Code review"]),
        "add 'Deploy'  (dep: Code review)",
        5,
    )

    print(flush=True)
    step(env, TaskManagerAction(command="complete", title="Reproduce bug"), "complete 'Reproduce bug'", 6)
    step(env, TaskManagerAction(command="complete", title="Write tests"), "complete 'Write tests'", 7)
    step(env, TaskManagerAction(command="complete", title="Write fix"), "complete 'Write fix'", 8)
    step(env, TaskManagerAction(command="complete", title="Code review"), "complete 'Code review'", 9)
    obs = step(env, TaskManagerAction(command="complete", title="Deploy to production"), "complete 'Deploy'  [goal]", 10)
    print(f"[END] task=HardMode score={obs.reward:.3f} steps=10", flush=True)

    print("\n" + f"[START] task=ViolationDemo", flush=True)
    print("  -- Violation demo (fresh Hard episode) --", flush=True)
    env2 = OpenenvJayeshEnvironment()
    env2._reset_count = 2
    env2.reset()
    step(env2, TaskManagerAction(command="add", title="Task A", priority="High", deadline=future), "add 'Task A' High", 1)
    step(env2, TaskManagerAction(command="add", title="Task B", priority="Normal", deadline=future, depends_on=["Task A"]), "add 'Task B'  (dep: Task A)", 2)
    obs2 = step(env2, TaskManagerAction(command="complete", title="Task B"), "WRONG: complete 'Task B' before 'Task A'", 3)
    print(f"  Score after dep violation: {obs2.reward:.3f}  (penalty applied)", flush=True)
    print(f"[END] task=ViolationDemo score={obs2.reward:.3f} steps=3", flush=True)

    return obs.reward


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("  Task Manager OpenEnv - Inference Runner", flush=True)
    print("=" * 70, flush=True)

    easy_score = run_easy()
    medium_score = run_medium()
    hard_score = run_hard()

    avg = (easy_score + medium_score + hard_score) / 3

    print("\n" + "=" * 70, flush=True)
    print(f"  EASY   final score : {easy_score:.3f}", flush=True)
    print(f"  MEDIUM final score : {medium_score:.3f}  (deadline miss penalty)", flush=True)
    print(f"  HARD   final score : {hard_score:.3f}", flush=True)
    print(f"  AVERAGE            : {avg:.3f}", flush=True)
    print("=" * 70, flush=True)
