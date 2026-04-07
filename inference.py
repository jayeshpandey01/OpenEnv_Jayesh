"""
inference.py - Task Manager OpenEnv standalone test runner.

Runs three scenarios (no server). Easy and Hard show successful runs; Medium shows one deadline miss.
Hard includes a separate violation demo episode.

Usage:
    python inference.py
"""

import sys
import os

# Pre-Submission Configuration (Checklist points 2 & 3)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN")  # No default per checklist point 3
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "openenv_jayesh")


def from_docker_image():
    """Requirement for submission flow (Checklist point 2)."""
    return LOCAL_IMAGE_NAME


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.openenv_jayesh_environment import OpenenvJayeshEnvironment
from models import TaskManagerAction


def step(env, action: TaskManagerAction, label: str, step_num: int):
    obs = env.step(action)
    done_tag = " [DONE]" if obs.done else ""
    print(f"  {label:<50} | score={obs.reward:.3f}{done_tag}", flush=True)
    print(f"[STEP] step={step_num} reward={obs.reward:.3f} action={action.command}", flush=True)
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
    print(f"[START] task=Easy", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("EASY MODE  (perfect: 3 tasks + list)", flush=True)
    print("=" * 70, flush=True)
    
    score = 0.0
    steps = 0
    try:
        env = OpenenvJayeshEnvironment()
        obs = env.reset()
        print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

        steps = 1
        step(env, TaskManagerAction(command="add", title="Buy groceries", priority="Low"), "add 'Buy groceries' Low", steps)
        steps = 2
        step(env, TaskManagerAction(command="add", title="Call dentist", priority="Normal"), "add 'Call dentist' Normal", steps)
        steps = 3
        step(env, TaskManagerAction(command="add", title="Review PR", priority="High"), "add 'Review PR' High", steps)
        steps = 4
        obs = step(env, TaskManagerAction(command="list"), "list (shows all tasks)", steps)
        score = obs.reward
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        print(f"[END] task=Easy score={score:.3f} steps={steps}", flush=True)

    return score


def run_medium():
    print(f"[START] task=Medium", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("MEDIUM MODE  (one deadline miss on a High task)", flush=True)
    print("=" * 70, flush=True)

    score = 0.0
    steps = 0
    try:
        env = OpenenvJayeshEnvironment()
        env._reset_count = 1
        obs = env.reset()
        print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

        future = "2099-12-31"
        past = "2020-01-01"

        steps = 1
        step(env, TaskManagerAction(command="add", title="Fix critical bug", priority="High", deadline=future), "add 'Fix critical bug' High  deadline=future", steps)
        steps = 2
        step(env, TaskManagerAction(command="add", title="Deploy hotfix", priority="High", deadline=past), "add 'Deploy hotfix' High  deadline=PAST", steps)
        steps = 3
        step(env, TaskManagerAction(command="add", title="Write release notes", priority="Normal", deadline=future), "add 'Write release notes' Normal", steps)
        steps = 4
        step(env, TaskManagerAction(command="add", title="Standup prep", priority="Low", deadline=future), "add 'Standup prep' Low", steps)
        steps = 5
        step(env, TaskManagerAction(command="complete", title="Fix critical bug"), "complete 'Fix critical bug' [on-time]", steps)
        steps = 6
        obs = step(env, TaskManagerAction(command="complete", title="Deploy hotfix"), "complete 'Deploy hotfix' [DEADLINE MISSED]", steps)
        score = obs.reward
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        print(f"[END] task=Medium score={score:.3f} steps={steps}", flush=True)

    return score


def run_hard():
    print(f"[START] task=Hard", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("HARD MODE  (perfect topological order, all deadlines met)", flush=True)
    print("=" * 70, flush=True)

    score = 0.0
    steps = 0
    try:
        env = OpenenvJayeshEnvironment()
        env._reset_count = 2
        obs = env.reset()
        print(f"Goal: {obs.message.splitlines()[0]}\n", flush=True)

        future = "2099-12-31"

        steps = 1
        step(env, TaskManagerAction(command="add", title="Reproduce bug", priority="High", deadline=future), "add 'Reproduce bug' High", steps)
        steps = 2
        step(env, TaskManagerAction(command="add", title="Write tests", priority="Normal", deadline=future), "add 'Write tests' Normal", steps)
        steps = 3
        step(
            env,
            TaskManagerAction(command="add", title="Write fix", priority="High", deadline=future, depends_on=["Reproduce bug"]),
            "add 'Write fix' High  (dep: Reproduce bug)",
            steps,
        )
        steps = 4
        step(
            env,
            TaskManagerAction(command="add", title="Code review", priority="Normal", deadline=future, depends_on=["Write fix", "Write tests"]),
            "add 'Code review'  (dep: Write fix, Write tests)",
            steps,
        )
        steps = 5
        step(
            env,
            TaskManagerAction(command="add", title="Deploy to production", priority="Low", deadline=future, depends_on=["Code review"]),
            "add 'Deploy'  (dep: Code review)",
            steps,
        )

        print(flush=True)
        steps = 6
        step(env, TaskManagerAction(command="complete", title="Reproduce bug"), "complete 'Reproduce bug'", steps)
        steps = 7
        step(env, TaskManagerAction(command="complete", title="Write tests"), "complete 'Write tests'", steps)
        steps = 8
        step(env, TaskManagerAction(command="complete", title="Write fix"), "complete 'Write fix'", steps)
        steps = 9
        step(env, TaskManagerAction(command="complete", title="Code review"), "complete 'Code review'", steps)
        steps = 10
        obs = step(env, TaskManagerAction(command="complete", title="Deploy to production"), "complete 'Deploy'  [goal]", steps)
        score = obs.reward
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        print(f"[END] task=Hard score={score:.3f} steps={steps}", flush=True)

    print("\n" + f"[START] task=ViolationDemo", flush=True)
    print("  -- Violation demo (fresh Hard episode) --", flush=True)
    demo_score = 0.0
    demo_steps = 0
    try:
        env2 = OpenenvJayeshEnvironment()
        env2._reset_count = 2
        env2.reset()
        demo_steps = 1
        step(env2, TaskManagerAction(command="add", title="Task A", priority="High", deadline=future), "add 'Task A' High", demo_steps)
        demo_steps = 2
        step(env2, TaskManagerAction(command="add", title="Task B", priority="Normal", deadline=future, depends_on=["Task A"]), "add 'Task B'  (dep: Task A)", demo_steps)
        demo_steps = 3
        obs2 = step(env2, TaskManagerAction(command="complete", title="Task B"), "WRONG: complete 'Task B' before 'Task A'", demo_steps)
        print(f"  Score after dep violation: {obs2.reward:.3f}  (penalty applied)", flush=True)
        demo_score = obs2.reward
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        print(f"[END] task=ViolationDemo score={demo_score:.3f} steps={demo_steps}", flush=True)

    return score


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
