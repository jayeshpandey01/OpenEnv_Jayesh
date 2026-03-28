"""
inference.py - Demonstration of the Smart Personal Task Manager OpenEnv.

Runs all three difficulty scenarios and prints step-by-step reward signals.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import OpenenvJayeshEnv
from models import TaskManagerAction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(client, action: TaskManagerAction, label: str):
    res = client.step(action)
    obs = res.observation
    done_tag = " [DONE]" if res.done else ""
    violation_tag = ""
    if hasattr(obs, "violations") and obs.violations:
        violation_tag = f"\n      Violations: {obs.violations[-1]}"
    print(
        f"  {label:<40} | Score: {res.reward:.3f}{done_tag}"
        f"\n      -> {obs.message}{violation_tag}"
    )
    return res


# ---------------------------------------------------------------------------
# Easy scenario
# ---------------------------------------------------------------------------

def run_easy(client):
    print("\n" + "=" * 70)
    print("SCENARIO 1 - EASY MODE")
    print("=" * 70)
    res = client.reset()
    print(f"\nGoal:\n{res.observation.message}\n")

    today = "2026-04-15"

    _step(client, TaskManagerAction(command="add", title="Buy groceries", priority="Normal", deadline=today), "add 'Buy groceries' Normal")
    _step(client, TaskManagerAction(command="add", title="Call the dentist", priority="Low"), "add 'Call the dentist' Low")
    _step(client, TaskManagerAction(command="add", title="Review meeting notes", priority="Normal"), "add 'Review meeting notes'")
    _step(client, TaskManagerAction(command="list"), "list (triggers goal completion)")

    print()


# ---------------------------------------------------------------------------
# Medium scenario
# ---------------------------------------------------------------------------

def run_medium(client):
    print("\n" + "=" * 70)
    print("SCENARIO 2 - MEDIUM MODE")
    print("=" * 70)
    res = client.reset()
    print(f"\nGoal:\n{res.observation.message}\n")

    # Use future deadlines so completions are on-time
    today = "2026-04-15"
    tomorrow = "2026-04-16"
    next_week = "2026-04-22"

    _step(client, TaskManagerAction(command="add", title="Fix critical bug", priority="High", deadline=today), "add 'Fix critical bug' High")
    _step(client, TaskManagerAction(command="add", title="Deploy hotfix", priority="High", deadline=tomorrow), "add 'Deploy hotfix' High")
    _step(client, TaskManagerAction(command="add", title="Write release notes", priority="Normal", deadline=next_week), "add 'Write release notes' Normal")
    _step(client, TaskManagerAction(command="add", title="Team standup prep", priority="Low"), "add 'Team standup prep' Low")

    # Complete High-priority tasks on time
    _step(client, TaskManagerAction(command="complete", title="Fix critical bug"), "complete 'Fix critical bug' [High, on-time]")
    _step(client, TaskManagerAction(command="complete", title="Deploy hotfix"), "complete 'Deploy hotfix' [High, on-time -> GOAL]")

    print()


# ---------------------------------------------------------------------------
# Hard scenario
# ---------------------------------------------------------------------------

def run_hard(client):
    print("\n" + "=" * 70)
    print("SCENARIO 3 - HARD MODE (dependencies + deadlines)")
    print("=" * 70)
    res = client.reset()
    print(f"\nGoal:\n{res.observation.message}\n")

    t1  = "2026-04-15"
    t2  = "2026-04-16"
    t3  = "2026-04-18"
    t4  = "2026-04-20"
    t5  = "2026-04-22"

    # Build dependency graph:
    #   Reproduce bug  ---> Write fix  ---> Code review  ---> Deploy
    #   Write tests    ---> Code review
    _step(client, TaskManagerAction(command="add", title="Reproduce bug", priority="High", deadline=t1), "add 'Reproduce bug' High")
    _step(client, TaskManagerAction(command="add", title="Write tests", priority="Normal", deadline=t2), "add 'Write tests' Normal")
    _step(client, TaskManagerAction(command="add", title="Write fix", priority="High", deadline=t3, depends_on=["Reproduce bug"]), "add 'Write fix' High (depends: Reproduce bug)")
    _step(client, TaskManagerAction(command="add", title="Code review", priority="Normal", deadline=t4, depends_on=["Write fix", "Write tests"]), "add 'Code review' Normal (depends: Write fix, Write tests)")
    _step(client, TaskManagerAction(command="add", title="Deploy to production", priority="Low", deadline=t5, depends_on=["Code review"]), "add 'Deploy to prod' Low (depends: Code review)")

    print()
    print("  -- Completing in CORRECT topological order --")
    _step(client, TaskManagerAction(command="complete", title="Reproduce bug"), "complete 'Reproduce bug' [no deps, on-time]")
    _step(client, TaskManagerAction(command="complete", title="Write tests"), "complete 'Write tests' [no deps, on-time]")
    _step(client, TaskManagerAction(command="complete", title="Write fix"), "complete 'Write fix' [dep met, on-time]")
    _step(client, TaskManagerAction(command="complete", title="Code review"), "complete 'Code review' [deps met, on-time]")
    _step(client, TaskManagerAction(command="complete", title="Deploy to production"), "complete 'Deploy to prod' [all deps met -> GOAL]")

    print()
    print("  -- Violations demo (new reset, same Hard mode) --")
    res = client.reset()  # same Hard cycle
    print(f"  (Re-reset into: {res.observation.message.splitlines()[0]})")

    _step(client, TaskManagerAction(command="add", title="Task A", priority="High", deadline=t1), "add 'Task A' High")
    _step(client, TaskManagerAction(command="add", title="Task B", priority="Normal", deadline=t2, depends_on=["Task A"]), "add 'Task B' Normal (depends: Task A)")
    res = _step(client, TaskManagerAction(command="complete", title="Task B"), "WRONG: complete 'Task B' before 'Task A'")
    print(f"      Score after violation: {res.reward:.3f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Smart Personal Task Manager - OpenEnv Inference Demo")
    print("Connecting to http://127.0.0.1:8000 ...")
    print("=" * 70)

    try:
        with OpenenvJayeshEnv(base_url="http://127.0.0.1:8000").sync() as client:
            run_easy(client)
            run_medium(client)
            run_hard(client)
        print("\nAll scenarios completed successfully.")
    except Exception as e:
        print(f"\nError: Could not connect to server. Is it running?\n  {e}")
        sys.exit(1)
