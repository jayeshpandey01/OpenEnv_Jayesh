---
title: OpenEnv Jayesh - Task Manager
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - task-manager
---

# OpenEnv Jayesh - Task Manager Environment

An AI agent environment for managing tasks across three difficulty levels.

HF Space: https://huggingface.co/spaces/jayesh20/openenv_jayesh

---

## Difficulty levels

| Level | Goal | Scoring (summary) |
|-------|------|---------------------|
| **Easy** | Add **3** tasks with **at least two different** priority levels among them, then call **`list`** so all current tasks are shown. | Partial credit builds with **~+0.11** per add and **~+0.12** for `list`. **Full reward (1.0)** and **`done=True`** only when there are **3 tasks**, **`list`** was called, and **at least two distinct** priorities appear. If you `list` with three tasks but **only one** priority level, you get a **lower** goal bonus (no full 1.0 / no episode success). |
| **Medium** | Add **4+** tasks, each with a **priority** and **deadline**. **Complete every High-priority task on time** (on or before its deadline). At least **one** High task must exist. | **Gradual** partial credit: small **~+0.07** steps for add / non-Normal / deadline (capped buckets), plus credit for **High** completions on time. **−0.15** per **deadline miss**. **`done=True`** and **1.0** when four tasks are in place, all Highs are done **without** being late, and there are **no** deadline misses. Our inference runner deliberately incurs one deadline miss, then completes extra High-priority tasks to demonstrate **penalty recovery** yielding a strong 0.870 score. |
| **Hard** | Add **5** tasks with **priorities**, **deadlines**, and **dependencies**. Complete **all** tasks in valid **topological** order and **respect every deadline** (no dependency violations, no missed deadlines). | **Adds** use slightly smaller per-field steps than Medium (for smoother score), plus **~+0.012** per **dependency edge** when you set `depends_on`. **Clean completes** rise in steps, with a small extra when **dependencies were satisfied**; **topological bonus** when all five finish cleanly. **−0.15** per deadline miss, **−0.18** per dependency violation. **`done=True`** and **1.0** on a perfect run. |

Penalties are constants in `server/openenv_jayesh_environment.py`. Reward stays in **`[0.0, 1.0]`**.

---

## Action space

```python
TaskManagerAction(
    command="add",              # "add" | "complete" | "list"
    title="Fix bug",
    priority="High",
    deadline="2026-04-01",
    depends_on=["Other task"],
)
```

## Observation space

```python
TaskManagerObservation(
    success=True,
    message="...",
    tasks=[...],
    violations=[...],
    reward=0.0,
    done=False,
)
```

---

## Sample inference output

```text
[START] task=OpenEnvJayesh

======================================================================
--- [EASY] MODE ---
Goal: Basic task addition and list command.
======================================================================
[STEP] step=1 reward=0.110
  -> Action: add 'Buy groceries' (Low)                     | Score: 0.110
[STEP] step=2 reward=0.220
  -> Action: add 'Call dentist' (Normal)                   | Score: 0.220
[STEP] step=3 reward=0.330
  -> Action: add 'Review PR' (High)                        | Score: 0.330
[STEP] step=4 reward=1.000
  -> Action: list tasks (goal completion)                  | Score: 1.000 [DONE]

======================================================================
--- [MEDIUM] MODE ---
Goal: Handle deadlines with a controlled penalty but recover well.
======================================================================
[STEP] step=5 reward=0.210
  -> Action: add 'Fix bug' (High, future)                  | Score: 0.210
[STEP] step=6 reward=0.420
  -> Action: add 'Hotfix' (High, PAST!)                    | Score: 0.420
[STEP] step=7 reward=0.630
  -> Action: add 'Standup' (Low, future)                   | Score: 0.630
[STEP] step=8 reward=0.840
  -> Action: add 'Docs' (Low, future)                      | Score: 0.840
[STEP] step=9 reward=0.840
  -> Action: add 'DB Backup' (High, future)                | Score: 0.840
[STEP] step=10 reward=0.930
  -> Action: complete 'Fix bug' [on time]                  | Score: 0.930
[STEP] step=11 reward=0.780
  -> Action: complete 'Hotfix' [MISSES DEADLINE]           | Score: 0.780
     (!) DEADLINE MISSED: 'Deploy hotfix' was due 2020-01-01 (completed after deadline).
[STEP] step=12 reward=0.780
  -> Action: complete 'Standup' [on time]                  | Score: 0.780
[STEP] step=13 reward=0.780
  -> Action: complete 'Docs' [on time]                     | Score: 0.780
[STEP] step=14 reward=0.870
  -> Action: complete 'DB Backup' [recovery]               | Score: 0.870

======================================================================
--- [HARD] MODE ---
Goal: Perfect dependency handling and deadline compliance.
======================================================================
[STEP] step=15 reward=0.186
  -> Action: add 'Reproduce bug' (High)                    | Score: 0.186
[STEP] step=16 reward=0.310
  -> Action: add 'Write tests' (Normal)                    | Score: 0.310
[STEP] step=17 reward=0.508
  -> Action: add 'Write fix' (dep: Reproduce bug)          | Score: 0.508
[STEP] step=18 reward=0.656
  -> Action: add 'Code review' (dep: Write fix, tests)     | Score: 0.656
[STEP] step=19 reward=0.804
  -> Action: add 'Deploy' (dep: Code review)               | Score: 0.804
[STEP] step=20 reward=0.869
  -> Action: complete 'Reproduce bug'                      | Score: 0.869
[STEP] step=21 reward=0.946
  -> Action: complete 'Write tests'                        | Score: 0.946
[STEP] step=22 reward=0.990
  -> Action: complete 'Write fix'                          | Score: 0.990
[STEP] step=23 reward=0.990
  -> Action: complete 'Code review'                        | Score: 0.990
[STEP] step=24 reward=1.000
  -> Action: complete 'Deploy' [goal]                      | Score: 1.000 [DONE]

======================================================================
--- [DEMO] VIOLATION DEMO ---
Goal: Intentionally violate dependencies to test penalty logic.
======================================================================
[STEP] step=25 reward=0.186
  -> Action: add 'Task A'                                  | Score: 0.186
[STEP] step=26 reward=0.322
  -> Action: add 'Task B' (dep: Task A)                    | Score: 0.322
[STEP] step=27 reward=0.142
  -> Action: WRONG: complete 'Task B' before 'Task A'      | Score: 0.142
     (!) DEP VIOLATION: 'Task B' completed before ['Task A'].

======================================================================
  [EASY] SCORE   : 1.000
  [MEDIUM] SCORE : 0.870 (with controlled recovery)
  [HARD] SCORE   : 1.000
  [DEMO] SCORE   : 0.142 (not in avg)
  [FINAL] AVERAGE: 0.957 (target >= 0.95)
======================================================================

Inference completed successfully.
[END] task=OpenEnvJayesh score=0.957 steps=27 done=True
```

---

## Quick start

```bash
uv sync
uvicorn server.app:app --host 127.0.0.1 --port 8000
python inference.py
```

## API endpoints

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/docs | Swagger UI |
| http://127.0.0.1:8000/health | Health check |
| POST /reset | Start new episode |
| POST /step | Execute action |
| GET /state | Current episode state |

## Project structure

```
openenv_jayesh/
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── models.py
├── client.py
├── inference.py
└── server/
    ├── app.py
    ├── openenv_jayesh_environment.py
    └── requirements.txt
```

## Deploy

```bash
openenv push --repo-id jayesh20/openenv_jayesh
```
