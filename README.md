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
| **Medium** | Add **4** tasks, each with a **priority** and **deadline**. **Complete every High-priority task on time** (on or before its deadline). At least **one** High task must exist. | **Gradual** partial credit: small **~+0.07** steps for add / non-Normal / deadline (capped buckets), plus credit for **High** completions on time. **−0.15** per **deadline miss**. **`done=True`** and **1.0** when four tasks are in place, all Highs are done **without** being late, and there are **no** deadline misses. |
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

```
======================================================================
  Task Manager OpenEnv - Inference Runner
======================================================================

======================================================================
EASY MODE  (perfect: 3 tasks + list)
======================================================================
Goal: EASY MODE

  add 'Buy groceries' Low                            | score=0.110
  add 'Call dentist' Normal                          | score=0.220
  add 'Review PR' High                               | score=0.330
  list (shows all tasks)                             | score=1.000 [DONE]

======================================================================
MEDIUM MODE  (one deadline miss on a High task)
======================================================================
Goal: MEDIUM MODE

  add 'Fix critical bug' High  deadline=future       | score=0.210
  add 'Deploy hotfix' High  deadline=PAST            | score=0.420
  add 'Write release notes' Normal                   | score=0.560
  add 'Standup prep' Low                             | score=0.770
  complete 'Fix critical bug' [on-time]              | score=0.860
  complete 'Deploy hotfix' [DEADLINE MISSED]         | score=0.710
    (!) DEADLINE MISSED: 'Deploy hotfix' was due 2020-01-01 (completed after deadline).
    >> Score impact (Medium): -0.15 for this miss (total deadline-miss penalty: 0.15).

======================================================================
HARD MODE  (perfect topological order, all deadlines met)
======================================================================
Goal: HARD MODE

  add 'Reproduce bug' High                           | score=0.186
  add 'Write tests' Normal                           | score=0.310
  add 'Write fix' High  (dep: Reproduce bug)         | score=0.508
  add 'Code review'  (dep: Write fix, Write tests)   | score=0.656
  add 'Deploy'  (dep: Code review)                   | score=0.804

  complete 'Reproduce bug'                           | score=0.869
  complete 'Write tests'                             | score=0.946
  complete 'Write fix'                               | score=0.990
  complete 'Code review'                             | score=0.990
  complete 'Deploy'  [goal]                          | score=1.000 [DONE]

  -- Violation demo (fresh Hard episode) --
  add 'Task A' High                                  | score=0.186
  add 'Task B'  (dep: Task A)                        | score=0.322
  WRONG: complete 'Task B' before 'Task A'           | score=0.142
    (!) DEP VIOLATION: 'Task B' completed before ['Task A'].
  Score after dep violation: 0.188  (penalty applied)

======================================================================
  EASY   final score : 1.000
  MEDIUM final score : 0.710  (deadline miss penalty)
  HARD   final score : 1.000
  AVERAGE            : 0.903
======================================================================
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
