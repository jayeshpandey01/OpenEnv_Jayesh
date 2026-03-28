---
title: OpenEnv Jayesh - Smart Personal Task Manager
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - task-manager
  - ai-agent
  - planning
---

# Smart Personal Task Manager - OpenEnv Jayesh

> An AI agent environment for managing tasks with priorities, deadlines, and dependencies -- built for the **OpenEnv Hackathon Round 1**.

- HF Space: https://huggingface.co/spaces/jayesh20/openenv_jayesh
- Python 3.10+

---

## What is this?

A **real-world Task Manager environment** where an AI agent must add, prioritize, and complete tasks while respecting deadlines and dependency constraints. The environment cycles through three meaningfully distinct difficulty levels -- each demanding progressively more sophisticated planning.

This environment targets real-world utility: the kind of task scheduling problems that users, productivity apps, and organizational tools deal with every day.

---

## Difficulty Levels

| Level | Goal | Key Constraints |
|-------|------|-----------------|
| **Easy** | Add 2-3 tasks, then `list` them | None -- basic task CRUD |
| **Medium** | Add 4 tasks with priorities & deadlines; complete all High-priority before deadline | Deadline enforcement, priority management |
| **Hard** | Add 5 tasks with priorities, deadlines, AND dependencies; complete in valid topological order | Dependency ordering + deadline enforcement + penalty accumulation |

---

## Action Space

```python
TaskManagerAction(
    command   = "add",            # "add" | "complete" | "list"
    title     = "Fix critical bug",  # required for add / complete
    priority  = "High",           # "Low" | "Normal" | "High"  (default: "Normal")
    deadline  = "2026-04-15",     # ISO-8601 date (optional; relevant in Medium & Hard)
    depends_on= ["Reproduce bug"] # list of prerequisite task titles (Hard only)
)
```

### Commands
| Command | Description |
|---------|-------------|
| `add` | Create a new task. Returns error if title already exists or dependency is unresolved. |
| `complete` | Mark a task as done. Checks deadline & dependency constraints and applies penalties. |
| `list` | Display all current tasks with status, priority, deadline, and dependency info. |

---

## Observation Space

```python
TaskManagerObservation(
    success    = True,                   # whether the last action succeeded
    message    = "Task 'Fix bug' added", # status message or error description
    tasks      = [...],                  # full task list snapshot
    violations = [...],                  # list of rule violations this episode
    reward     = 0.45,                   # cumulative partial reward (0.0-1.0)
    done       = False,                  # True when episode goal is achieved
    metadata   = {
        "difficulty": "Hard",
        "step": 7,
        "tasks_added": 5,
        "tasks_completed": 3,
        "deadline_misses": 0,
        "dependency_violations": 0
    }
)
```

### Task Object Fields
| Field | Type | Description |
|-------|------|-------------|
| `title` | str | Task name |
| `priority` | str | `"Low"` / `"Normal"` / `"High"` |
| `deadline` | str | ISO-8601 date or `"none"` |
| `depends_on` | list[str] | Prerequisite task titles |
| `completed` | bool | Whether the task is done |
| `deadline_missed` | bool | True if completed after deadline |
| `dependency_violation` | bool | True if completed before all prerequisites |

---

## Reward Function

### Easy Mode
| Event | Reward |
|-------|--------|
| Each task added (up to 3) | +0.15 |
| Calling `list` | +0.20 |
| **Goal: >=2 tasks added + list called** | **1.0** |

### Medium Mode
| Event | Reward |
|-------|--------|
| Each task added (up to 4) | +0.15 |
| Each task with explicit non-Normal priority | +0.10 |
| Each High-priority task completed **on time** | +0.20 |
| Deadline missed | **-0.25** |
| **Goal: 4 tasks, >=2 High, all High completed on time** | **1.0** |

### Hard Mode
| Event | Reward |
|-------|--------|
| Each task added (up to 5) | +0.15 |
| Each task with non-Normal priority | +0.10 |
| Each task completed without any violation | +0.25 |
| Perfect run bonus (all done, zero violations) | **+0.10** |
| Dependency violation | **-0.30** |
| Deadline missed | **-0.25** |
| **Goal: 5 tasks, >=2 High, all completed, zero violations** | **1.0** |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --host 127.0.0.1 --port 8000

# In another terminal, run the inference demo
python inference.py
```

---

## Usage Examples

### Easy Mode
```python
env = OpenenvJayeshEnvironment()
obs = env.reset()   # cycles to Easy

env.step(TaskManagerAction(command="add", title="Buy groceries", priority="Normal"))
# reward: 0.15

env.step(TaskManagerAction(command="add", title="Call dentist", priority="Low"))
# reward: 0.30

obs = env.step(TaskManagerAction(command="list"))
# reward: 1.0, done: True
```

### Medium Mode
```python
obs = env.reset()   # cycles to Medium

env.step(TaskManagerAction(command="add", title="Fix critical bug", priority="High", deadline="2026-04-15"))
env.step(TaskManagerAction(command="add", title="Deploy hotfix",    priority="High", deadline="2026-04-16"))
env.step(TaskManagerAction(command="add", title="Write release notes", priority="Normal", deadline="2026-04-22"))
env.step(TaskManagerAction(command="add", title="Team prep",        priority="Low"))

env.step(TaskManagerAction(command="complete", title="Fix critical bug"))   # +0.20 on-time
obs = env.step(TaskManagerAction(command="complete", title="Deploy hotfix"))  # +0.20 -> done=True, reward=1.0
```

### Hard Mode (with dependencies)
```python
obs = env.reset()   # cycles to Hard

env.step(TaskManagerAction(command="add", title="Reproduce bug", priority="High",   deadline="2026-04-15"))
env.step(TaskManagerAction(command="add", title="Write tests",   priority="Normal", deadline="2026-04-16"))
env.step(TaskManagerAction(command="add", title="Write fix",     priority="High",   deadline="2026-04-18",
    depends_on=["Reproduce bug"]))
env.step(TaskManagerAction(command="add", title="Code review",   priority="Normal", deadline="2026-04-20",
    depends_on=["Write fix", "Write tests"]))
env.step(TaskManagerAction(command="add", title="Deploy",        priority="Low",    deadline="2026-04-22",
    depends_on=["Code review"]))

# Complete in valid topological order
env.step(TaskManagerAction(command="complete", title="Reproduce bug"))  # no deps
env.step(TaskManagerAction(command="complete", title="Write tests"))    # no deps
env.step(TaskManagerAction(command="complete", title="Write fix"))      # dep met
env.step(TaskManagerAction(command="complete", title="Code review"))    # deps met
obs = env.step(TaskManagerAction(command="complete", title="Deploy"))   # done=True, reward=1.0 + bonus
```

---

## Environment Design Rationale

### Why these three levels?
- **Easy** establishes baseline task CRUD competency -- can the agent perform basic operations?
- **Medium** adds time pressure and priority trade-offs -- a realistic proxy for real project management.
- **Hard** requires multi-step planning with constraint satisfaction -- approximates real dependency scheduling (e.g., CI/CD pipelines, project Gantt charts).

### Why partial rewards?
Smooth, dense reward signals (+0.15 per task, +0.10 per priority, etc.) enable reinforcement learning agents to make meaningful progress even without solving the full episode. This is superior to sparse reward environments where only terminal success counts.

### Why penalties?
- Deadline misses (-0.25) discourage agents from completing tasks arbitrarily late.
- Dependency violations (-0.30) teach agents that **order matters** -- a fundamental property of real-world task graphs.

---

## API Endpoints

| URL | Description |
|-----|-------------|
| `GET /health` | Health check |
| `POST /reset` | Start a new episode |
| `POST /step` | Execute an action |
| `GET /state` | Current episode metadata |
| `GET /docs` | Interactive Swagger UI |

---

## Project Structure

```
openenv_jayesh/
+-- Dockerfile
+-- openenv.yaml
+-- pyproject.toml
+-- models.py                  <- Action + Observation types
+-- client.py                  <- HTTP client helper
+-- inference.py               <- End-to-end demo (all 3 levels)
+-- server/
    +-- app.py                 <- FastAPI app entry point
    +-- openenv_jayesh_environment.py  <- Core environment logic
```

---

## Deploy

```bash
openenv push --repo-id jayesh20/openenv_jayesh
```

Live space: https://huggingface.co/spaces/jayesh20/openenv_jayesh
