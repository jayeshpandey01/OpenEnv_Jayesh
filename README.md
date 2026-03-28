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

An AI agent environment for managing tasks across three difficulty levels, built for the OpenEnv Hackathon Round 1.

- HF Space: https://huggingface.co/spaces/jayesh20/openenv_jayesh
- Python 3.10+

---

## What is this?

A real-world Task Manager environment where an AI agent must add, prioritize, and complete tasks to earn rewards. The environment cycles through Easy, Medium, and Hard difficulty levels, each with stricter goals and richer reward signals.

---

## Difficulty Levels

| Level  | Goal |
|--------|------|
| Easy   | Add 2 tasks and list them |
| Medium | Add 3 tasks (mixed priorities), complete all High priority ones |
| Hard   | Add 4 tasks (at least 2 High priority), complete at least 2 High priority tasks |

---

## Action Space

```python
TaskManagerAction(
    command="add",         # "add" | "complete" | "list"
    title="Fix bug",       # required for add / complete
    priority="High",       # "Low" | "Normal" | "High"  (optional, default: Normal)
    deadline="2026-04-01"  # optional
)
```

---

## Observation Space

```python
TaskManagerObservation(
    success=True,          # whether the action succeeded
    message="Task added",  # status message
    tasks=[...],           # current task list
    reward=0.4,            # partial progress score (0.0 to 1.0)
    done=False             # True when episode goal is achieved
)
```

---

## Reward Function

### Easy
| Event | Reward |
|-------|--------|
| Add 1st task | +0.4 |
| Add 2nd task | +0.4 |
| Call list | +0.2 |
| Goal complete | 1.0 |

### Medium
| Event | Reward |
|-------|--------|
| Each task added (up to 3) | +0.2 each |
| Adding a High priority task | +0.1 |
| Completing High priority tasks | +0.3 (proportional) |
| Goal complete | 1.0 |

### Hard
| Event | Reward |
|-------|--------|
| Each task added (up to 4) | +0.1 each |
| Each High priority task added (up to 2) | +0.1 each |
| Each High priority task completed (up to 2) | +0.2 each |
| Goal complete | 1.0 |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uvicorn server.app:app --host 127.0.0.1 --port 8000

# In another terminal, run inference
python inference.py
```

---

## Usage Example

```python
from server.openenv_jayesh_environment import OpenenvJayeshEnvironment
from models import TaskManagerAction

env = OpenenvJayeshEnvironment()

obs = env.reset()
print(obs.message)
# Task Manager started in Easy mode. Goal: Add 2 tasks and list them.

obs = env.step(TaskManagerAction(command="add", title="Buy groceries", priority="Normal"))
print(obs.reward)  # 0.4

obs = env.step(TaskManagerAction(command="add", title="Fix bug", priority="High"))
print(obs.reward)  # 0.8

obs = env.step(TaskManagerAction(command="list"))
print(obs.reward)  # 1.0
print(obs.done)    # True
```

---

## API Endpoints

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/docs | Swagger UI - interactive API testing |
| http://127.0.0.1:8000/health | Health check |
| POST /reset | Start a new episode |
| POST /step | Execute an action |
| GET /state | Get current episode state |

---

## Project Structure

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

---

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id jayesh20/openenv_jayesh
```

Live space: https://huggingface.co/spaces/jayesh20/openenv_jayesh
