import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import OpenenvJayeshEnv
from models import TaskManagerAction

def run_easy_task(client):
    print("\n=== Scenario: Easy ===")
    res = client.reset()
    default_msg = res.observation.message if hasattr(res.observation, 'message') else "No message"
    print(f"Goal: {default_msg}")

    res = client.step(TaskManagerAction(command="add", title="Buy groceries", priority="Normal"))
    print(f"Action 'add 1': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Do laundry", priority="Low"))
    print(f"Action 'add 2': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="list"))
    print(f"Action 'list': {res.observation.message} | Score: {res.reward} | Done: {res.done}")

def run_medium_task(client):
    print("\n=== Scenario: Medium ===")
    res = client.reset()
    default_msg = res.observation.message if hasattr(res.observation, 'message') else "No message"
    print(f"Goal: {default_msg}")
    
    res = client.step(TaskManagerAction(command="add", title="Write code", priority="Normal"))
    print(f"Action 'add Normal': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Review PR", priority="High"))
    print(f"Action 'add High': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Check emails", priority="Low"))
    print(f"Action 'add Low': {res.observation.message} | Score: {res.reward}")
    
    res = client.step(TaskManagerAction(command="complete", title="Review PR"))
    print(f"Action 'complete High': {res.observation.message} | Score: {res.reward} | Done: {res.done}")

def run_hard_task(client):
    print("\n=== Scenario: Hard ===")
    res = client.reset()
    default_msg = res.observation.message if hasattr(res.observation, 'message') else "No message"
    print(f"Goal: {default_msg}")
    
    res = client.step(TaskManagerAction(command="add", title="Fix prod bug", priority="High"))
    print(f"Action 'add High 1': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Write incident report", priority="High"))
    print(f"Action 'add High 2': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Refactor", priority="Normal"))
    print(f"Action 'add Normal': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="add", title="Update docs", priority="Low"))
    print(f"Action 'add Low': {res.observation.message} | Score: {res.reward}")
    
    res = client.step(TaskManagerAction(command="complete", title="Fix prod bug"))
    print(f"Action 'complete High 1': {res.observation.message} | Score: {res.reward}")
    res = client.step(TaskManagerAction(command="complete", title="Write incident report"))
    print(f"Action 'complete High 2': {res.observation.message} | Score: {res.reward} | Done: {res.done}")


if __name__ == "__main__":
    print("Connecting to OpenEnv Task Manager Server on http://127.0.0.1:8000...")
    try:
        with OpenenvJayeshEnv(base_url="http://127.0.0.1:8000").sync() as client:
            run_easy_task(client)
            run_medium_task(client)
            run_hard_task(client)
    except Exception as e:
        print(f"Error connecting to server. Is it running? {e}")
