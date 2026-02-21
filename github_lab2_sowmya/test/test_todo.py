import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.todo_manager import TodoManager

def test_add_task():
    manager = TodoManager()
    task = manager.add_task("Buy groceries")
    assert task["name"] == "Buy groceries"

def test_complete_task():
    manager = TodoManager()
    task = manager.add_task("Test")
    manager.complete_task(task["id"])
    assert manager.get_task(task["id"])["completed"] == True
