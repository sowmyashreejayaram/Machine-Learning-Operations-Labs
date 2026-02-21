"""Todo Manager - Task management system"""

class TodoManager:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task_name, description=""):
        task = {
            "id": len(self.tasks) + 1,
            "name": task_name,
            "description": description,
            "completed": False
        }
        self.tasks.append(task)
        return task
    
    def get_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                return task
        return None
    
    def complete_task(self, task_id):
        task = self.get_task(task_id)
        if task:
            task["completed"] = True
            return True
        return False
    
    def get_all_tasks(self):
        return self.tasks
    
    def get_pending_tasks(self):
        return [task for task in self.tasks if not task["completed"]]

    def set_priority(self, task_id, priority):
        """Set priority: high, medium, low"""
        task = self.get_task(task_id)
        if task and priority in ["high", "medium", "low"]:
            task["priority"] = priority
            return True
        return False
    
    def get_high_priority_tasks(self):
        """Get all high priority tasks"""
        return [t for t in self.tasks if t.get("priority") == "high"]
