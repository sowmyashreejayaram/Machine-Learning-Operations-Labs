import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.user_manager import UserManager

def test_add_user():
    manager = UserManager()
    user = manager.add_user("john", "john@test.com")
    assert user["username"] == "john"

def test_duplicate_user():
    manager = UserManager()
    manager.add_user("john", "john@test.com")
    result = manager.add_user("john", "other@test.com")
    assert result is None
