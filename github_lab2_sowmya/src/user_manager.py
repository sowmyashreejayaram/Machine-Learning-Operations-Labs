"""User Manager - Handle user accounts"""

class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, username, email):
        if username in self.users:
            return None
        user = {"username": username, "email": email, "active": True}
        self.users[username] = user
        return user
    
    def get_user(self, username):
        return self.users.get(username)
    
    def deactivate_user(self, username):
        if username in self.users:
            self.users[username]["active"] = False
            return True
        return False
