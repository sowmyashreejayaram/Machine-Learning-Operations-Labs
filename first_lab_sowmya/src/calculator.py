"""
Simple Calculator Module
This module contains basic mathematical operations.

MODIFICATION: Added fun5() for division - this makes it different from the original!
"""

def fun1(x, y):
    """Add two numbers together."""
    return x + y


def fun2(x, y):
    """Subtract y from x."""
    return x - y


def fun3(x, y):
    """Multiply two numbers."""
    return x * y


def fun4(x, y):
    """
    Combine all operations above.
    Returns: x + y + (x - y) + (x * y)
    """
    return fun1(x, y) + fun2(x, y) + fun3(x, y)


# YOUR MODIFICATION: New function that the original doesn't have!
def fun5(x, y):
    """
    Divide x by y.
    Returns the division result or None if division by zero.
    """
    if y == 0:
        return None  # Cannot divide by zero
    return x / y


# Example usage
if __name__ == "__main__":
    print("Calculator Demo:")
    print(f"5 + 3 = {fun1(5, 3)}")
    print(f"5 - 3 = {fun2(5, 3)}")
    print(f"5 * 3 = {fun3(5, 3)}")
    print(f"Combined: {fun4(5, 3)}")
    print(f"5 / 3 = {fun5(5, 3)}")  # NEW!
    print(f"5 / 0 = {fun5(5, 0)}")  # Handles division by zero
