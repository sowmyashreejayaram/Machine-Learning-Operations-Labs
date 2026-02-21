"""
Pytest test file for calculator functions
"""
import sys
import os

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calculator import fun1, fun2, fun3, fun4, fun5


def test_fun1():
    """Test addition function."""
    assert fun1(3, 2) == 5
    assert fun1(-1, 1) == 0
    assert fun1(0, 0) == 0


def test_fun2():
    """Test subtraction function."""
    assert fun2(5, 3) == 2
    assert fun2(10, 5) == 5
    assert fun2(0, 0) == 0


def test_fun3():
    """Test multiplication function."""
    assert fun3(4, 5) == 20
    assert fun3(0, 10) == 0
    assert fun3(-2, 3) == -6


def test_fun4():
    """Test combined function."""
    # fun4(5,3) = fun1(5,3) + fun2(5,3) + fun3(5,3)
    #           = (5+3) + (5-3) + (5*3)
    #           = 8 + 2 + 15 = 25
    assert fun4(5, 3) == 25


def test_fun5():
    """Test division function (YOUR MODIFICATION!)."""
    assert fun5(10, 2) == 5
    assert fun5(9, 3) == 3
    assert fun5(5, 0) is None  # Division by zero returns None
