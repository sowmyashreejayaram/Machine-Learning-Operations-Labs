"""
Unittest test file for calculator functions
"""
import unittest
import sys
import os

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calculator import fun1, fun2, fun3, fun4, fun5


class TestCalculator(unittest.TestCase):
    """Test cases for calculator functions."""

    def test_fun1(self):
        """Test addition function."""
        self.assertEqual(fun1(3, 2), 5)
        self.assertEqual(fun1(-1, 1), 0)
        self.assertEqual(fun1(0, 0), 0)

    def test_fun2(self):
        """Test subtraction function."""
        self.assertEqual(fun2(5, 3), 2)
        self.assertEqual(fun2(10, 5), 5)
        self.assertEqual(fun2(0, 0), 0)

    def test_fun3(self):
        """Test multiplication function."""
        self.assertEqual(fun3(4, 5), 20)
        self.assertEqual(fun3(0, 10), 0)
        self.assertEqual(fun3(-2, 3), -6)

    def test_fun4(self):
        """Test combined function."""
        # fun4(5,3) = fun1(5,3) + fun2(5,3) + fun3(5,3)
        #           = (5+3) + (5-3) + (5*3)
        #           = 8 + 2 + 15 = 25
        self.assertEqual(fun4(5, 3), 25)

    def test_fun5(self):
        """Test division function (YOUR MODIFICATION!)."""
        self.assertEqual(fun5(10, 2), 5)
        self.assertEqual(fun5(9, 3), 3)
        self.assertIsNone(fun5(5, 0))  # Division by zero


if __name__ == '__main__':
    unittest.main()
