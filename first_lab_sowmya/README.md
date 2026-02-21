# GitHub Lab 1 - Calculator with Testing

This project demonstrates basic Python programming, unit testing, and CI/CD with GitHub Actions.

## 📝 Project Description

A simple calculator application with automated testing using both `pytest` and `unittest`.

### ✨ My Modifications

**This project is NOT identical to the original repository!**

I added:
- `fun5()` - A division function with zero-division protection
- Additional test cases for the new division function
- Custom comments and documentation

## 🏗️ Project Structure
```
first_lab_sowmya/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── __init__.py
├── src/
│   ├── __init__.py
│   └── calculator.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
└── .github/
    └── workflows/
        ├── pytest_action.yml
        └── unittest_action.yml
```

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/sowmyashreejayaram/first_lab_sowmya.git
cd first_lab_sowmya
```

### 2. Create a virtual environment
```bash
python3 -m venv github_lab1_env
source github_lab1_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🧪 Running Tests

### Using Pytest
```bash
cd test
pytest test_pytest.py
```

### Using Unittest
```bash
cd test
python test_unittest.py
```

## 📦 Functions

### `fun1(x, y)` - Addition
Adds two numbers together.

### `fun2(x, y)` - Subtraction
Subtracts y from x.

### `fun3(x, y)` - Multiplication
Multiplies two numbers.

### `fun4(x, y)` - Combined Operations
Returns: (x + y) + (x - y) + (x * y)

### `fun5(x, y)` - Division (MY MODIFICATION!)
Divides x by y. Returns `None` if division by zero.

## 🤖 GitHub Actions

This project uses GitHub Actions for automated testing:
- **Pytest Action**: Runs on every push/PR to main branch
- **Unittest Action**: Runs on every push/PR to main branch

## 👤 Author

Sowmya Shree Jayaram  
MLOps Student - Northeastern University

## 📄 License

This project is for educational purposes.
