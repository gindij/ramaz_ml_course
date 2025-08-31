# Linear and Logistic Regression from Scratch

A comprehensive assignment for high school seniors to implement both linear and logistic regression from first principles, building deep understanding of optimization and machine learning foundations through gradient descent.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- NumPy (the only allowed external library)
- Basic understanding of linear algebra and calculus

### Installation

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd ramaz_ml_course_2025/hw3
   ```

2. **Install required packages:**
   ```bash
   pip install numpy
   ```

3. **Verify your setup:**
   ```bash
   python test.py
   ```
   You should see test results showing 0 points (since functions aren't implemented yet).

### Quick Start

1. **Open the assignment files** (`utilities.py`, `linear_regression.py`, `logistic_regression.py`)
2. **Read the docstrings carefully** - they contain mathematical derivations and examples
3. **Start with utilities.py** - these functions support both regression types
4. **Implement functions one by one** by replacing `pass` with your code
5. **Test your progress** regularly with `python test.py --part linear` and `python test.py --part logistic`

## Assignment Structure

This assignment focuses on implementing both **linear and logistic regression using gradient descent optimization**. You'll build deep understanding through mathematical derivation and iterative optimization, learning how machines actually "learn" from data.

### Part 1: Shared Utilities (20 points)
**File: `utilities.py` | Functions: 8 | Duration: ~4 hours**

Master the essential shared functionality:
- **standardize_features** [3 pts]: Normalize features to mean=0, std=1
- **add_intercept** [2 pts]: Add bias term for both regression types
- **train_test_split** [3 pts]: Split data for model validation
- **accuracy_score** [2 pts]: Classification evaluation metric
- **mean_squared_error** [2 pts]: Regression evaluation metric
- **gradient_descent** [5 pts]: **[PROVIDED]** Generic optimization algorithm
- **k_fold_split** [2 pts]: Cross-validation fold generation
- **compute_classification_metrics** [1 pt]: Comprehensive classification metrics

*Key Learning: Many ML algorithms share common preprocessing and evaluation functions.*

### Part 2: Linear Regression (22 points)
**File: `linear_regression.py` | Functions: 7 | Duration: ~5 hours**

Implement linear regression through gradient descent:
- **compute_cost** [2 pts]: Mean squared error cost function
- **linear_gradient** [4 pts]: ∂J/∂θ = (1/m) * X^T * (X*θ - y)
- **linear_gradient_descent** [3 pts]: Train using gradient descent
- **predict** [4 pts]: Make predictions using learned parameters
- **polynomial_features** [3 pts]: Generate polynomial basis functions
- **fit_polynomial_regression** [3 pts]: Polynomial regression via gradient descent
- **r_squared** [2 pts]: Coefficient of determination (R²)
- **cross_validate_linear_regression** [1 pt]: K-fold cross-validation

*Key Learning: Gradient descent is the fundamental optimization algorithm in machine learning.*

### Part 3: Logistic Regression (30 points)
**File: `logistic_regression.py` | Functions: 9 | Duration: ~6 hours**

Implement binary classification through gradient descent:
- **sigmoid** [2 pts]: σ(z) = 1/(1 + e^(-z)) activation function
- **sigmoid_derivative** [2 pts]: σ'(z) = σ(z)(1 - σ(z))
- **compute_logistic_cost** [3 pts]: Cross-entropy loss function
- **logistic_cost_from_predictions** [2 pts]: Cost wrapper for gradient descent
- **logistic_gradient** [5 pts]: ∂J/∂θ = (1/m) * X^T * (σ(X*θ) - y)
- **logistic_gradient_descent** [3 pts]: Train using gradient descent
- **predict_proba** [4 pts]: Predict class probabilities
- **predict_classes** [3 pts]: Predict binary classes with threshold
- **cross_validate_logistic_regression** [6 pts]: K-fold cross-validation

*Key Learning: Classification requires different cost functions and outputs probabilities.*

## Mathematical Concepts

### Gradient Descent Optimization
Gradient descent is the fundamental algorithm that powers machine learning:

#### Linear Regression Mathematics
1. **Cost Function**: J(θ) = (1/2m) Σ(h_θ(x^i) - y^i)² where h_θ(x) = θ^T x
2. **Gradient**: ∇J(θ) = (1/m) X^T(Xθ - y)
3. **Parameter Update**: θ = θ - α ∇J(θ)
4. **Repeat until convergence**

#### Logistic Regression Mathematics
1. **Hypothesis**: h_θ(x) = σ(θ^T x) = 1/(1 + e^(-θ^T x))
2. **Cost Function**: J(θ) = -(1/m) Σ[y^i log(h_θ(x^i)) + (1-y^i) log(1-h_θ(x^i))]
3. **Gradient**: ∇J(θ) = (1/m) X^T(σ(Xθ) - y)
4. **Parameter Update**: θ = θ - α ∇J(θ)

### Key Advantages of Gradient Descent
- **Scales to large datasets** (unlike analytical solutions)
- **Works for any differentiable cost function**
- **Foundation for neural networks** and deep learning
- **Intuitive optimization process** (follow the slope downhill)

## Testing Your Work

```bash
# Test all functions
python test.py

# Test specific parts
python test.py --part utilities
python test.py --part linear
python test.py --part logistic

# Test specific functions
python test.py standardize_features linear_gradient sigmoid
```

### Understanding Test Output

The testing system provides:
- **Individual function results** ([PASS] / [FAIL])
- **Points earned** for each function
- **Total percentage score** out of 100%
- **Detailed failure information** showing which inputs failed and expected vs actual outputs

**Example Test Output:**
```
*** HW3: LINEAR & LOGISTIC REGRESSION FROM SCRATCH - TEST RESULTS
================================================================

UTILITIES.PY (20 points):
[PASS] standardize_features            3 /  3 points
[FAIL] gradient_descent                0 /  5 points (PROVIDED - should pass automatically)
     -> Issue: Implementation is provided for you!

LINEAR_REGRESSION.PY (22 points):
[FAIL] linear_gradient                 0 /  4 points
     -> 1/3 test cases passed
     -> FAILED on input 1: Mathematical gradient calculation incorrect

LOGISTIC_REGRESSION.PY (30 points):
[PASS] sigmoid                         2 /  2 points
[FAIL] logistic_gradient              0 /  5 points
     -> Need to implement gradient calculation

PROGRESS FINAL SCORE: 5/72 points (6.9%) - KEEP WORKING
================================================================
```

## Implementation Tips

### Mathematical Implementation
- **Use matrix operations**: Leverage NumPy's linear algebra functions
- **Vectorize computations**: Avoid Python loops when possible
- **Watch for numerical stability**: Use clipping to prevent overflow in sigmoid
- **Understand broadcasting**: NumPy operations work element-wise when dimensions align

### Debugging Strategies
- **Start with simple cases**: Test with perfect linear/separable data first
- **Check matrix dimensions**: Ensure shapes are compatible for operations
- **Verify gradients numerically**: Compare your gradient with numerical approximation
- **Monitor cost function**: Should decrease during gradient descent
- **Check convergence**: Algorithm should stop when cost changes become small

## Common Challenges

### Gradient Descent Issues
```python
# Problem: Learning rate too large (cost increases)
# Solution: Reduce learning rate
learning_rate = 0.001  # Try smaller values

# Problem: Learning rate too small (slow convergence)
# Solution: Increase learning rate or max iterations
learning_rate = 0.1    # Try larger values
```

### Sigmoid Overflow Issues
```python
# Problem: exp(-z) can overflow for large negative z
# Solution: Clip z values to prevent numerical issues
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))
```

### Cross-Validation Indexing
```python
# Problem: Incorrect fold creation
# Solution: Ensure all data points are used exactly once
fold_size = n_samples // k
# Handle remainder samples in the last fold
```

## File Structure

```
hw3/
├── README.md                          # This file
├── utilities.py                       # Shared functions (20 points)
├── linear_regression.py               # Linear regression (22 points)
├── logistic_regression.py             # Logistic regression (30 points)
├── test.py                           # Test runner
├── test_cases.json                   # Test cases (auto-generated)
└── answers.txt                       # Conceptual questions
```

## Learning Objectives

By completing this assignment, students will:

1. **Master gradient descent optimization** - the foundation of machine learning
2. **Understand both regression and classification** paradigms
3. **Implement mathematical derivatives** by hand for cost functions
4. **Compare linear vs logistic regression** applications and differences
5. **Handle polynomial feature engineering** for non-linear relationships
6. **Apply proper model evaluation** for both regression and classification
7. **Build intuition** for when to use each algorithm type
8. **Understand probabilistic outputs** in classification

## Key Mathematical Skills Developed

- **Linear algebra**: Matrix multiplication, transpose, vectorization
- **Calculus**: Partial derivatives, chain rule, gradient computation
- **Optimization**: Gradient descent, learning rates, convergence criteria
- **Probability**: Sigmoid function, log-likelihood, cross-entropy
- **Statistics**: R-squared, accuracy, precision/recall, cross-validation
- **Numerical computing**: Overflow handling, numerical stability

## Scoring System

**Total: 72 points across 24 functions**

| File | Functions | Points | Difficulty |
|------|-----------|--------|------------|
| Utilities | 8 | 20 | Easy-Medium |
| Linear Regression | 7 | 22 | Medium |
| Logistic Regression | 9 | 30 | Medium-Hard |

## Timeline Recommendation (2 weeks)

**Week 1: Foundation (42 points)**
- Days 1-2: Utilities implementation (20 points)
- Days 3-5: Linear regression implementation (22 points)
- Weekend: Review concepts, debug gradient descent

**Week 2: Classification (30 points)**
- Days 1-3: Logistic regression implementation (30 points)
- Days 4-5: Final testing, compare algorithms
- Weekend: Conceptual questions, polish implementations

## Getting Help

- **Read the docstrings** - Every function has detailed mathematical derivations with examples
- **Check the test output** - Failed tests show exactly what went wrong
- **Review calculus** - Khan Academy has excellent derivative and optimization tutorials
- **Understand the math** - Don't just code; understand why gradient descent works
- **Verify gradients** - Check your gradient calculations against numerical approximations
- **Ask conceptual questions** - Focus on "why" rather than "how to code"

## Extensions (Optional)

For students who finish early:
- Implement different learning rate schedules (decay, adaptive)
- Add regularization (L1/L2) to prevent overfitting
- Implement multi-class logistic regression (one-vs-all)
- Compare gradient descent variants (momentum, Adam)
- Visualize gradient descent optimization paths
- Implement feature selection techniques

Good luck building your understanding of machine learning optimization from the ground up!