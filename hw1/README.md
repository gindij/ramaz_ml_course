# Linear Algebra Programming Assignment

This assignment will help you get familiar with linear algebra from both math and programming perspectives. In the first part, you will implement some basic linear algebra from scratch. In the second part, you will get familiar for numpy, a very popular numerical computing python library. In the third part, you will do some more challenging exercises to test your understanding.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Basic understanding of Python programming

### Installation

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd ramaz_ml_course_2025/hw1
   ```

2. **Install required packages:**
   ```bash
   pip install numpy
   ```

3. **Verify your setup:**
   ```bash
   python test.py --part 1
   ```
   You should see test results showing 0 points (since functions aren't implemented yet).

### Quick Start

1. **Open the assignment files in your favorite editor**
2. **Start with Part 1** (`part1_from_scratch.py`)
3. **Implement the functions** by replacing `pass` with your code
4. **Test your progress** regularly with `python test.py --part 1`

## Assignment Structure

The assignment is divided into three parts, each in a separate Python file:

### Part 1: From Scratch Implementation (25 points)
**File:** `part1_from_scratch.py`

Implement basic linear algebra operations using only Python's built-in data structures:
- Vector operations: addition, scalar multiplication, dot product, magnitude, normalization
- Matrix operations: addition, matrix-vector multiplication, matrix multiplication, transpose

*No NumPy allowed in this part - build understanding from first principles.*

**Functions (9 total):**
- **vector_add** [2 pts]: Element-wise vector addition
- **scalar_multiply** [2 pts]: Multiply vector by scalar
- **dot_product** [3 pts]: Vector dot product
- **vector_magnitude** [2 pts]: Calculate vector length
- **normalize_vector** [3 pts]: Create unit vector
- **matrix_add** [3 pts]: Element-wise matrix addition
- **matrix_vector_multiply** [3 pts]: Matrix times vector
- **matrix_multiply** [4 pts]: Matrix multiplication
- **matrix_transpose** [3 pts]: Transpose matrix

### Part 2: NumPy Essentials (37 points)
**File:** `part2_numpy_essentials.py`

Learn NumPy's core functionality:
- Array creation and manipulation
- Indexing and slicing
- Broadcasting operations
- Boolean masking and filtering
- Finding indices (argmax/argmin)
- Array stacking and concatenation
- Linear algebra functions
- Statistical operations

**Functions (14 total):**
- **create_zeros_array** [2 pts]: Create zero-filled array
- **create_scaled_identity_matrix** [3 pts]: Scaled identity matrix
- **create_range_matrix** [2 pts]: Matrix with sequential values
- **get_diagonal** [2 pts]: Extract diagonal elements
- **get_submatrix** [3 pts]: Extract submatrix using slicing
- **matrix_plus_row_vector** [3 pts]: Broadcasting addition
- **matrix_plus_column_vector** [3 pts]: Broadcasting addition
- **conditional_values** [3 pts]: Boolean masking
- **numpy_matrix_multiply** [3 pts]: Matrix multiplication
- **compute_row_means** [2 pts]: Row-wise averages
- **normalize_features** [4 pts]: Standardize columns
- **find_max_indices** [2 pts]: Locate maximum values
- **stack_arrays** [3 pts]: Combine arrays
- **filter_positive_values** [2 pts]: Filter using conditions

### Part 3: Advanced Problems (21 points)
**File:** `part3_advanced_problems.py`

Challenging problems requiring mathematical thinking:
- **Permutation matrices**: Create matrices that reorder rows according to a given ordering
- **Back substitution**: Implement the algorithm to solve upper triangular systems
- **2D geometric transformations**: Rotation, scaling, and translation matrices  
- **Matrix composition**: Combine multiple transformations in the correct order

**Functions (6 total):**
- **create_permutation_matrix** [3 pts]: Create row reordering matrix
- **back_substitution** [4 pts]: Solve upper triangular system
- **rotation_matrix_2d** [3 pts]: 2D rotation transformation
- **scaling_matrix_2d** [3 pts]: 2D scaling transformation  
- **translation_matrix_2d** [3 pts]: 2D translation transformation
- **complex_transformation_challenge** [5 pts]: Combine multiple transformations

## Testing Your Work

Use the unified test script to test any part or combination:

```bash
# Test all parts
python test.py

# Test individual parts
python test.py --part 1
python test.py --part 2
python test.py --part 3

# Test specific functions (auto-detects which part they belong to)
python test.py vector_add dot_product
python test.py create_zeros_array normalize_features

# Test specific functions in a specific part
python test.py --part 1 vector_add dot_product

# Test a different file (e.g., reference solution)
python test.py --file part1_solution --part 1
```

### Understanding Test Output

The testing system provides:
- **Individual function results** ([PASS] / [FAIL])
- **Points earned** for each function
- **Total percentage score** out of 100%
- **Detailed breakdown** by part
- **Specific failure information** showing which inputs failed and why

**Example Test Output:**
```
*** HW1 PART 1: LINEAR ALGEBRA FROM SCRATCH - TEST RESULTS
============================================================

[PASS] vector_add                      2 /  2 points
     -> 5/5 test cases passed
[PASS] scalar_multiply                 2 /  2 points
     -> 5/5 test cases passed
[FAIL] dot_product                     0 /  3 points
     -> 2/5 test cases passed
     -> FAILED on input 0: [[1, 2, 3], [4, 5, 6]]
      Expected: 32, Got: 30
     -> FAILED on input 3: [[-1, 2, -3], [1, 2, 3]]
      Expected: -6, Got: -8
[FAIL] matrix_multiply                 0 /  4 points
     -> 0/5 test cases passed
     -> NOT IMPLEMENTED

PASSING FINAL SCORE: 4/25 points (16.0%) - KEEP WORKING
============================================================
```

### Implementation Tips

- **Read the docstrings carefully** - they contain examples and explanations
- **Test frequently** - run tests after implementing each function
- **Research when stuck** - Part 3 problems may require looking up concepts

## File Structure

```
hw1/
├── README.md                          # This file
├── test.py                            # Unified test runner for all parts
├── part1_from_scratch.py              # Part 1: Implement from scratch (25 pts)
├── part1_test_cases.json              # Part 1 test cases
├── part2_numpy_essentials.py          # Part 2: Learn NumPy (37 pts)
├── part2_test_cases.json              # Part 2 test cases
├── part3_advanced_problems.py         # Part 3: Advanced problems (21 pts)
└── part3_test_cases.json              # Part 3 test cases
```

## Key Mathematical Concepts Covered

- **Vectors**: Addition, scalar multiplication, dot product, magnitude, normalization
- **Matrices**: Addition, multiplication, transpose
- **NumPy**: Broadcasting, indexing, linear algebra functions
- **Permutation matrices**: Row reordering operations
- **Triangular systems**: Back substitution algorithm
- **2D transformations**: Rotation, scaling, translation, homogeneous coordinates
- **Matrix composition**: Combining multiple transformations

## Scoring System

**Point System:**
- Functions are weighted based on complexity and learning objectives
- Each part focuses on different aspects of linear algebra

| Part | File | Functions | Points |
|------|------|-----------|--------|
| 1 | `part1_from_scratch.py` | 9 functions | 25 |
| 2 | `part2_numpy_essentials.py` | 14 functions | 37 |
| 3 | `part3_advanced_problems.py` | 6 functions | 21 |

## Getting Help
- Discuss with your classmates
- Look things up online or in the textbool
- Send me an email (though response times may be a little bit long)

Good luck!
