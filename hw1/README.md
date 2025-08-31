# Linear Algebra Programming Assignment

A comprehensive assignment for high school seniors to learn linear algebra through programming, progressing from basic concepts to advanced mathematical problems.

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
   python part1_test.py
   ```
   You should see test results showing 0 points (since functions aren't implemented yet).

### Quick Start

1. **Open the assignment files in your favorite editor**
2. **Start with Part 1** (`part1_from_scratch.py`)
3. **Implement the functions** by replacing `pass` with your code
4. **Test your progress** regularly with `python part1_test.py`

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

Each part has its own testing system:

```bash
# Test individual parts
python part1_test.py
python part2_test.py
python part3_test.py

# Test specific functions within a part
python part1_test.py vector_add dot_product
python part2_test.py create_zeros_array normalize_features
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

## How to Work on the Assignment

1. **Start with Part 1** - Build conceptual understanding by implementing operations from scratch
2. **Move to Part 2** - Learn efficient NumPy operations
3. **Challenge yourself with Part 3** - Solve advanced problems requiring research and mathematical thinking

### Implementation Tips

- **Read the docstrings carefully** - they contain examples and explanations
- **Test frequently** - run tests after implementing each function
- **Don't skip Part 1** - conceptual understanding is crucial
- **Research when stuck** - Part 3 problems may require looking up concepts
- **Use the examples** - every function has usage examples in the docstrings

## File Structure

```
hw1/
├── README.md                          # This file
├── part1_from_scratch.py              # Part 1: Implement from scratch (25 pts)
├── part1_test.py                      # Part 1 test runner
├── part1_test_cases.json              # Part 1 test cases
├── part2_numpy_essentials.py          # Part 2: Learn NumPy (37 pts)
├── part2_test.py                      # Part 2 test runner
├── part2_test_cases.json              # Part 2 test cases
├── part3_advanced_problems.py         # Part 3: Advanced problems (21 pts)
├── part3_test.py                      # Part 3 test runner
└── part3_test_cases.json              # Part 3 test cases
```

## Learning Objectives

By completing this assignment, students will:

1. **Understand linear algebra conceptually** through from-scratch implementation
2. **Master NumPy** for efficient numerical computing
3. **Develop problem-solving skills** with advanced mathematical challenges
4. **Learn practical applications** like geometric transformations
5. **Build foundation skills** for future work in data science, computer graphics, and engineering

## Key Mathematical Concepts Covered

- **Vectors**: Addition, scalar multiplication, dot product, magnitude, normalization
- **Matrices**: Addition, multiplication, transpose
- **NumPy**: Broadcasting, indexing, linear algebra functions
- **Permutation matrices**: Row reordering operations
- **Triangular systems**: Back substitution algorithm
- **2D transformations**: Rotation, scaling, translation, homogeneous coordinates
- **Matrix composition**: Combining multiple transformations

## Scoring System

**Total: 83 points across 29 functions**

**Point System:**
- Functions are weighted based on complexity and learning objectives
- Each part focuses on different aspects of linear algebra

| Part | File | Functions | Points |
|------|------|-----------|--------|
| 1 | `part1_from_scratch.py` | 9 functions | 25 |
| 2 | `part2_numpy_essentials.py` | 14 functions | 37 |
| 3 | `part3_advanced_problems.py` | 6 functions | 21 |

## Getting Help

- **Read the docstrings** - Every function has detailed documentation with examples
- **Check the test output** - Failed tests show exactly what went wrong
- **Look up concepts** - Part 3 problems may require researching mathematical concepts
- **Ask questions** - If you're stuck, ask about the mathematical concepts, not just code

## Common Issues

**Import Errors:**
```bash
# Make sure NumPy is installed
pip install numpy
```

**Test Not Running:**
```bash
# Make sure you're in the right directory
cd ramaz_ml_course_2025/hw1
python part1_test.py
```

**Functions Returning None:**
```bash
# Replace 'pass' with your implementation
def vector_add(v1, v2):
    pass  # <- Replace this with your code
```

Good luck!