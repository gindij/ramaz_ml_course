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
   python test.py
   ```
   You should see test results showing 0 points (since functions aren't implemented yet).

### Quick Start

1. **Open the assignment files in your favorite editor**
2. **Start with Part 1** (`part1_from_scratch.py`)
3. **Implement the functions** by replacing `pass` with your code
4. **Test your progress** regularly with `python test.py --part 1`

## Assignment Structure

The assignment is divided into three parts, each in a separate Python file:

### Part 1: From Scratch Implementation (17 points)
**File:** `part1_from_scratch.py`

Implement basic linear algebra operations using only Python's built-in data structures:
- Vector operations: addition, scalar multiplication, dot product, magnitude, normalization
- Matrix operations: addition, matrix-vector multiplication, matrix multiplication, transpose

*No NumPy allowed in this part - build understanding from first principles.*

**Difficulty breakdown:**
- **Easy (1 pt):** vector_add, scalar_multiply, matrix_add
- **Medium (2 pts):** dot_product, vector_magnitude, matrix_vector_multiply, matrix_transpose  
- **Hard (3 pts):** normalize_vector, matrix_multiply

### Part 2: NumPy Essentials (22 points)
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

**Difficulty breakdown:**
- **Easy (1 pt):** create_zeros_array, create_range_matrix, get_diagonal, numpy_matrix_multiply, find_max_indices, stack_arrays, filter_positive_values
- **Medium (2 pts):** create_scaled_identity_matrix, get_submatrix, matrix_plus_row_vector, matrix_plus_column_vector, conditional_values, compute_row_means
- **Hard (3 pts):** normalize_features

### Part 3: Advanced Problems (14 points)
**File:** `part3_advanced_problems.py`

Challenging problems requiring mathematical thinking:
- **Permutation matrices**: Create matrices that reorder rows according to a given ordering
- **Back substitution**: Implement the algorithm to solve upper triangular systems
- **2D geometric transformations**: Rotation, scaling, and translation matrices  
- **Matrix composition**: Combine multiple transformations in the correct order

**Difficulty breakdown:**
- **Easy (1 pt):** scaling_matrix_2d
- **Medium (2 pts):** rotation_matrix_2d, translation_matrix_2d
- **Hard (3 pts):** create_permutation_matrix, back_substitution, complex_transformation_challenge

## Testing Your Work

The assignment includes a comprehensive testing system:

```bash
# Test all parts
python test.py

# Test specific parts
python test.py --part 1
python test.py --part 2  
python test.py --part 3
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
*** LINEAR ALGEBRA ASSIGNMENT - TEST RESULTS
============================================================

*** PART 1: FROM SCRATCH IMPLEMENTATION
----------------------------------------
[PASS] vector_add                      1 /  1 points
[PASS] scalar_multiply                 1 /  1 points
[FAIL] dot_product                     0 /  2 points
     -> 2/5 test cases passed
     -> FAILED on input 0: [[1, 2, 3], [4, 5, 6]]
      Expected: 32, Got: 30
     -> FAILED on input 3: [[-1, 2, -3], [1, 2, 3]]
      Expected: -6, Got: -8
[WARN] matrix_multiply                 NOT IMPLEMENTED

Part 1 Score: 2/17 points (11.8%)

*** FINAL SCORE: 2/53 points (3.8%) - KEEP WORKING
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
├── part1_from_scratch.py              # Part 1: Implement from scratch (17 pts)
├── part2_numpy_essentials.py          # Part 2: Learn NumPy (22 pts)
├── part3_advanced_problems.py         # Part 3: Advanced problems (14 pts)
├── test.py                            # Simple test runner
├── test_runner.py                     # Detailed test system
├── test_cases.json                    # Pre-computed test cases
└── generate_test_cases.py             # Test case generator (for reference)
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

**Total: 53 points across 21 functions**

**Point System:**
- **Easy (1 point):** Simple, single-concept problems
- **Medium (2 points):** Multi-step problems requiring understanding  
- **Hard (3 points):** Complex problems requiring research and deep thinking

| Part | File | Functions | Points |
|------|------|-----------|--------|
| 1 | `part1_from_scratch.py` | 9 functions | 17 |
| 2 | `part2_numpy_essentials.py` | 14 functions | 22 |
| 3 | `part3_advanced_problems.py` | 6 functions | 14 |

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
python test.py
```

**Functions Returning None:**
```bash
# Replace 'pass' with your implementation
def vector_add(v1, v2):
    pass  # <- Replace this with your code
```

Good luck!