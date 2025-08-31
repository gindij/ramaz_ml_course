# HW2: K-Nearest Neighbors from Scratch

Implement the K-Nearest Neighbors algorithm from first principles to understand fundamental machine learning concepts including classification, regression, and model validation.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Basic understanding of Python programming
- Completion of HW1 (Linear Algebra) recommended

### Installation

1. **Navigate to this assignment:**
   ```bash
   cd ramaz_ml_course/hw2
   ```

2. **No additional packages needed** - this assignment uses only Python built-ins

3. **Verify your setup:**
   ```bash
   python test.py
   ```
   You should see test results showing 0 points (since functions aren't implemented yet).

### Quick Start

1. **Open `knn_from_scratch.py` in your editor**
2. **Start with distance functions** (euclidean_distance, manhattan_distance)
3. **Implement functions by replacing `pass` with your code**
4. **Test frequently** with `python test.py`

## Assignment Overview

This assignment teaches machine learning fundamentals by implementing K-Nearest Neighbors (K-NN) from scratch. You'll build understanding of:

- **Distance metrics** for measuring similarity
- **Neighbor selection** algorithms  
- **Classification and regression** techniques
- **Model validation** and performance metrics
- **Cross-validation** for hyperparameter tuning

**Total: 44 points across 15 functions**

## Function Categories

### 1. Distance Metrics (7 points)
Learn how to measure similarity between data points:

- **euclidean_distance** [EASY - 2 pts]: Standard Euclidean distance
- **manhattan_distance** [EASY - 2 pts]: L1 distance (city block)
- **minkowski_distance** [MEDIUM - 3 pts]: Generalized distance metric

### 2. Neighbor Finding (7 points)
Implement core neighbor selection algorithms:

- **find_k_nearest_neighbors** [MEDIUM - 4 pts]: Find K closest points
- **find_neighbors_within_radius** [MEDIUM - 3 pts]: Find all neighbors within distance

### 3. Classification (9 points)
Build classification prediction methods:

- **majority_vote** [EASY - 2 pts]: Simple voting mechanism
- **weighted_majority_vote** [MEDIUM - 3 pts]: Distance-weighted voting
- **knn_classify** [HARD - 4 pts]: Complete K-NN classification

### 4. Regression (9 points)
Implement continuous value prediction:

- **mean_prediction** [EASY - 2 pts]: Average neighbor values
- **weighted_mean_prediction** [MEDIUM - 3 pts]: Distance-weighted averaging
- **knn_regress** [HARD - 4 pts]: Complete K-NN regression

### 5. Model Validation (12 points)
Learn ML evaluation techniques:

- **train_test_split** [MEDIUM - 3 pts]: Split data for validation
- **calculate_accuracy** [EASY - 2 pts]: Classification performance metric
- **mean_squared_error** [EASY - 2 pts]: Regression performance metric
- **cross_validate_knn** [HARD - 5 pts]: Find optimal K value

## Testing Your Work

The assignment includes comprehensive testing:

```bash
# Test all functions
python test.py

# Test specific functions
python test.py euclidean_distance manhattan_distance
python test.py knn_classify knn_regress
```

### Understanding Test Output

Tests include both **functional correctness** and **ML performance evaluation**:

```
*** HW2: K-NEAREST NEIGHBORS - TEST RESULTS
============================================================

[PASS] euclidean_distance              2 /  2 points
     -> 7/7 test cases passed
[FAIL] manhattan_distance              0 /  2 points
     -> 5/7 test cases passed
     -> FAILED on input 0: [[0, 0], [3, 4]]
      Expected: 7.0, Got: 5.0
[WARN] knn_classify                    NOT IMPLEMENTED

EXCELLENT FINAL SCORE: 35/44 points (79.5%) - PASSING
============================================================
```

## Implementation Guidelines

### Core Concepts to Understand

1. **Distance Metrics**: How to measure similarity between data points
2. **K-NN Algorithm**: Find neighbors, make predictions based on their values
3. **Classification vs Regression**: Discrete labels vs continuous values
4. **Weighted vs Unweighted**: Should closer neighbors have more influence?
5. **Cross-validation**: How to find the best hyperparameters

### Implementation Tips

- **Start with distance functions** - they're used by everything else
- **Test with simple examples** - use small datasets to verify logic
- **Handle edge cases** - empty lists, invalid K values, zero distances
- **Use descriptive variable names** - `neighbor_distances`, not `nd`
- **Add error checking** - validate inputs before processing

### Common Pitfalls

- **Forgetting to sort by distance** when finding neighbors
- **Not handling zero distances** in weighted functions (use large weight)
- **Mixing up classification and regression** logic
- **Off-by-one errors** in cross-validation fold creation
- **Not validating K value** against dataset size

## Learning Objectives

By completing this assignment, you will:

1. **Understand K-NN fundamentals** through hands-on implementation
2. **Learn distance metrics** and their properties
3. **Implement both classification and regression** variants
4. **Practice model validation** techniques
5. **Gain experience with ML evaluation metrics**
6. **Build intuition for hyperparameter tuning**

## File Structure

```
hw2/
├── README.md                          # This file
├── knn_from_scratch.py               # Main assignment (implement these!)
├── test.py                           # Test runner
└── test_cases.json                   # Pre-computed test cases
```

## Key Machine Learning Concepts

- **Supervised Learning**: Learning from labeled training data
- **Instance-based Learning**: Making predictions based on similar examples
- **Lazy Learning**: No explicit training phase, all computation at prediction time
- **Hyperparameter Tuning**: Finding optimal K value through validation
- **Bias-Variance Tradeoff**: Small K (high variance) vs Large K (high bias)
- **Distance Metrics**: Different ways to measure similarity affect results

## Algorithm Complexity

Understanding computational complexity:
- **Training**: O(1) - just store the data
- **Prediction**: O(n*d) - calculate distance to all n points in d dimensions
- **Memory**: O(n*d) - store all training data

## Real-World Applications

K-NN is used in:
- **Recommendation systems** (find similar users/items)
- **Image classification** (find similar images)
- **Anomaly detection** (points with no nearby neighbors)
- **Missing value imputation** (predict based on similar examples)
- **Collaborative filtering** (Netflix, Spotify recommendations)

## Scoring System

**Point Distribution:**
- **Easy (1-2 points)**: Single-concept functions, basic operations
- **Medium (3-4 points)**: Multi-step algorithms, combining concepts  
- **Hard (4-5 points)**: Complex algorithms requiring deep understanding

**Performance Expectations:**
- Focus on correctness over optimization
- Handle edge cases properly
- Implement error checking
- Follow the function signatures exactly

## Getting Help

- **Read function docstrings** - they contain detailed examples
- **Start with simple test cases** - verify logic on small examples
- **Check test output carefully** - shows exactly what failed
- **Understand the math** - K-NN is intuitive once you grasp the concepts
- **Ask conceptual questions** - focus on understanding algorithms, not just coding

## Common Issues

**Function Not Found:**
```bash
# Make sure function names match exactly
def euclidean_distance(...)  # Correct
def euclidian_distance(...)  # Wrong - typo
```

**Distance Calculation Errors:**
```python
# Remember to take square root for Euclidean
return sum((x - y) ** 2 for x, y in zip(point1, point2))  # Wrong
return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))  # Correct
```

**Weighted Voting Issues:**
```python
# Handle zero distance properly
weight = 1000 if distance == 0 else 1 / distance  # Correct
weight = 1 / distance  # Wrong - division by zero
```

Good luck building your K-NN algorithm from scratch!