# Day 3 - Statistics & Probability

**Objective:** Master the statistical foundations needed for data science interviews—descriptive stats, probability, distributions, hypothesis testing, and A/B testing concepts.

> **All code blocks include detailed inline comments and are followed by a line-by-line plain-English explanation.**

---

## Time Budget

| Block | Duration | Focus |
|-------|----------|-------|
| Morning | 2.5 hours | Descriptive stats + Probability basics |
| Afternoon | 2.5 hours | Distributions + Hypothesis testing |
| Evening | 2 hours | A/B testing + Practice problems + Mock interview |
| **Total** | **7 hours** | |

**Micro-blocks:** 45 min study → 10 min break → repeat

---

## Topics Covered

1. Descriptive Statistics (Mean, Median, Mode, Variance, Std Dev)
2. Probability Fundamentals (Rules, Conditional, Bayes)
3. Common Distributions (Normal, Binomial, Poisson)
4. Hypothesis Testing (t-test, p-values, significance)
5. A/B Testing Framework
6. Correlation vs Causation

---

# Topic 1: Descriptive Statistics

## What is Descriptive Statistics? (Theory Deep Dive)

**Descriptive statistics** is the branch of statistics that helps us **summarize and describe** the main features of a dataset. Think of it as creating a "summary report" of your data.

### Why Do We Need Descriptive Statistics?

Imagine you have transaction data for 10 million customers. You can't look at every number! Descriptive statistics gives you:
- **One number** to represent the "typical" value (central tendency)
- **One number** to represent how "spread out" the data is (dispersion)
- **A few numbers** to understand the shape and range (distribution)

### The Two Pillars of Descriptive Statistics

```
Descriptive Statistics
├── 1. Central Tendency (Where is the "center" of the data?)
│   ├── Mean (arithmetic average)
│   ├── Median (middle value)
│   └── Mode (most frequent value)
│
└── 2. Dispersion/Spread (How spread out is the data?)
    ├── Range (max - min)
    ├── Variance (average squared distance from mean)
    ├── Standard Deviation (square root of variance)
    └── IQR - Interquartile Range (middle 50% spread)
```

---

## 1.1 Central Tendency: Mean, Median, Mode

### The Mean (Arithmetic Average)

**What is it?** The mean is the sum of all values divided by the count of values.

**Formula:**
$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} = \frac{x_1 + x_2 + x_3 + ... + x_n}{n}$$

**Symbol meanings:**
- $\bar{x}$ (x-bar) = the mean (for sample data)
- $\mu$ (mu) = the mean (for population data)
- $\sum$ = "sum of" (add up everything)
- $n$ = count of values

**Step-by-step example:**

Transaction values: [20, 30, 40, 50, 60]

```
Step 1: Add all values
        20 + 30 + 40 + 50 + 60 = 200

Step 2: Count the values
        n = 5

Step 3: Divide sum by count
        Mean = 200 ÷ 5 = 40
```

**When to use mean:** When data is roughly symmetric (no extreme outliers).

**Problem with mean:** It's sensitive to outliers!

Example: Salaries [30K, 35K, 40K, 45K, 500K]
- Mean = (30 + 35 + 40 + 45 + 500) ÷ 5 = 650 ÷ 5 = **130K** ❌ (misleading!)
- Most people earn around 40K, but one CEO inflates the average.

---

### The Median (Middle Value)

**What is it?** The median is the middle value when data is sorted in order.

**How to find it:**
1. Sort the data from smallest to largest
2. If odd count: pick the middle value
3. If even count: average the two middle values

**Step-by-step example (odd count):**

Transaction values: [60, 20, 40, 30, 50]

```
Step 1: Sort the data
        [20, 30, 40, 50, 60]

Step 2: Find the middle position
        Position = (n + 1) ÷ 2 = (5 + 1) ÷ 2 = 3rd position

Step 3: The 3rd value is 40
        Median = 40
```

**Step-by-step example (even count):**

Transaction values: [60, 20, 40, 30]

```
Step 1: Sort the data
        [20, 30, 40, 60]

Step 2: Find the two middle positions
        Positions = n÷2 and (n÷2)+1 = 2nd and 3rd positions
        Values at these positions: 30 and 40

Step 3: Average them
        Median = (30 + 40) ÷ 2 = 35
```

**When to use median:** When data has outliers or is skewed (like income, house prices).

**Back to our salary example:** [30K, 35K, 40K, 45K, 500K]
- Median = 40K ✅ (much more representative!)

---

### The Mode (Most Frequent Value)

**What is it?** The mode is the value that appears most often.

**Example:**

Customer ratings: [5, 4, 5, 3, 5, 4, 5, 2, 5]

```
Count each value:
- 5 appears: 5 times ← Most frequent!
- 4 appears: 2 times
- 3 appears: 1 time
- 2 appears: 1 time

Mode = 5
```

**Note:** Data can have:
- **One mode** (unimodal): [1, 2, 2, 3] → mode = 2
- **Two modes** (bimodal): [1, 1, 2, 3, 3] → modes = 1 and 3
- **No mode**: [1, 2, 3, 4] → all appear once

**When to use mode:** For categorical data (like "most common product category").

---

### Mean vs Median: The Key Decision

| Situation | Use | Why |
|-----------|-----|-----|
| Symmetric data (bell-shaped) | Mean | Mean = Median, either works |
| Skewed data / Outliers | Median | Mean gets pulled toward outliers |
| Comparing to "typical" customer | Median | More representative |
| Calculating totals | Mean | Mean × n = Total |

**Interview Tip:** "If mean and median are very different, that signals skewed data. I always check both."

---

## 1.2 Dispersion: Variance and Standard Deviation

### Why Measure Spread?

Two datasets can have the same mean but very different spreads:

```
Dataset A: [48, 49, 50, 51, 52]  → Mean = 50, Data is clustered
Dataset B: [10, 30, 50, 70, 90]  → Mean = 50, Data is spread out
```

Knowing only the mean (50) doesn't tell us if customers spend consistently or wildly differently!

---

### Range (Simplest Measure)

**Formula:** Range = Maximum - Minimum

**Example:** [10, 30, 50, 70, 90]
- Range = 90 - 10 = 80

**Problem:** Range only uses 2 values—one outlier ruins it!

---

### Variance (Average Squared Distance from Mean)

**What is it?** Variance measures how far each value is from the mean, on average.

**Why squared?** 
- Distances can be positive or negative (above or below mean)
- If we just average them, positives and negatives cancel out to zero!
- Squaring makes all distances positive

**Population Variance Formula:**
$$\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}$$

**Sample Variance Formula (Bessel's Correction):**
$$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}$$

**Why n-1 for samples?** When we estimate from a sample, we "use up" one degree of freedom estimating the mean. Dividing by n-1 corrects for this bias.

**Step-by-step example:**

Data: [2, 4, 4, 4, 5, 5, 7, 9]

```
Step 1: Calculate the mean
        Sum = 2+4+4+4+5+5+7+9 = 40
        n = 8
        Mean = 40 ÷ 8 = 5

Step 2: Find each value's distance from mean
        Value   Distance (x - mean)
        2       2 - 5 = -3
        4       4 - 5 = -1
        4       4 - 5 = -1
        4       4 - 5 = -1
        5       5 - 5 = 0
        5       5 - 5 = 0
        7       7 - 5 = +2
        9       9 - 5 = +4

Step 3: Square each distance
        (-3)² = 9
        (-1)² = 1
        (-1)² = 1
        (-1)² = 1
        (0)² = 0
        (0)² = 0
        (+2)² = 4
        (+4)² = 16

Step 4: Sum the squared distances
        9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32

Step 5: Divide by (n-1) for sample variance
        Sample Variance = 32 ÷ 7 = 4.57
```

---

### Standard Deviation (Square Root of Variance)

**What is it?** Standard deviation is variance brought back to the original units.

**Formula:**
$$s = \sqrt{s^2} = \sqrt{\text{variance}}$$

**Why use it?** Variance is in "squared units" (like dollars²), which doesn't make sense. Standard deviation is in original units (dollars).

**From our example:**
```
Variance = 4.57
Standard Deviation = √4.57 = 2.14
```

**Interpretation:** Values typically differ from the mean by about 2.14 units.

---

### Interquartile Range (IQR)

**What is it?** IQR measures the spread of the middle 50% of data.

**Related concepts - Quartiles:**
- **Q1 (25th percentile):** 25% of data is below this value
- **Q2 (50th percentile):** 50% below = The Median!
- **Q3 (75th percentile):** 75% of data is below this value

**Formula:**
$$IQR = Q3 - Q1$$

**Step-by-step example:**

Data (sorted): [2, 4, 4, 4, 5, 5, 7, 9]

```
Step 1: Find Q1 (median of lower half)
        Lower half: [2, 4, 4, 4]
        Q1 = (4 + 4) ÷ 2 = 4

Step 2: Find Q3 (median of upper half)
        Upper half: [5, 5, 7, 9]
        Q3 = (5 + 7) ÷ 2 = 6

Step 3: Calculate IQR
        IQR = Q3 - Q1 = 6 - 4 = 2
```

**Why IQR is useful:** It's robust to outliers! The middle 50% isn't affected by extreme values.

---

### Detecting Outliers with IQR

**Rule:** A value is an outlier if it falls outside these bounds:
- **Lower bound:** Q1 - 1.5 × IQR
- **Upper bound:** Q3 + 1.5 × IQR

**Example:**
```
Q1 = 4, Q3 = 6, IQR = 2

Lower bound = 4 - (1.5 × 2) = 4 - 3 = 1
Upper bound = 6 + (1.5 × 2) = 6 + 3 = 9

Any value < 1 or > 9 is an outlier.
```

---

## 1.3 Python Code for Descriptive Statistics

Now let's see how to calculate everything in Python using pandas.

### Understanding the Libraries First

**pandas:** A Python library for data manipulation. Think of it as Excel in Python.
- **Series:** A single column of data (like one Excel column)
- **DataFrame:** A table with rows and columns (like an Excel sheet)

**numpy:** A library for numerical computations. Provides mathematical functions.

```python
# ============================================================
# IMPORTING LIBRARIES
# ============================================================
# 'import' loads a library into Python so we can use it
# 'as' gives it a shorter nickname for convenience

import pandas as pd   # pd is the standard nickname for pandas
import numpy as np    # np is the standard nickname for numpy

# ============================================================
# CREATING A PANDAS SERIES (A Single Column of Data)
# ============================================================
# pd.Series() creates a one-dimensional array with labels
# The list inside contains our transaction values in dollars

transactions = pd.Series([25, 30, 35, 40, 45, 50, 150])

# Let's see what we created
print("Our data:")
print(transactions)
print()  # Prints a blank line for readability

# Output will look like:
# 0     25     <- Index 0, Value 25
# 1     30     <- Index 1, Value 30
# 2     35     ... and so on
# 3     40
# 4     45
# 5     50
# 6    150     <- This is our outlier!

# ============================================================
# CENTRAL TENDENCY CALCULATIONS
# ============================================================

# ----- MEAN -----
# .mean() calculates the arithmetic average
# Formula: sum of all values ÷ count of values
# Calculation: (25+30+35+40+45+50+150) ÷ 7 = 375 ÷ 7 = 53.57

mean_value = transactions.mean()

print(f"Mean: {mean_value:.2f}")
# Explanation of f-string: f"text {variable:.2f}"
# - f at the start means "formatted string"
# - {mean_value} inserts the variable's value
# - :.2f means "format as float with 2 decimal places"
# Output: Mean: 53.57

# ----- MEDIAN -----
# .median() finds the middle value when sorted
# Our data sorted: [25, 30, 35, 40, 45, 50, 150]
# 7 values, so middle is position 4 (index 3) = 40

median_value = transactions.median()

print(f"Median: {median_value:.2f}")
# Output: Median: 40.00

# Notice: Mean (53.57) > Median (40)
# This tells us data is RIGHT-SKEWED (pulled up by the outlier 150)

# ----- MODE -----
# .mode() finds the most frequent value
# If all values appear once, it returns all of them

mode_value = transactions.mode()

print(f"Mode: {mode_value.values}")
# Output: Mode: [25 30 35 40 45 50 150]
# (All values appear once, so all are "modes" - not useful here)

# ============================================================
# DISPERSION CALCULATIONS
# ============================================================

# ----- VARIANCE -----
# .var() calculates sample variance (uses n-1 by default)
# This is Bessel-corrected for sample data

variance_value = transactions.var()

print(f"Variance (sample, n-1): {variance_value:.2f}")
# Output: Variance (sample, n-1): 1922.62

# For population variance (uses n), specify ddof=0
# ddof = "delta degrees of freedom" = what to subtract from n
variance_pop = transactions.var(ddof=0)

print(f"Variance (population, n): {variance_pop:.2f}")
# Output: Variance (population, n): 1647.96

# ----- STANDARD DEVIATION -----
# .std() calculates sample standard deviation
# It's the square root of variance

std_value = transactions.std()

print(f"Standard Deviation: {std_value:.2f}")
# Output: Standard Deviation: 43.85
# Interpretation: Values typically differ from mean by ~$43.85

# ----- RANGE -----
# .max() returns largest value, .min() returns smallest

range_value = transactions.max() - transactions.min()

print(f"Range: {range_value}")
# Output: Range: 125
# Calculation: 150 - 25 = 125

# ============================================================
# PERCENTILES AND IQR
# ============================================================

# .quantile(p) returns the value where p% of data is below
# p is between 0 and 1 (so 0.25 = 25%)

q1 = transactions.quantile(0.25)   # 25th percentile
q2 = transactions.quantile(0.50)   # 50th percentile = median
q3 = transactions.quantile(0.75)   # 75th percentile

print(f"Q1 (25th percentile): {q1}")
print(f"Q2 (50th percentile / Median): {q2}")
print(f"Q3 (75th percentile): {q3}")

# Calculate IQR
iqr = q3 - q1

print(f"IQR (Q3 - Q1): {iqr}")

# ============================================================
# OUTLIER DETECTION
# ============================================================

# Calculate outlier boundaries
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

# Find values outside bounds
# The | symbol means "OR" in pandas boolean operations
# We need parentheses around each condition

outliers = transactions[
    (transactions < lower_bound) | (transactions > upper_bound)
]

print(f"Outliers: {outliers.values}")
# Output: Outliers: [150]
# The value 150 is identified as an outlier!

# ============================================================
# DESCRIBE() - ONE COMMAND SUMMARY
# ============================================================

# .describe() gives you all basic stats at once
print("\n=== DESCRIBE OUTPUT ===")
print(transactions.describe())

# Output:
# count      7.00    <- Number of non-null values
# mean      53.57    <- Arithmetic average
# std       43.85    <- Standard deviation
# min       25.00    <- Minimum value
# 25%       32.50    <- Q1 (25th percentile)
# 50%       40.00    <- Median (50th percentile)
# 75%       47.50    <- Q3 (75th percentile)
# max      150.00    <- Maximum value
```

**Line-by-Line Code Explanation Summary:**

| Line | What It Does |
|------|--------------|
| `import pandas as pd` | Loads pandas library, calls it 'pd' |
| `pd.Series([...])` | Creates a 1D array with an index |
| `.mean()` | Calculates arithmetic average |
| `.median()` | Finds the middle value |
| `.mode()` | Finds most frequent value(s) |
| `.var()` | Calculates variance (n-1 by default) |
| `.std()` | Calculates standard deviation |
| `.quantile(0.25)` | Finds value at 25th percentile |
| `.describe()` | Shows summary statistics |
| `f"text {var:.2f}"` | Formatted string with 2 decimal places |

---

# Topic 2: Probability Fundamentals

## What is Probability? (Theory Deep Dive)

**Probability** is a number between 0 and 1 that measures how likely something is to happen.

- **P = 0:** Impossible (will never happen)
- **P = 1:** Certain (will definitely happen)
- **P = 0.5:** Equally likely to happen or not happen (like a fair coin flip)

**Common notations:**
- P(A) = Probability of event A happening
- P(not A) or P(A') = Probability of event A NOT happening
- P(A and B) = Probability of BOTH A and B happening
- P(A or B) = Probability of EITHER A or B (or both) happening
- P(A|B) = Probability of A GIVEN that B has already happened

---

## 2.1 Basic Probability Formula

$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

**Example:** Rolling a die, what's P(getting a 4)?
```
Favorable outcomes: 1 (only the number 4)
Total outcomes: 6 (numbers 1, 2, 3, 4, 5, 6)

P(4) = 1/6 = 0.1667 = 16.67%
```

---

## 2.2 Complement Rule

The probability of something NOT happening equals 1 minus the probability of it happening.

$$P(\text{not A}) = 1 - P(A)$$

**Example:** If P(rain) = 0.30, then P(no rain) = 1 - 0.30 = 0.70

---

## 2.3 Addition Rule (OR)

**When can A and B NOT happen together (mutually exclusive):**
$$P(A \text{ or } B) = P(A) + P(B)$$

**Example:** P(rolling 1 OR 6) = 1/6 + 1/6 = 2/6 = 1/3

**When A and B CAN happen together (not mutually exclusive):**
$$P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$$

We subtract P(A and B) because we counted it twice!

**Example:** In a deck of 52 cards, P(Heart OR Queen)?
```
P(Heart) = 13/52 (13 hearts in deck)
P(Queen) = 4/52 (4 queens in deck)
P(Heart AND Queen) = 1/52 (only 1 queen of hearts)

P(Heart OR Queen) = 13/52 + 4/52 - 1/52 = 16/52 = 0.308
```

---

## 2.4 Multiplication Rule (AND)

**When A and B are independent (one doesn't affect the other):**
$$P(A \text{ and } B) = P(A) \times P(B)$$

**Example:** Flipping two fair coins, P(both heads)?
```
P(first coin heads) = 0.5
P(second coin heads) = 0.5

P(both heads) = 0.5 × 0.5 = 0.25
```

**When A and B are dependent:**
$$P(A \text{ and } B) = P(A) \times P(B|A)$$

---

## 2.5 Conditional Probability

**What is it?** The probability of A happening GIVEN that we already know B happened.

$$P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$$

**The vertical bar "|" means "given that"**

**Example:** Customer data

| | Churned | Stayed | Total |
|---|---|---|---|
| Gold | 10 | 90 | 100 |
| Silver | 30 | 70 | 100 |
| Bronze | 50 | 50 | 100 |
| **Total** | 90 | 210 | 300 |

**Question:** What's P(Churn | Gold)? (Probability of churning given the customer is Gold)

```
Step 1: Identify what we need
        - We only look at GOLD customers (that's our "given")
        - Among those, how many churned?

Step 2: Calculate
        Gold customers who churned: 10
        Total Gold customers: 100

        P(Churn | Gold) = 10/100 = 0.10 = 10%
```

**Question:** What's P(Gold | Churn)? (Given they churned, probability they were Gold)

```
Step 1: Identify what we need
        - We only look at customers who CHURNED
        - Among those, how many were Gold?

Step 2: Calculate
        Churned customers who were Gold: 10
        Total churned customers: 90

        P(Gold | Churn) = 10/90 = 0.111 = 11.1%
```

**Key Insight:** P(A|B) ≠ P(B|A) — they're different questions!

---

## 2.6 Bayes' Theorem

**What is it?** A formula to "flip" conditional probabilities. If you know P(B|A), you can find P(A|B).

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

**When to use it?** When you know the probability "in one direction" but need it "in the other direction."

**Classic Example: Medical Testing**

A disease affects 1% of the population. A test has:
- 99% accuracy for people WITH disease (detects it correctly)
- 95% accuracy for people WITHOUT disease (correctly says negative)

**Question:** If someone tests positive, what's the probability they actually have the disease?

```
Let D = has disease, T+ = tests positive

Given information:
- P(D) = 0.01 (1% have disease)
- P(not D) = 0.99 (99% don't have disease)
- P(T+ | D) = 0.99 (if you have disease, 99% chance test is positive)
- P(T+ | not D) = 0.05 (if you don't have disease, 5% false positive)

We want: P(D | T+)

Step 1: Find P(T+) using Law of Total Probability
        P(T+) = P(T+ | D) × P(D) + P(T+ | not D) × P(not D)
        P(T+) = (0.99 × 0.01) + (0.05 × 0.99)
        P(T+) = 0.0099 + 0.0495
        P(T+) = 0.0594

Step 2: Apply Bayes' Theorem
        P(D | T+) = P(T+ | D) × P(D) / P(T+)
        P(D | T+) = (0.99 × 0.01) / 0.0594
        P(D | T+) = 0.0099 / 0.0594
        P(D | T+) = 0.167 = 16.7%
```

**Surprising result:** Even with a 99% accurate test, a positive result only means 16.7% chance of having the disease! This is because the disease is rare (1%), so false positives outnumber true positives.

---

## 2.7 Python Code for Probability

```python
# ============================================================
# PROBABILITY CALCULATIONS IN PYTHON
# ============================================================

import pandas as pd

# ============================================================
# CREATING OUR DATA
# ============================================================
# Let's create the customer churn data from our theory section
# We'll use a DataFrame (like a table/spreadsheet in Python)

# pd.DataFrame() creates a table from a dictionary
# Keys become column names, values become column data
data = pd.DataFrame({
    'Segment': ['Gold', 'Silver', 'Bronze'],
    'Churned': [10, 30, 50],
    'Stayed': [90, 70, 50]
})

# Add a Total column by summing Churned and Stayed
data['Total'] = data['Churned'] + data['Stayed']

# Display the data
print("Customer Data:")
print(data)
print()

# ============================================================
# CALCULATING TOTALS
# ============================================================
# .sum() adds up all values in a column

total_customers = data['Total'].sum()        # 100 + 100 + 100 = 300
total_churned = data['Churned'].sum()        # 10 + 30 + 50 = 90
total_stayed = data['Stayed'].sum()          # 90 + 70 + 50 = 210

print(f"Total customers: {total_customers}")
print(f"Total churned: {total_churned}")
print(f"Total stayed: {total_stayed}")
print()

# ============================================================
# SIMPLE PROBABILITY
# ============================================================
# P(Churn) = Number who churned / Total customers

p_churn = total_churned / total_customers
# Calculation: 90 / 300 = 0.30

print(f"P(Churn) = {total_churned}/{total_customers} = {p_churn:.4f}")
print(f"Interpretation: 30% of all customers churn")
print()

# ============================================================
# CONDITIONAL PROBABILITY
# ============================================================
# P(Churn | Gold) = Gold customers who churned / All Gold customers

# First, get Gold row data using boolean indexing
# data['Segment'] == 'Gold' creates [True, False, False]
# data[...] filters to only True rows
gold_row = data[data['Segment'] == 'Gold']

gold_churned = gold_row['Churned'].values[0]   # Gets 10
gold_total = gold_row['Total'].values[0]        # Gets 100
# .values[0] extracts the first (only) value from the resulting array

p_churn_given_gold = gold_churned / gold_total
# Calculation: 10 / 100 = 0.10

print(f"P(Churn | Gold) = {gold_churned}/{gold_total} = {p_churn_given_gold:.4f}")
print(f"Interpretation: Among Gold customers, only 10% churn")
print()

# Let's compare all segments
print("Churn rate by segment:")
for segment in ['Gold', 'Silver', 'Bronze']:
    # Filter data to this segment
    row = data[data['Segment'] == segment]
    churned = row['Churned'].values[0]
    total = row['Total'].values[0]
    rate = churned / total
    print(f"  {segment}: {rate:.0%}")
    # :.0% formats decimal as percentage with 0 decimal places

# Output:
#   Gold: 10%
#   Silver: 30%
#   Bronze: 50%

print()

# ============================================================
# BAYES' THEOREM IN PYTHON
# ============================================================
# P(Gold | Churn) = P(Churn | Gold) × P(Gold) / P(Churn)

# We already calculated:
# P(Churn | Gold) = 0.10
# P(Churn) = 0.30

# Calculate P(Gold) = Gold customers / All customers
p_gold = gold_total / total_customers
# Calculation: 100 / 300 = 0.333

print("Bayes' Theorem Calculation:")
print(f"P(Gold) = {gold_total}/{total_customers} = {p_gold:.4f}")
print(f"P(Churn | Gold) = {p_churn_given_gold:.4f}")
print(f"P(Churn) = {p_churn:.4f}")
print()

# Apply Bayes' Theorem
p_gold_given_churn = (p_churn_given_gold * p_gold) / p_churn
# Calculation: (0.10 × 0.333) / 0.30 = 0.0333 / 0.30 = 0.111

print(f"P(Gold | Churn) = (P(Churn|Gold) × P(Gold)) / P(Churn)")
print(f"                = ({p_churn_given_gold:.4f} × {p_gold:.4f}) / {p_churn:.4f}")
print(f"                = {p_gold_given_churn:.4f}")
print()

# Verify by direct calculation
direct_calc = gold_churned / total_churned
# Among 90 churned, 10 were Gold = 10/90 = 0.111
print(f"Verification (direct): {gold_churned}/{total_churned} = {direct_calc:.4f}")
```

---

# Topic 3: Common Distributions

## What is a Probability Distribution?

A **probability distribution** describes all possible values a random variable can take and how likely each value is.

Think of it as an answer to: "If I repeat this experiment many times, what pattern will the results follow?"

---

## 3.1 The Normal Distribution (Bell Curve)

### Theory

The **Normal distribution** (also called Gaussian distribution) is the most important distribution in statistics. It's symmetric and bell-shaped.

**Why is it so important?**
1. Many natural phenomena follow it (heights, test scores, measurement errors)
2. The **Central Limit Theorem** says: averages of large samples follow a normal distribution, regardless of the original data's shape!

**Parameters:**
- **μ (mu):** Mean - the center of the bell curve
- **σ (sigma):** Standard deviation - how wide the bell is

**Notation:** $X \sim N(\mu, \sigma^2)$ means "X follows a normal distribution with mean μ and variance σ²"

### The 68-95-99.7 Rule (Empirical Rule)

For any normal distribution:
- **68%** of values fall within 1 standard deviation of the mean
- **95%** of values fall within 2 standard deviations
- **99.7%** of values fall within 3 standard deviations

```
         99.7% (within 3σ)
    |---------------------------|
         95% (within 2σ)
      |---------------------|
          68% (within 1σ)
        |---------------|
             _____
            /     \
           /       \
          /    |    \
    ____/     μ      \____
    |-3σ  -2σ  -1σ  +1σ  +2σ  +3σ|
```

### Z-Scores: Standardizing Values

A **z-score** tells you how many standard deviations a value is from the mean.

$$z = \frac{x - \mu}{\sigma}$$

- z = 0: value equals the mean
- z = 1: value is 1 standard deviation above mean
- z = -2: value is 2 standard deviations below mean

**Why use z-scores?**
- Compare values from different distributions
- Look up probabilities in standard normal tables

### Python Code for Normal Distribution

```python
# ============================================================
# NORMAL DISTRIBUTION IN PYTHON
# ============================================================

import numpy as np
from scipy import stats

# ============================================================
# UNDERSTANDING scipy.stats
# ============================================================
# scipy is a library for scientific computing
# stats module contains statistical functions and distributions
#
# For any distribution, scipy provides:
#   .pdf(x) = Probability Density Function - height of curve at x
#   .cdf(x) = Cumulative Distribution Function - P(X ≤ x)
#   .ppf(p) = Percent Point Function - inverse of CDF, returns x for given probability
#   .rvs(size=n) = Random Variates - generate n random samples

# ============================================================
# CREATING A NORMAL DISTRIBUTION
# ============================================================

# Customer spending follows Normal distribution
# Mean spending = $50, Standard deviation = $10
mean = 50
std = 10

# Create the distribution object
# loc = location parameter = mean
# scale = scale parameter = standard deviation
spending_dist = stats.norm(loc=mean, scale=std)

print("Distribution: X ~ Normal(μ=50, σ=10)")
print()

# ============================================================
# PROBABILITY CALCULATIONS WITH CDF
# ============================================================
# CDF = Cumulative Distribution Function
# cdf(x) gives P(X ≤ x) = "probability of getting x or less"

# Question 1: What percentage of customers spend less than $40?
x1 = 40
prob_less_40 = spending_dist.cdf(x1)

print(f"Q1: P(X < 40)?")
print(f"    Answer: {prob_less_40:.4f} = {prob_less_40*100:.2f}%")
print(f"    Meaning: About 16% of customers spend less than $40")
print()

# Question 2: What percentage spend MORE than $70?
# P(X > 70) = 1 - P(X ≤ 70)
x2 = 70
prob_more_70 = 1 - spending_dist.cdf(x2)

print(f"Q2: P(X > 70)?")
print(f"    P(X > 70) = 1 - P(X ≤ 70)")
print(f"             = 1 - {spending_dist.cdf(x2):.4f}")
print(f"             = {prob_more_70:.4f} = {prob_more_70*100:.2f}%")
print()

# Question 3: What percentage spend between $45 and $65?
# P(45 < X < 65) = P(X < 65) - P(X < 45)
x_low, x_high = 45, 65
prob_between = spending_dist.cdf(x_high) - spending_dist.cdf(x_low)

print(f"Q3: P(45 < X < 65)?")
print(f"    = P(X < 65) - P(X < 45)")
print(f"    = {spending_dist.cdf(x_high):.4f} - {spending_dist.cdf(x_low):.4f}")
print(f"    = {prob_between:.4f} = {prob_between*100:.2f}%")
print()

# ============================================================
# Z-SCORE CALCULATIONS
# ============================================================

# Z-score for spending $70:
x = 70
z = (x - mean) / std
# z = (70 - 50) / 10 = 20 / 10 = 2

print(f"Z-Score Calculation:")
print(f"    For x = ${x}:")
print(f"    z = (x - μ) / σ")
print(f"    z = ({x} - {mean}) / {std}")
print(f"    z = {z}")
print(f"    Meaning: $70 is 2 standard deviations above the mean")
print()

# ============================================================
# INVERSE: FINDING VALUES FROM PROBABILITIES (PPF)
# ============================================================
# PPF = Percent Point Function (inverse of CDF)
# ppf(p) answers: "What value has probability p below it?"

# Question: What is the 90th percentile of spending?
# (What value has 90% of customers below it?)
percentile = 0.90
spending_90 = spending_dist.ppf(percentile)

print(f"90th Percentile:")
print(f"    The value where 90% spend less: ${spending_90:.2f}")
print(f"    (Only 10% of customers spend more than ${spending_90:.2f})")
```

---

## 3.2 The Binomial Distribution

### Theory

The **Binomial distribution** counts the number of "successes" in a fixed number of independent trials, where each trial has only two outcomes (yes/no, success/failure, convert/don't convert).

**When to use Binomial:**
- Fixed number of trials (n)
- Each trial is independent
- Two possible outcomes per trial
- Same probability of success (p) for each trial

**Examples:**
- Out of 100 website visitors, how many will convert? (n=100, p=conversion rate)
- Out of 10 coin flips, how many heads? (n=10, p=0.5)

**Parameters:**
- **n:** Number of trials
- **p:** Probability of success on each trial

**Notation:** $X \sim Binomial(n, p)$

**Formulas:**
- **Expected value (mean):** $E[X] = n \times p$
- **Variance:** $Var(X) = n \times p \times (1-p)$

### Python Code for Binomial Distribution

```python
# ============================================================
# BINOMIAL DISTRIBUTION IN PYTHON
# ============================================================

from scipy import stats
from math import comb  # comb(n, k) calculates "n choose k"

# ============================================================
# SCENARIO
# ============================================================
# A store has a 20% conversion rate (20% of visitors buy something)
# If 10 customers enter, how many will convert?

n = 10       # Number of trials (customers)
p = 0.20     # Probability of success (conversion rate = 20%)

# Create binomial distribution
conversion_dist = stats.binom(n=n, p=p)

print(f"Distribution: X ~ Binomial(n={n}, p={p})")
print(f"This models: number of conversions out of {n} customers")
print()

# ============================================================
# EXPECTED VALUE AND STANDARD DEVIATION
# ============================================================

expected = n * p
variance = n * p * (1 - p)
std_dev = variance ** 0.5  # Square root for std dev

print("Expected Value and Spread:")
print(f"    E[X] = n × p = {n} × {p} = {expected}")
print(f"    Var(X) = n × p × (1-p) = {n} × {p} × {1-p} = {variance}")
print(f"    Std Dev = √{variance} = {std_dev:.2f}")
print(f"    Meaning: We expect about {expected} conversions, give or take {std_dev:.1f}")
print()

# ============================================================
# PROBABILITY OF EXACT VALUE (PMF)
# ============================================================
# PMF = Probability Mass Function
# For discrete distributions, pmf(k) = P(X = k)

# Question: What's the probability of EXACTLY 3 conversions?
k = 3
prob_exactly_3 = conversion_dist.pmf(k)

print(f"P(X = 3) = {prob_exactly_3:.4f} = {prob_exactly_3*100:.1f}%")
print(f"Meaning: There's about a 20% chance exactly 3 of 10 customers convert")
print()

# ============================================================
# MANUAL CALCULATION (to understand the formula)
# ============================================================
# Binomial formula: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
#
# where C(n,k) = "n choose k" = n! / (k! × (n-k)!)
# This counts the NUMBER OF WAYS to choose k successes from n trials

# For P(X = 3):
# C(10, 3) = number of ways to have exactly 3 conversions
# p^3 = probability those 3 convert
# (1-p)^7 = probability the other 7 don't convert

c_n_k = comb(n, k)           # C(10, 3) = 120
p_power_k = p ** k           # 0.20^3 = 0.008
q_power_rest = (1-p) ** (n-k)  # 0.80^7 = 0.2097

manual_prob = c_n_k * p_power_k * q_power_rest

print("Manual Calculation:")
print(f"    P(X = 3) = C({n},{k}) × p^{k} × (1-p)^{n-k}")
print(f"            = {c_n_k} × {p_power_k:.6f} × {q_power_rest:.6f}")
print(f"            = {manual_prob:.4f}")
print(f"    Matches scipy: {prob_exactly_3:.4f} ✓")
print()

# ============================================================
# CUMULATIVE PROBABILITIES (CDF)
# ============================================================
# cdf(k) = P(X ≤ k) = probability of k OR FEWER successes

# Question: What's the probability of 2 or fewer conversions?
prob_at_most_2 = conversion_dist.cdf(2)

print(f"P(X ≤ 2) = {prob_at_most_2:.4f} = {prob_at_most_2*100:.1f}%")
print(f"This equals: P(0) + P(1) + P(2)")
# Verify:
verify = conversion_dist.pmf(0) + conversion_dist.pmf(1) + conversion_dist.pmf(2)
print(f"Verification: {verify:.4f}")
print()

# Question: What's the probability of 4 OR MORE conversions?
# P(X ≥ 4) = 1 - P(X ≤ 3) = 1 - P(X < 4)
prob_at_least_4 = 1 - conversion_dist.cdf(3)

print(f"P(X ≥ 4) = 1 - P(X ≤ 3) = 1 - {conversion_dist.cdf(3):.4f} = {prob_at_least_4:.4f}")
```

---

# Topic 4: Hypothesis Testing

## What is Hypothesis Testing? (Theory Deep Dive)

**Hypothesis testing** is a formal framework for making decisions based on data. You have a question ("Is this new design better?"), collect data, and use statistics to answer it.

### The Logic of Hypothesis Testing

Think of it like a courtroom trial:
- **Null Hypothesis (H₀):** The defendant is innocent (nothing special happening)
- **Alternative Hypothesis (H₁):** The defendant is guilty (there IS an effect)
- **Evidence:** The data you collect
- **Verdict:** Either "reject H₀" (guilty) or "fail to reject H₀" (not proven guilty)

**Key insight:** We never "prove" H₀ is true. We only determine if there's enough evidence against it.

### Key Terminology

| Term | Definition | Analogy |
|------|------------|---------|
| **H₀ (Null)** | Default assumption: no effect, no difference | "Innocent until proven guilty" |
| **H₁ (Alternative)** | What we're trying to prove: there IS an effect | "The defendant is guilty" |
| **α (alpha)** | Significance level, typically 0.05 (5%) | How much "reasonable doubt" we allow |
| **p-value** | Probability of seeing our data if H₀ is true | How surprising is the evidence? |
| **Type I Error** | Rejecting H₀ when it's actually true | Convicting an innocent person |
| **Type II Error** | Failing to reject H₀ when H₁ is true | Letting a guilty person go free |
| **Power** | Probability of correctly rejecting a false H₀ | Ability to catch guilty people |

### The p-value Explained

**What it is:** The probability of observing data at least as extreme as yours, assuming H₀ is true.

**What it is NOT:** The probability that H₀ is true! (Common mistake!)

**Example:** 
- You flip a coin 20 times and get 15 heads.
- H₀: The coin is fair (P=0.5)
- p-value = probability of getting 15 or more heads (or 5 or fewer) with a fair coin
- p-value answers: "If the coin IS fair, how likely is this result?"
- If p-value is tiny (like 0.02), this result would be very surprising for a fair coin, so we doubt H₀.

### Decision Rule

```
If p-value < α (typically 0.05):
    Reject H₀
    Conclusion: Evidence suggests H₁ is true
    
If p-value ≥ α:
    Fail to reject H₀
    Conclusion: Not enough evidence to reject H₀
                (This doesn't mean H₀ is true!)
```

### One-tailed vs Two-tailed Tests

- **Two-tailed (most common):** Testing for "different from" (greater OR less)
  - H₁: μ ≠ μ₀
  - Example: "Is the new mean different from 50?"
  
- **One-tailed:** Testing for a specific direction
  - H₁: μ > μ₀ OR H₁: μ < μ₀
  - Example: "Is the new mean GREATER than 50?"

### The t-test Explained

The **t-test** compares means. It answers: "Is the difference in means statistically significant?"

**t-statistic formula (one-sample):**
$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

Where:
- $\bar{x}$ = sample mean
- $\mu_0$ = hypothesized mean (from H₀)
- $s$ = sample standard deviation
- $n$ = sample size
- $s / \sqrt{n}$ = **standard error** (how much sample mean varies)

**Interpretation:** t measures how many "standard errors" the sample mean is from the hypothesized mean.

- t ≈ 0: sample mean is close to hypothesized mean
- |t| large: sample mean is far from hypothesized mean (more evidence against H₀)

---

## 4.1 Python Code for Hypothesis Testing

```python
# ============================================================
# HYPOTHESIS TESTING IN PYTHON
# ============================================================

import numpy as np
from scipy import stats

# ============================================================
# SCENARIO: ONE-SAMPLE T-TEST
# ============================================================
# A store claims their average transaction is $50.
# We collect a sample of 25 transactions and want to test this claim.

# Set random seed for reproducibility
# (so you get the same "random" numbers every time)
np.random.seed(42)

# Generate sample data: actually comes from mean=53, std=12
# But we're pretending we don't know this - testing if mean = 50
sample = np.random.normal(loc=53, scale=12, size=25)

# ============================================================
# STEP 1: EXAMINE THE SAMPLE DATA
# ============================================================

sample_mean = sample.mean()
sample_std = sample.std(ddof=1)  # ddof=1 for sample std dev
n = len(sample)

print("=" * 50)
print("STEP 1: SAMPLE STATISTICS")
print("=" * 50)
print(f"Sample size (n): {n}")
print(f"Sample mean (x̄): {sample_mean:.2f}")
print(f"Sample std dev (s): {sample_std:.2f}")
print()

# ============================================================
# STEP 2: STATE THE HYPOTHESES
# ============================================================

hypothesized_mean = 50  # The value we're testing against

print("=" * 50)
print("STEP 2: STATE HYPOTHESES")
print("=" * 50)
print("H₀ (Null): μ = 50 (mean equals $50)")
print("H₁ (Alt):  μ ≠ 50 (mean is different from $50)")
print("This is a TWO-TAILED test (testing for 'different', not 'greater' or 'less')")
print(f"Significance level: α = 0.05")
print()

# ============================================================
# STEP 3: CALCULATE THE T-STATISTIC (manually)
# ============================================================

# Standard Error = How much the sample mean varies
# SE = s / √n
standard_error = sample_std / np.sqrt(n)

# t-statistic = How far is sample mean from hypothesized mean,
#               measured in units of standard error
t_statistic = (sample_mean - hypothesized_mean) / standard_error

# Degrees of freedom (for t-distribution)
df = n - 1

print("=" * 50)
print("STEP 3: CALCULATE T-STATISTIC")
print("=" * 50)
print(f"Standard Error (SE) = s / √n")
print(f"                    = {sample_std:.2f} / √{n}")
print(f"                    = {sample_std:.2f} / {np.sqrt(n):.2f}")
print(f"                    = {standard_error:.4f}")
print()
print(f"t-statistic = (x̄ - μ₀) / SE")
print(f"            = ({sample_mean:.2f} - {hypothesized_mean}) / {standard_error:.4f}")
print(f"            = {sample_mean - hypothesized_mean:.2f} / {standard_error:.4f}")
print(f"            = {t_statistic:.4f}")
print()
print(f"Degrees of freedom = n - 1 = {n} - 1 = {df}")
print()

# ============================================================
# STEP 4: CALCULATE THE P-VALUE
# ============================================================

# For two-tailed test:
# p-value = 2 × P(T > |t|)
# This accounts for extreme values in BOTH tails

# stats.t.cdf gives cumulative probability P(T ≤ t)
# We want P(T > |t|) = 1 - P(T ≤ |t|)
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

print("=" * 50)
print("STEP 4: CALCULATE P-VALUE")
print("=" * 50)
print(f"For two-tailed test:")
print(f"p-value = 2 × P(T > |t|)")
print(f"        = 2 × P(T > {abs(t_statistic):.4f})")
print(f"        = 2 × {1 - stats.t.cdf(abs(t_statistic), df):.4f}")
print(f"        = {p_value:.4f}")
print()

# ============================================================
# STEP 5: MAKE A DECISION
# ============================================================

alpha = 0.05

print("=" * 50)
print("STEP 5: MAKE A DECISION")
print("=" * 50)
print(f"Compare p-value ({p_value:.4f}) to α ({alpha})")
print()

if p_value < alpha:
    print(f"Since {p_value:.4f} < {alpha}:")
    print("→ REJECT H₀")
    print()
    print("Conclusion: The sample provides evidence that the true mean")
    print("            is DIFFERENT from $50 (statistically significant).")
else:
    print(f"Since {p_value:.4f} ≥ {alpha}:")
    print("→ FAIL TO REJECT H₀")
    print()
    print("Conclusion: The sample does NOT provide strong enough evidence")
    print("            to conclude the mean differs from $50.")

print()

# ============================================================
# VERIFICATION: USING SCIPY'S BUILT-IN FUNCTION
# ============================================================

# stats.ttest_1samp performs a one-sample t-test automatically
t_scipy, p_scipy = stats.ttest_1samp(sample, hypothesized_mean)

print("=" * 50)
print("VERIFICATION WITH scipy.stats.ttest_1samp()")
print("=" * 50)
print(f"t-statistic: {t_scipy:.4f} (manual: {t_statistic:.4f})")
print(f"p-value: {p_scipy:.4f} (manual: {p_value:.4f})")
```

---

# Topic 5: A/B Testing Framework

## What is A/B Testing? (Theory Deep Dive)

**A/B testing** (also called split testing) is an experiment where you compare two versions:
- **A (Control):** The current version
- **B (Treatment):** The new version you want to test

You randomly assign users to either A or B, measure outcomes, then use statistics to determine if B is better.

### The A/B Testing Process

```
1. HYPOTHESIS: "The new checkout button will increase conversion rate"

2. METRICS: Define what you'll measure (conversion rate)

3. SAMPLE SIZE: Calculate how many users you need

4. RANDOMIZE: Randomly assign users to A or B

5. RUN TEST: Let it run without peeking!

6. ANALYZE: Use statistical test (usually z-test for proportions)

7. DECIDE: If significant and meaningful, roll out B
```

### Sample Size Calculation

**Why it matters:** Too few users → can't detect real effects. Too many → waste of time/resources.

**Key inputs:**
- **Baseline rate:** Current conversion rate (e.g., 10%)
- **MDE (Minimum Detectable Effect):** Smallest improvement worth detecting (e.g., 2% absolute)
- **Significance level (α):** Usually 0.05
- **Power (1-β):** Usually 0.80 (80% chance of detecting a real effect)

---

## Quick Memorization List

1. **Mean vs Median:** Mean for symmetric data; Median for skewed/outliers
2. **Standard Deviation:** Typical distance from the mean
3. **Variance:** Average squared distance from mean (std² = variance)
4. **z-score:** z = (x - μ) / σ — how many std devs from mean
5. **p-value:** P(data this extreme | H₀ true) — NOT P(H₀ true | data)!
6. **Type I Error (α):** False positive — rejecting true H₀
7. **Type II Error (β):** False negative — failing to reject false H₀
8. **Power:** 1 - β — ability to detect true effect
9. **Bayes:** P(A|B) = P(B|A) × P(A) / P(B)
10. **Normal:** 68% within 1σ, 95% within 2σ, 99.7% within 3σ
11. **Binomial:** E[X] = n×p, Var = n×p×(1-p)
12. **Correlation ≠ Causation:** Need experiments to prove causation

---

## End-of-Day Mock Interview

### Question 1 (Short Answer): Mean vs Median
> "When would you use median instead of mean?"

**Model Answer:**
"I use median when data is skewed or has outliers. For example, customer lifetime value often has a few very high-value customers that inflate the mean. Median gives the 'typical' customer value that's more representative. In practice, I check if mean ≈ median; a big difference signals skewness."

### Question 2 (Calculation): Probability
> "60% of customers buy dairy. 40% of dairy buyers also buy bakery. What's P(Dairy AND Bakery)?"

**Model Answer:**
```
P(Dairy) = 0.60
P(Bakery | Dairy) = 0.40

P(Dairy AND Bakery) = P(Dairy) × P(Bakery | Dairy)
                    = 0.60 × 0.40
                    = 0.24 or 24%
```

### Question 3 (Coding): Hypothesis Test
> "Write code to test if sample mean differs significantly from 100."

**Model Answer:**
```python
from scipy import stats
sample = [98, 102, 105, 97, 101, 99, 103, 100, 96, 104]
t_stat, p_value = stats.ttest_1samp(sample, 100)
print(f"t = {t_stat:.3f}, p = {p_value:.3f}")
# If p < 0.05, reject H₀; otherwise, no significant difference
```

### Question 4 (Case): A/B Test Decision
> "Your A/B test shows Treatment beats Control with p=0.08. What do you recommend?"

**Model Answer:**
"At p=0.08 > α=0.05, we technically fail to reject H₀. But before concluding:
1. Check effect size—is the difference practically meaningful?
2. Consider extending the test for more data
3. If effect is small AND p=0.08, likely no meaningful impact
4. Business context matters: high-risk change → stick with significant results"

---

## 1-Page Cheat Sheet

```
=== STATISTICS CHEAT SHEET ===

CENTRAL TENDENCY
  Mean = sum/count (sensitive to outliers)
  Median = middle value (robust to outliers)
  Mode = most frequent value

SPREAD
  Variance = avg squared distance from mean
  Std Dev = √variance (same units as data)
  IQR = Q3 - Q1 (middle 50% spread)

PROBABILITY
  P(A or B) = P(A) + P(B) - P(A and B)
  P(A and B) = P(A) × P(B|A)
  P(A|B) = P(A and B) / P(B)
  Bayes: P(A|B) = P(B|A) × P(A) / P(B)

DISTRIBUTIONS
  Normal: X ~ N(μ, σ²)
    68-95-99.7 rule for 1-2-3 std devs
    z = (x - μ) / σ
  Binomial: X ~ Binom(n, p)
    E[X] = np, Var = np(1-p)

HYPOTHESIS TESTING
  H₀: null (no effect)
  H₁: alternative (there IS effect)
  p-value < α → reject H₀
  t = (x̄ - μ₀) / (s/√n)

SCIPY FUNCTIONS
  stats.norm(loc, scale) - normal dist
  stats.binom(n, p) - binomial dist
  .cdf(x) - P(X ≤ x)
  .ppf(p) - value at percentile p
  .pmf(k) - P(X = k) for discrete
  stats.ttest_1samp(data, value) - t-test
```

---

*Day 3 Complete. Statistics provides the language for data-driven decisions—understand these fundamentals deeply.*
