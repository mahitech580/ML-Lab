"""
questions.py

Local quiz question bank spanning Python, NumPy, Pandas, Statistics,
Machine Learning, and Data Science, at Beginner/Intermediate/Advanced
difficulty. No internet connection required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class QuizQuestion:
    category: str
    difficulty: str
    question: str
    options: list
    correct_index: int
    explanation: str = ""


QUESTIONS: list[QuizQuestion] = [
    QuizQuestion("Python", "Beginner", "What data type does `len([1, 2, 3])` return?",
                 ["str", "int", "list", "float"], 1, "len() always returns an integer count."),
    QuizQuestion("Python", "Beginner", "Which keyword defines a function in Python?",
                 ["func", "def", "lambda", "function"], 1, "'def' is used to define functions."),
    QuizQuestion("Python", "Intermediate", "What does a Python list comprehension `[x**2 for x in range(5)]` produce?",
                 ["[0,1,4,9,16]", "[1,4,9,16,25]", "[0,1,2,3,4]", "Error"], 0,
                 "It squares each value from 0 to 4."),
    QuizQuestion("Python", "Advanced", "What is the output of `id(a) == id(b)` when `a = [1,2]` and `b = [1,2]`?",
                 ["True", "False", "Error", "None"], 1,
                 "Separate list literals create distinct objects even with equal contents."),
    QuizQuestion("NumPy", "Beginner", "What does `np.array([1,2,3]).shape` return?",
                 ["(3,)", "(1,3)", "3", "[3]"], 0, "A 1D array of length 3 has shape (3,)."),
    QuizQuestion("NumPy", "Intermediate", "What does broadcasting allow in NumPy?",
                 ["Only equal-shaped arrays to be added",
                  "Operations between arrays of different but compatible shapes",
                  "Converting arrays to lists",
                  "Multi-threaded execution"], 1,
                 "Broadcasting lets NumPy operate on arrays of different shapes under certain rules."),
    QuizQuestion("NumPy", "Advanced", "What is the time complexity of vectorized NumPy operations "
                 "compared to Python loops for large arrays?",
                 ["Same", "Slower due to overhead", "Generally much faster", "Always O(n^2)"], 2,
                 "Vectorized NumPy operations run in optimized C loops, avoiding Python-level overhead."),
    QuizQuestion("Pandas", "Beginner", "Which method shows the first 5 rows of a DataFrame?",
                 [".head()", ".top()", ".first()", ".preview()"], 0, ".head() defaults to 5 rows."),
    QuizQuestion("Pandas", "Intermediate", "What does `df.groupby('col').mean()` compute?",
                 ["The mean of 'col' only", "The mean of all numeric columns per group in 'col'",
                  "A single overall mean", "An error"], 1,
                 "groupby splits data by unique values in 'col' then aggregates other columns."),
    QuizQuestion("Pandas", "Advanced", "What is the key difference between `.loc[]` and `.iloc[]`?",
                 [".loc uses labels, .iloc uses integer positions",
                  "They are identical",
                  ".loc is faster in all cases",
                  ".iloc only works on Series"], 0,
                 ".loc is label-based indexing; .iloc is purely positional."),
    QuizQuestion("Statistics", "Beginner", "What does the median represent?",
                 ["The most frequent value", "The middle value when sorted",
                  "The average value", "The range of values"], 1,
                 "The median is the middle value of a sorted dataset."),
    QuizQuestion("Statistics", "Intermediate", "What does a high positive skewness indicate?",
                 ["A symmetric distribution", "A long tail on the right side",
                  "A long tail on the left side", "No variance"], 1,
                 "Positive skew means the distribution's tail extends toward higher values."),
    QuizQuestion("Statistics", "Advanced", "What does a p-value of 0.03 in a hypothesis test typically suggest "
                 "at a 0.05 significance level?",
                 ["Strong evidence for the null hypothesis",
                  "Statistically significant evidence against the null hypothesis",
                  "The test failed", "Nothing, p-values are meaningless"], 1,
                 "A p-value below the significance threshold leads to rejecting the null hypothesis."),
    QuizQuestion("Machine Learning", "Beginner", "What type of problem predicts a category?",
                 ["Regression", "Classification", "Clustering", "Dimensionality reduction"], 1,
                 "Classification predicts discrete class labels."),
    QuizQuestion("Machine Learning", "Intermediate", "What is the purpose of a validation set?",
                 ["To train the final model", "To tune hyperparameters without touching the test set",
                  "To replace the test set entirely", "It has no purpose"], 1,
                 "Validation sets help tune models while keeping the test set for final unbiased evaluation."),
    QuizQuestion("Machine Learning", "Advanced", "Why can Random Forests reduce variance compared to a single "
                 "Decision Tree?",
                 ["They use less data", "Averaging many decorrelated trees reduces variance",
                  "They always have lower bias", "They eliminate the need for evaluation"], 1,
                 "Bagging averages predictions from many trees trained on bootstrapped samples, reducing variance."),
    QuizQuestion("Data Science", "Beginner", "What is EDA short for?",
                 ["Estimated Data Analysis", "Exploratory Data Analysis",
                  "Extracted Data Aggregation", "Extended Data Array"], 1,
                 "EDA stands for Exploratory Data Analysis."),
    QuizQuestion("Data Science", "Intermediate", "Why is train/test splitting important?",
                 ["To make training faster", "To estimate how a model generalizes to unseen data",
                  "To reduce dataset size permanently", "It is not important"], 1,
                 "Held-out test data estimates real-world generalization performance."),
    QuizQuestion("Data Science", "Advanced", "What is a common risk of imputing missing values with the mean "
                 "for a skewed distribution?",
                 ["It has no risk", "It can distort the distribution and understate variance",
                  "It always improves accuracy", "It removes all outliers"], 1,
                 "Mean imputation can distort skewed data and artificially reduce variance."),
]


def get_questions(
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    n: int = 10,
    shuffle: bool = True,
) -> list[QuizQuestion]:
    pool = QUESTIONS
    if categories:
        pool = [q for q in pool if q.category in categories]
    if difficulties:
        pool = [q for q in pool if q.difficulty in difficulties]
    pool = list(pool)
    if shuffle:
        random.shuffle(pool)
    return pool[:n] if n else pool


def all_categories() -> list[str]:
    return sorted({q.category for q in QUESTIONS})


def all_difficulties() -> list[str]:
    return ["Beginner", "Intermediate", "Advanced"]
