"""
lessons.py

Educational content for the ML Academy section. Each lesson includes a
simple explanation, a technical explanation, an example, when to use
it, common mistakes, and an interview question.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Lesson:
    title: str
    simple_explanation: str
    technical_explanation: str
    example: str
    when_to_use: str
    common_mistakes: str
    interview_question: str


LESSONS: list[Lesson] = [
    Lesson(
        title="Machine Learning Basics",
        simple_explanation="ML lets a computer learn patterns from examples instead of "
                            "being explicitly programmed with rules.",
        technical_explanation="An ML algorithm optimizes parameters of a model to minimize "
                               "a loss function computed over training data, aiming to "
                               "generalize to unseen data.",
        example="Learning to predict house prices from square footage and location.",
        when_to_use="When rules are too complex to hand-code but data with patterns is available.",
        common_mistakes="Assuming more data always fixes a bad model; ignoring data quality.",
        interview_question="What is the difference between AI, ML, and Deep Learning?",
    ),
    Lesson(
        title="Supervised Learning",
        simple_explanation="The model learns from labeled examples (input + correct answer).",
        technical_explanation="Supervised learning fits f(X) -> y by minimizing prediction "
                               "error against known labels y in the training set.",
        example="Predicting whether an email is spam using labeled spam/not-spam emails.",
        when_to_use="When you have historical data with known outcomes.",
        common_mistakes="Training and testing on the same data; leaking future information.",
        interview_question="Give an example of a supervised learning problem and the label it uses.",
    ),
    Lesson(
        title="Unsupervised Learning",
        simple_explanation="The model finds structure in data that has no labels.",
        technical_explanation="Unsupervised methods (e.g. clustering, dimensionality reduction) "
                               "discover latent structure by optimizing an internal criterion "
                               "such as within-cluster variance, without ground-truth labels.",
        example="Grouping customers into segments based on purchasing behavior.",
        when_to_use="When labels are unavailable or you want to explore data structure.",
        common_mistakes="Choosing the number of clusters arbitrarily without validation.",
        interview_question="How would you choose the number of clusters for K-Means?",
    ),
    Lesson(
        title="Classification",
        simple_explanation="Predicting which category something belongs to.",
        technical_explanation="Classification models estimate P(y=c|X) or a decision boundary "
                               "over discrete class labels c.",
        example="Predicting whether a customer will churn (Yes/No).",
        when_to_use="When the target variable is categorical.",
        common_mistakes="Using accuracy alone on imbalanced classes instead of F1/ROC-AUC.",
        interview_question="Why might accuracy be a misleading metric for imbalanced data?",
    ),
    Lesson(
        title="Regression",
        simple_explanation="Predicting a continuous numeric value.",
        technical_explanation="Regression models estimate E[y|X] by minimizing an error metric "
                               "such as mean squared error between predictions and true values.",
        example="Predicting house price from square footage, location, and age.",
        when_to_use="When the target variable is continuous and numeric.",
        common_mistakes="Not checking residuals for patterns indicating a poor model fit.",
        interview_question="What does R\u00b2 actually measure, and what are its limitations?",
    ),
    Lesson(
        title="Clustering",
        simple_explanation="Grouping similar data points together without predefined labels.",
        technical_explanation="Clustering algorithms partition data to maximize intra-cluster "
                               "similarity and inter-cluster separation, using distance or "
                               "density-based criteria.",
        example="Segmenting customers by spending habits into distinct groups.",
        when_to_use="Exploratory analysis, customer segmentation, anomaly detection.",
        common_mistakes="Not scaling features before distance-based clustering (e.g. K-Means).",
        interview_question="What is the difference between K-Means and DBSCAN?",
    ),
    Lesson(
        title="Feature Engineering",
        simple_explanation="Creating better input variables to help the model learn.",
        technical_explanation="Feature engineering transforms raw variables (via logs, "
                               "interactions, binning, ratios) to expose relationships the "
                               "raw representation obscures.",
        example="Creating a 'price per square foot' feature from price and area.",
        when_to_use="When domain knowledge suggests a transformation captures signal better.",
        common_mistakes="Creating features that leak the target variable's information.",
        interview_question="Give an example of an interaction feature and why it might help.",
    ),
    Lesson(
        title="Data Leakage",
        simple_explanation="When information from outside the training set improperly "
                            "influences model training, making results look better than they are.",
        technical_explanation="Leakage occurs when features encode target information not "
                               "available at prediction time, or when preprocessing is fit on "
                               "data that includes the test set.",
        example="Including 'cancellation_date' as a feature to predict 'will_cancel'.",
        when_to_use="N/A — this is a failure mode to detect and avoid, not a technique to use.",
        common_mistakes="Fitting scalers/encoders on the full dataset before splitting.",
        interview_question="How would you detect data leakage in a new dataset?",
    ),
    Lesson(
        title="Overfitting",
        simple_explanation="The model memorizes training data instead of learning general patterns.",
        technical_explanation="Overfitting occurs when a model's variance is too high relative "
                               "to the signal in the data, achieving low training error but high "
                               "test error.",
        example="A very deep decision tree with 100% training accuracy but poor test accuracy.",
        when_to_use="N/A — a problem to diagnose and mitigate (regularization, more data, "
                    "simpler models, cross-validation).",
        common_mistakes="Judging model quality using only training performance.",
        interview_question="Name three techniques to reduce overfitting.",
    ),
    Lesson(
        title="Underfitting",
        simple_explanation="The model is too simple to capture the real patterns in the data.",
        technical_explanation="Underfitting occurs when model bias is too high, resulting in "
                               "poor performance on both training and test data.",
        example="Fitting a straight line to clearly non-linear data.",
        when_to_use="N/A — a problem to diagnose; fix with more expressive models or features.",
        common_mistakes="Adding regularization so strong that the model can't learn signal at all.",
        interview_question="How can you tell if a model is underfitting rather than overfitting?",
    ),
    Lesson(
        title="Bias vs Variance",
        simple_explanation="Bias is being consistently wrong; variance is being inconsistent "
                            "across different training sets.",
        technical_explanation="Expected test error decomposes into bias\u00b2 + variance + "
                               "irreducible error; more flexible models trade higher variance "
                               "for lower bias.",
        example="A linear model has high bias/low variance; a deep tree has low bias/high variance.",
        when_to_use="Use this framework to reason about model complexity choices.",
        common_mistakes="Trying to reduce both bias and variance with the same lever "
                         "(e.g. only tuning one hyperparameter).",
        interview_question="Explain the bias-variance tradeoff with a concrete example.",
    ),
    Lesson(
        title="Cross Validation",
        simple_explanation="Testing a model on several different train/test splits to get a "
                            "more reliable performance estimate.",
        technical_explanation="K-fold cross-validation partitions data into k folds, trains on "
                               "k-1 folds and validates on the remaining fold, repeating k times "
                               "and averaging the score.",
        example="5-fold CV on the Iris dataset to estimate accuracy more robustly.",
        when_to_use="Whenever a single train/test split might give a noisy performance estimate.",
        common_mistakes="Fitting preprocessing outside the CV loop, leaking test-fold information.",
        interview_question="What is the difference between K-Fold and Stratified K-Fold?",
    ),
    Lesson(
        title="Regularization",
        simple_explanation="Penalizing model complexity to prevent overfitting.",
        technical_explanation="L1 (Lasso) and L2 (Ridge) regularization add a penalty term "
                               "proportional to the magnitude of model coefficients to the loss "
                               "function, shrinking or zeroing them.",
        example="Ridge regression shrinking coefficients of correlated features.",
        when_to_use="When a model overfits or when many correlated features are present.",
        common_mistakes="Not scaling features before applying L1/L2 penalties.",
        interview_question="What's the practical difference between L1 and L2 regularization?",
    ),
    Lesson(
        title="Evaluation Metrics",
        simple_explanation="Numbers that tell you how well a model is performing.",
        technical_explanation="Different metrics (accuracy, precision, recall, F1, ROC-AUC, "
                               "MAE, RMSE, R\u00b2) capture different aspects of error and must be "
                               "chosen based on the problem and its costs.",
        example="Using recall over precision when missing a fraud case is costlier than a false alarm.",
        when_to_use="Always — pick the metric that matches the real-world cost of errors.",
        common_mistakes="Reporting only accuracy on an imbalanced classification problem.",
        interview_question="When would you prioritize recall over precision, and why?",
    ),
    Lesson(
        title="Ensemble Learning",
        simple_explanation="Combining multiple models to get better predictions than any single one.",
        technical_explanation="Bagging (e.g. Random Forest) reduces variance by averaging "
                               "independently trained models; boosting (e.g. Gradient Boosting) "
                               "reduces bias by sequentially correcting prior errors.",
        example="A Random Forest averaging 200 decision trees.",
        when_to_use="When a single model's variance or bias limits performance.",
        common_mistakes="Assuming ensembles always outperform simpler models regardless of data size.",
        interview_question="What's the difference between bagging and boosting?",
    ),
]


def get_lesson(title: str) -> Lesson | None:
    for lesson in LESSONS:
        if lesson.title.lower() == title.lower():
            return lesson
    return None
