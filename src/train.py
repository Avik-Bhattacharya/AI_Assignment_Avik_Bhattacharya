"""
Customer Support Ticket Classification System
Model Training Pipeline (train.py)

Preprocesses data, compares candidate classifiers using cross-validation on
training data only, evaluates the selected pipeline on an untouched test set,
and saves the trained model and confusion matrix.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib
# Use Agg backend for headless chart generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
CONFUSION_MATRIX_PATH = SCREENSHOTS_DIR / "confusion_matrix.png"


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the customer support tickets dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{file_path}'.")

    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    print(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    required_cols = {"ticket_description", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")

    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Preprocess records, then create a stratified held-out test split."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING & TRAIN-TEST SPLIT")
    print("=" * 70)

    # Cleaning, missing-value handling, and deduplication are deterministic
    # data preparation steps. All learned TF-IDF/model fitting happens later
    # using training data only.
    cleaned_df = preprocess_dataset(df, verbose=False)
    X = cleaned_df["ticket_description"]
    y = cleaned_df["category"]

    print(f"Prepared {len(cleaned_df)} records across {y.nunique()} categories.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set records : {len(X_train)} ({len(X_train) / len(cleaned_df) * 100:.1f}%)")
    print(f"Testing set records  : {len(X_test)} ({len(X_test) / len(cleaned_df) * 100:.1f}%)")
    print("\nTarget class distribution in Training Set:")
    for cat, count in y_train.value_counts().items():
        print(f"  - {cat:<20}: {count} records")
    print("\nTarget class distribution in Testing Set:")
    for cat, count in y_test.value_counts().items():
        print(f"  - {cat:<20}: {count} records")

    return X_train, X_test, y_train, y_test, cleaned_df

def build_pipeline(model_name: str = "LinearSVC", random_state: int = 42) -> Pipeline:
    """Build a TF-IDF pipeline for one candidate classifier."""
    print("\n" + "=" * 70)
    print("STEP 4: BUILDING SELECTED CLASSIFIER PIPELINE")
    print("=" * 70)
    print(f"Selected classifier: {model_name}")

    if model_name == "LinearSVC":
        classifier = LinearSVC(random_state=random_state, dual="auto")
    elif model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=2000, random_state=random_state)
    elif model_name == "Multinomial Naive Bayes":
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), lowercase=True)),
        ("classifier", classifier)
    ])


def compare_models_with_cross_validation(X_train, y_train, random_state: int = 42) -> str:
    """Compare candidates using training data only; the test set stays untouched."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL COMPARISON (5-FOLD CV ON TRAINING DATA ONLY)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    candidates = ["LinearSVC", "Logistic Regression", "Multinomial Naive Bayes"]
    results = []

    for name in candidates:
        scores = cross_validate(
            build_pipeline(name, random_state), X_train, y_train,
            cv=cv, scoring={"accuracy": "accuracy", "weighted_precision": "precision_weighted",
                     "weighted_recall": "recall_weighted", "weighted_f1": "f1_weighted"},
            n_jobs=1, return_train_score=False
        )
        mean_accuracy = scores["test_accuracy"].mean()
        mean_precision = scores["test_weighted_precision"].mean()
        mean_recall = scores["test_weighted_recall"].mean()
        mean_f1 = scores["test_weighted_f1"].mean()
        results.append((name, mean_accuracy, mean_precision, mean_recall, mean_f1))
        print(
            f"{name:<24} Accuracy: {mean_accuracy:.4f} ± {scores['test_accuracy'].std():.4f} | "
            f"Precision: {mean_precision:.4f} ± {scores['test_weighted_precision'].std():.4f} | "
            f"Recall: {mean_recall:.4f} ± {scores['test_weighted_recall'].std():.4f} | "
            f"Weighted F1: {mean_f1:.4f} ± {scores['test_weighted_f1'].std():.4f}"
        )

    # Weighted F1 is the primary selection metric; accuracy breaks ties.
    selected = max(results, key=lambda item: (item[4], item[1]))[0]
    print(f"\nSelected model based on training-only CV weighted F1: {selected}")
    return selected

def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Fit the selected classifier pipeline on training data only."""
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING THE MODEL PIPELINE")
    print("=" * 70)
    print(f"Fitting TF-IDF Vectorizer and {pipeline.named_steps['classifier'].__class__.__name__} on X_train...")

    pipeline.fit(X_train, y_train)

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print("Training completed successfully!")
    print(f"Learned TF-IDF Vocabulary Size: {vocab_size} unique n-grams.")

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> tuple:
    """Evaluate pipeline on held-out test data using accuracy, precision, recall, and F1."""
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATING MODEL ON HELD-OUT TEST DATA")
    print("=" * 70)
    print(f"Model Evaluated: {pipeline.named_steps['classifier'].__class__.__name__}")

    # The selected classifier outputs discrete class predictions.
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    classes = pipeline.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} (Weighted)")
    print(f"Recall    : {recall:.4f} (Weighted)")
    print(f"F1-Score  : {f1:.4f} (Weighted)")

    print("\nDetailed Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print("-" * 70)
    cm_df = pd.DataFrame(cm, index=[f"Actual: {c}" for c in classes], columns=[f"Pred: {c}" for c in classes])
    print(cm_df.to_string())

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return metrics, y_pred, cm, classes


def analyze_misclassifications(
    df: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.Series,
    y_pred,
    output_path: Path = None
) -> pd.DataFrame:
    """Identify misclassified test tickets and print details to console."""
    print("\n" + "=" * 70)
    print("STEP 7: DETAILED MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    test_eval_df = df.loc[X_test.index].copy()
    test_eval_df["actual_category"] = y_test.values
    test_eval_df["predicted_category"] = y_pred

    is_incorrect = test_eval_df["actual_category"] != test_eval_df["predicted_category"]
    misclassified_df = test_eval_df[is_incorrect].copy()

    total_test = len(test_eval_df)
    total_incorrect = len(misclassified_df)
    total_correct = total_test - total_incorrect

    print(f"Total Test Records Evaluated : {total_test}")
    print(f"Total Correct Predictions     : {total_correct} ({(total_correct / total_test) * 100:.2f}%)")
    print(f"Total Incorrect Predictions   : {total_incorrect} ({(total_incorrect / total_test) * 100:.2f}%)")

    all_categories = sorted(df["category"].unique())
    misclass_counts = misclassified_df["actual_category"].value_counts().to_dict()

    print("\n" + "-" * 70)
    print("Total Incorrect Predictions by Actual Category:")
    print("-" * 70)
    for cat in all_categories:
        count = misclass_counts.get(cat, 0)
        print(f"  - {cat:<20}: {count} incorrect prediction(s)")

    print("\n" + "-" * 70)
    print("Detailed Misclassified Tickets (Grouped by Actual Category):")
    print("-" * 70)

    if total_incorrect == 0:
        print("Congratulations! Zero misclassifications found on the test set.")
    else:
        grouped = misclassified_df.groupby("actual_category")
        item_counter = 1

        for actual_cat, group in grouped:
            print(f"\n[Actual Category: {actual_cat}] ({len(group)} incorrect)")
            print("~" * 60)
            for _, row in group.iterrows():
                ticket_id_str = f"[{row['ticket_id']}] " if "ticket_id" in row else ""
                print(f"#{item_counter} {ticket_id_str}")
                print(f"  Description: \"{row['ticket_description']}\"")
                print(f"  Actual     : {row['actual_category']}")
                print(f"  Predicted  : {row['predicted_category']}")
                print("-" * 60)
                item_counter += 1

    if output_path is not None:
        export_cols = ["ticket_id", "ticket_description", "actual_category", "predicted_category"]
        export_cols = [c for c in export_cols if c in misclassified_df.columns]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        misclassified_df[export_cols].to_csv(output_path, index=False)
        print(f"\nMisclassification details saved to: {output_path}")

    return misclassified_df


def plot_confusion_matrix(cm, classes, output_path: Path) -> None:
    """Plot and save confusion matrix heatmap using Seaborn."""
    print("\n" + "=" * 70)
    print("STEP 8: GENERATING CONFUSION MATRIX HEATMAP")
    print("=" * 70)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=1,
        linecolor="gray",
        annot_kws={"size": 12, "weight": "bold"}
    )

    plt.title("Confusion Matrix - Selected Classifier Ticket Classification", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Actual Category", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Confusion matrix heatmap successfully saved to:")
    print(f"  -> {output_path}")


def save_model(pipeline: Pipeline, output_path: Path) -> None:
    """Serialize and save the trained pipeline using joblib."""
    print("\n" + "=" * 70)
    print("STEP 9: SAVING TRAINED MODEL PIPELINE")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    print("Model successfully saved to:")
    print(f"  -> {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    print("Customer Support Ticket Classification System")
    print("Model Training Pipeline Started...\n")

    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, cleaned_df = prepare_data(df, test_size=0.20, random_state=42)
    selected_model = compare_models_with_cross_validation(X_train, y_train, random_state=42)
    pipeline = build_pipeline(selected_model, random_state=42)
    pipeline = train_model(pipeline, X_train, y_train)
    metrics, y_pred, cm, classes = evaluate_model(pipeline, X_test, y_test)
    misclassified_df = analyze_misclassifications(cleaned_df, X_test, y_test, y_pred, output_path=None)
    plot_confusion_matrix(cm, classes, CONFUSION_MATRIX_PATH)
    save_model(pipeline, MODEL_PATH)

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Algorithm Selected          : {selected_model}")
    print(f"Dataset Used                : {DATA_PATH.name} ({len(df)} records)")
    print(f"Training Records            : {len(X_train)}")
    print(f"Testing Records             : {len(X_test)}")
    print(f"Final Test Accuracy         : {metrics['accuracy'] * 100:.2f}%")
    print(f"Final Weighted Precision    : {metrics['precision']:.4f}")
    print(f"Final Weighted Recall       : {metrics['recall']:.4f}")
    print(f"Final Weighted F1-Score     : {metrics['f1']:.4f}")
    print(f"Total Misclassified Tickets : {len(misclassified_df)} ({(len(misclassified_df) / len(X_test)) * 100:.2f}%)")
    print(f"Saved Model Location        : {MODEL_PATH}")
    print(f"Confusion Matrix Plot       : {CONFUSION_MATRIX_PATH}")
    print("=" * 70)
    print(f"{selected_model} model training script completed successfully!")


if __name__ == "__main__":
    main()
