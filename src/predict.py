"""
Customer Support Ticket Classification System
Inference & Prediction Service (predict.py)

Loads the trained LinearSVC pipeline from model/trained_model.joblib
to classify ticket descriptions programmatically or via an interactive CLI.
"""

from pathlib import Path
import re
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"

_CACHED_PIPELINE = None


def load_trained_model(model_path: Path = MODEL_PATH):
    """Load and cache the trained LinearSVC pipeline from disk."""
    global _CACHED_PIPELINE

    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by executing 'python src/train.py'."
        )

    try:
        _CACHED_PIPELINE = joblib.load(model_path)
        return _CACHED_PIPELINE
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact from '{model_path}': {e}")


def predict_ticket(ticket_description: str, pipeline=None) -> str:
    """Validate ticket text and predict its category using the trained pipeline."""
    if ticket_description is None:
        raise ValueError("Ticket description cannot be None.")

    if not isinstance(ticket_description, str):
        raise ValueError(f"Expected a string for ticket_description, got {type(ticket_description).__name__}.")

    cleaned_input = ticket_description.strip()
    if not cleaned_input:
        raise ValueError("Ticket description cannot be empty or contain only whitespace.")

    # Normalize text consistent with training (lowercase and strip punctuation)
    normalized_text = cleaned_input.lower()
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    if not normalized_text:
        raise ValueError("Ticket description contains no valid alphanumeric words to classify.")

    if pipeline is None:
        pipeline = load_trained_model(MODEL_PATH)

    # LinearSVC outputs discrete class predictions (no predict_proba)
    prediction = pipeline.predict([normalized_text])[0]
    return prediction


def run_cli() -> None:
    """Run an interactive CLI allowing repeated predictions with an exit option."""
    print("=" * 65)
    print("CUSTOMER SUPPORT TICKET CLASSIFICATION SYSTEM")
    print("               Inference Console")
    print("=" * 65)

    try:
        pipeline = load_trained_model(MODEL_PATH)
        print(f"[STATUS] Model loaded successfully from: {MODEL_PATH.name}")
        print(f"[STATUS] Categories: {', '.join(pipeline.classes_)}")
    except Exception as err:
        print(f"\n[FATAL ERROR] {err}")
        sys.exit(1)

    print("\nInstructions:")
    print("  - Type or paste a customer support ticket description and press Enter.")
    print("  - To terminate the program, type 'exit', 'quit', or 'q'.\n")

    while True:
        try:
            print("-" * 65)
            raw_input = input("Enter Ticket Description > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Goodbye!")
            break

        if raw_input.lower() in {"exit", "quit", "q"}:
            print("\nExiting Customer Support Ticket Classifier. Goodbye!")
            break

        if not raw_input:
            print("[Warning] Input cannot be empty. Please enter a valid ticket description.")
            continue

        try:
            predicted_category = predict_ticket(raw_input, pipeline=pipeline)

            print("\n--- Prediction Result ---")
            print(f"Entered Description : \"{raw_input}\"")
            print(f"Predicted Category  : {predicted_category}")
            print("-------------------------")

        except ValueError as val_err:
            print(f"[Input Error] {val_err}")
        except Exception as runtime_err:
            print(f"[Error] An unexpected error occurred during prediction: {runtime_err}")


if __name__ == "__main__":
    run_cli()
