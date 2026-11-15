import argparse
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model" / "text_model.joblib"


class EvaluationMetrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    classification_report: str


def _load_training_data(data_path: Path) -> tuple[pd.Series, pd.Series]:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Training data must include columns: {', '.join(sorted(missing_columns))}"
        )
    if df.empty:
        raise ValueError("Training data must contain at least one row")

    text = df["text"].fillna("").astype(str).str.strip()
    labels = df["label"].fillna("").astype(str).str.strip()
    if text.eq("").any() or labels.eq("").any():
        raise ValueError("Training data cannot contain empty text or labels")
    if labels.nunique() < 2:
        raise ValueError("Training data must contain at least two distinct labels")

    return text, labels


def _create_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                stop_words="english",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(C=1)),
        ]
    )


def evaluate_model(
    data_path: Path = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> EvaluationMetrics:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    text, labels = _load_training_data(data_path)
    train_text, test_text, train_labels, test_labels = train_test_split(
        text,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = _create_model()
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)
    class_labels = sorted(labels.unique().tolist())

    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "precision_macro": float(
            precision_score(
                test_labels, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                test_labels, predictions, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                test_labels, predictions, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                test_labels, predictions, average="weighted", zero_division=0
            )
        ),
        "labels": class_labels,
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=class_labels
        ).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=class_labels,
            target_names=class_labels,
            zero_division=0,
        ),
    }


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    text, labels = _load_training_data(data_path)
    model_path = Path(model_path)
    model = _create_model()
    model.fit(text, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def predict_text(
    text: str,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> str:
    if not text.strip():
        raise ValueError("Prediction text cannot be blank")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}; train it before requesting predictions"
        )
    model = joblib.load(model_path)
    return str(model.predict([text.strip()])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the text classifier")
    parser.add_argument(
        "--predict",
        metavar="TEXT",
        help="predict the sentiment of TEXT after training the model",
    )
    args = parser.parse_args()

    metrics = evaluate_model()
    model = train_model()
    print("Holdout metrics (20% stratified test split):")
    print(f"Accuracy:          {metrics['accuracy']:.2%}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Macro precision:   {metrics['precision_macro']:.2%}")
    print(f"Macro recall:      {metrics['recall_macro']:.2%}")
    print(f"Macro F1:          {metrics['f1_macro']:.2%}")
    print(f"Weighted F1:       {metrics['f1_weighted']:.2%}")
    print("\nPer-class report:")
    print(metrics["classification_report"])
    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["labels"],
            columns=metrics["labels"],
        )
    )
    print(f"Text model saved to {DEFAULT_MODEL_PATH}. Classes: {model.classes_}")
    if args.predict is not None:
        print(f"Prediction: {predict_text(args.predict)}")


if __name__ == "__main__":
    main()
