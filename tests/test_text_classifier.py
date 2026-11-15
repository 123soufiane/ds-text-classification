import io
import zipfile

import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from deploy import create_app
from hf_sentiment import MultilingualSentimentModel, SentimentPrediction
from prepare_data import parse_dataset
from text_classify import evaluate_model, predict_text, train_model


class FakeSentimentModel:
    def predict(self, text: str) -> SentimentPrediction:
        return {"prediction": "neutral", "score": 0.72}


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
        }


class FakeSession:
    def get_inputs(self):
        return [type("Input", (), {"name": "input_ids"})()]

    def run(self, _output_names, inputs):
        assert inputs["input_ids"].tolist() == [[1, 2]]
        return [np.array([[2.0, 0.1, -1.0]], dtype=np.float32)]


def write_training_data(path, rows):
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)


def test_uci_dataset_parser_reads_review_source_and_label():
    files = {
        "sentiment labelled sentences/imdb_labelled.txt": (
            "Great movie\t1\nTerrible plot\t0\n"
        ),
        "sentiment labelled sentences/amazon_cells_labelled.txt": (
            "Loved this phone\t1\nBad battery\t0\n"
        ),
        "sentiment labelled sentences/yelp_labelled.txt": (
            "Friendly staff\t1\nLong wait\t0\n"
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in files.items():
            archive.writestr(filename, contents)

    rows = parse_dataset(buffer.getvalue())

    assert len(rows) == 6
    assert rows[0] == ("Great movie", "positive", "imdb")
    assert rows[-1] == ("Long wait", "negative", "yelp")


def test_training_saves_a_text_classifier(tmp_path):
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model" / "text_model.joblib"
    write_training_data(
        data_path,
        [
            ("wonderful movie", "positive"),
            ("great film", "positive"),
            ("awful movie", "negative"),
            ("terrible film", "negative"),
        ],
    )

    model = train_model(data_path, model_path)

    assert model_path.is_file()
    assert model.predict(["wonderful film"])[0] == "positive"


def test_multilingual_model_returns_configured_label_and_probability():
    model = MultilingualSentimentModel(
        FakeTokenizer(),
        FakeSession(),
        {0: "negative", 1: "neutral", 2: "positive"},
    )

    result = model.predict("un film excellent")

    assert result["prediction"] == "negative"
    assert 0.0 < result["score"] < 1.0


def test_evaluation_and_text_prediction(tmp_path):
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "text_model.joblib"
    write_training_data(
        data_path,
        [
            ("wonderful movie", "positive"),
            ("amazing film", "positive"),
            ("great story", "positive"),
            ("excellent acting", "positive"),
            ("delightful plot", "positive"),
            ("terrible movie", "negative"),
            ("awful film", "negative"),
            ("dreadful story", "negative"),
            ("boring acting", "negative"),
            ("horrible plot", "negative"),
        ],
    )

    metrics = evaluate_model(data_path)
    train_model(data_path, model_path)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0
    assert metrics["labels"] == ["negative", "positive"]
    assert sum(map(sum, metrics["confusion_matrix"])) == 2
    assert "macro avg" in metrics["classification_report"]
    assert predict_text("wonderful movie", model_path) == "positive"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [("first text", "positive"), ("second text", "positive")],
            "at least two distinct labels",
        ),
        ([("", "positive"), ("bad", "negative")], "empty text or labels"),
    ],
)
def test_training_rejects_invalid_data(tmp_path, rows, message):
    data_path = tmp_path / "data.csv"
    write_training_data(data_path, rows)

    with pytest.raises(ValueError, match=message):
        train_model(data_path, tmp_path / "model.joblib")


def test_prediction_api_returns_label_and_rejects_blank_text(tmp_path):
    with TestClient(create_app(FakeSentimentModel)) as client:
        home_response = client.get("/")
        response = client.post("/predict", json={"text": "wonderful film"})
        blank_response = client.post("/predict", json={"text": "   "})
        too_long_response = client.post("/predict", json={"text": "x" * 10001})

    assert home_response.status_code == 200
    assert "How does this review feel?" in home_response.text
    assert response.status_code == 200
    assert response.json() == {"prediction": "neutral", "score": 0.72}
    assert blank_response.status_code == 422
    assert too_long_response.status_code == 422
