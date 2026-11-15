from pathlib import Path
from typing import Any, TypedDict

import numpy as np


MODEL_ID = "Horizon-Labs/multilingual-sentiment-small"
MODEL_REVISION = "e60eb86fd9e5bdd8c097bf6d6e75dade85a1ccac"
MAX_TOKEN_LENGTH = 512
PROJECT_MODEL_DIR = (
    Path(__file__).resolve().parent / "model" / "hf-sentiment-small"
)
MODEL_FILE = Path("onnx") / "model_quantized.onnx"


class SentimentPrediction(TypedDict):
    prediction: str
    score: float


class MultilingualSentimentModel:
    def __init__(
        self,
        tokenizer: Any,
        session: Any,
        labels: dict[int, str],
    ) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.labels = labels

    def predict(self, text: str) -> SentimentPrediction:
        if not text.strip():
            raise ValueError("Prediction text cannot be blank")

        encoded = self.tokenizer(
            [text.strip()],
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="np",
        )
        input_names = [item.name for item in self.session.get_inputs()]
        missing_inputs = [name for name in input_names if name not in encoded]
        if missing_inputs:
            raise RuntimeError(
                f"Tokenizer did not provide model inputs: {', '.join(missing_inputs)}"
            )
        model_inputs = {
            name: np.asarray(encoded[name], dtype=np.int64) for name in input_names
        }
        outputs = self.session.run(None, model_inputs)
        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Sentiment model returned an unexpected output shape")
        if logits.shape[1] != len(self.labels):
            raise RuntimeError("Sentiment model output does not match its label config")

        scores = np.exp(logits[0] - np.max(logits[0]))
        scores /= scores.sum()
        label_index = int(np.argmax(scores))
        return {
            "prediction": self.labels[label_index],
            "score": float(scores[label_index]),
        }


def load_sentiment_model(
    model_dir: Path = PROJECT_MODEL_DIR,
) -> MultilingualSentimentModel:
    import onnxruntime
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILE
    if not model_path.is_file():
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=model_dir,
            allow_patterns=[
                "config.json",
                "onnx/model_quantized.onnx",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    labels = {
        int(index): label.lower()
        for index, label in config.id2label.items()
    }
    if not labels or set(labels) != set(range(len(labels))):
        raise ValueError("Hugging Face model has an invalid sentiment label config")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return MultilingualSentimentModel(tokenizer, session, labels)
