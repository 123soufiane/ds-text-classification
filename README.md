# Text classification

A sentiment-classification project with an English TF-IDF/linear-SVM training
pipeline and a multilingual Hugging Face prediction API.
The classifier combines word and character TF-IDF features with a linear SVM
(`LinearSVC`) to recognize both phrase context and variations in word forms.

## Training data

`data.csv` contains 3,000 real, labelled review sentences from the UCI
Sentiment Labelled Sentences dataset: 1,000 each from IMDb, Amazon, and Yelp,
with 500 positive and 500 negative examples per source. The `source` column
records each sentence's origin.
The CSV is kept out of Git; GitHub Actions downloads and validates the UCI
archive with `prepare_data.py` before training. Running `text_classify.py`
directly also downloads the dataset if `data.csv` is missing.

The dataset is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: Dimitrios Kotzias, “Sentiment Labelled Sentences,” UCI Machine
Learning Repository, 2015, [doi:10.24432/C57604](https://doi.org/10.24432/C57604).
The dataset accompanies Kotzias et al., “From Group to Individual Labels
Using Deep Features,” KDD 2015.

## Setup

Use Python 3.11 or newer, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The training CSV must contain non-empty `text` and `label` columns and at least
two distinct labels. Additional columns, such as `source`, are allowed.

## Train

From any working directory, run:

```bash
python /path/to/project/text_classify.py
```

This trains and evaluates the project's English TF-IDF/linear-SVM model. It
prints accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, a
per-class report, and the confusion matrix for a deterministic, stratified 20%
holdout split. The holdout is used for evaluation, not training. The script
then trains and saves the final model using all rows to
`model/text_model.joblib`. The FastAPI app uses the separate pretrained
multilingual model described above.

On the current 3,000-row dataset, this custom English model scored 83.50% accuracy on the
fixed holdout (501 of 600 predictions correct), compared with 80.00% for the
previous word-only logistic regression baseline. Results can vary with the
dataset and split; use the same split when comparing future changes.
To also predict a label for your own text:

```bash
python /path/to/project/text_classify.py --predict "The movie was wonderful"
```

`train.py` remains available as an alias for the same commands.

## Multilingual user predictions

The API uses the
[Horizon Labs multilingual sentiment small model](https://huggingface.co/Horizon-Labs/multilingual-sentiment-small)
directly for English, Arabic, and French input. It returns `positive`,
`neutral`, or `negative` and a model score. The model is Apache-2.0 licensed.
Its quantized ONNX weights are about 256 MiB (268 MB), about half the previous
model's size. The first API startup downloads the model and tokenizer into
`model/hf-sentiment-small/` inside the project. This local copy is retained
between restarts and excluded from Git. After download, predictions use only
the local files; user text is processed locally and is not sent to Hugging
Face. It supports many languages, including English, Arabic, and French; it
was evaluated across multilingual review and sentiment benchmarks, but not
specifically fine-tuned on this project's review dataset.

Start the API from the project directory:

```bash
uvicorn deploy:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the browser app, where
users can enter review text and see its predicted sentiment. Use `GET /health`
for a health check and `POST /predict` with JSON such as
`{"text": "A wonderful movie"}` for direct API access. Submitted text is not
stored by this app.

## Test

```bash
python -m pip install pytest
python -m pytest
```

GitHub Actions trains the model and runs the tests. It does not publish or
deploy a server; configure a hosting provider separately if deployment is
needed.
