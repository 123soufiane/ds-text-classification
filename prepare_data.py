import csv
import io
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)
SOURCE_FILES = {
    "sentiment labelled sentences/imdb_labelled.txt": "imdb",
    "sentiment labelled sentences/amazon_cells_labelled.txt": "amazon",
    "sentiment labelled sentences/yelp_labelled.txt": "yelp",
}
LABEL_NAMES = {"0": "negative", "1": "positive"}
EXPECTED_ROWS_PER_SOURCE = 1000
EXPECTED_ROWS_PER_LABEL = 500


def parse_dataset(archive_bytes: bytes) -> list[tuple[str, str, str]]:
    rows = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for filename, source in SOURCE_FILES.items():
                contents = archive.read(filename).decode("utf-8-sig")
                for line_number, line in enumerate(contents.split("\n"), start=1):
                    line = line.rstrip("\r")
                    if not line.strip():
                        continue
                    try:
                        text, score = line.rsplit("\t", 1)
                    except ValueError as exc:
                        raise ValueError(
                            f"Unexpected row format in {filename}:{line_number}"
                        ) from exc
                    if not text.strip() or score not in LABEL_NAMES:
                        raise ValueError(
                            f"Invalid review or label in {filename}:{line_number}"
                        )
                    rows.append((text.strip(), LABEL_NAMES[score], source))
    except zipfile.BadZipFile as exc:
        raise ValueError("UCI dataset download is not a valid ZIP archive") from exc
    except KeyError as exc:
        raise ValueError(f"UCI dataset is missing expected file: {exc}") from exc

    return rows


def prepare_dataset(output_path: Path) -> int:
    request = Request(
        DATASET_URL,
        headers={"User-Agent": "ds-text-classification-dataset-loader/1.0"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    except OSError as exc:
        raise RuntimeError(f"Could not download training data from UCI: {exc}") from exc

    rows = parse_dataset(archive_bytes)
    if len(rows) != EXPECTED_ROWS_PER_SOURCE * len(SOURCE_FILES):
        raise ValueError(
            f"Expected 3000 UCI reviews, but downloaded {len(rows)}"
        )
    for source in SOURCE_FILES.values():
        source_rows = [row for row in rows if row[2] == source]
        if len(source_rows) != EXPECTED_ROWS_PER_SOURCE:
            raise ValueError(
                f"Expected 1000 reviews from {source}, got {len(source_rows)}"
            )
        for label in LABEL_NAMES.values():
            count = sum(row[1] == label for row in source_rows)
            if count != EXPECTED_ROWS_PER_LABEL:
                raise ValueError(
                    f"Expected 500 {label} {source} reviews, got {count}"
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["text", "label", "source"])
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return len(rows)


if __name__ == "__main__":
    count = prepare_dataset(Path(__file__).resolve().parent / "data.csv")
    print(f"Prepared {count} labelled UCI reviews in data.csv")
