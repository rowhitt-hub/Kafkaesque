from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_sentences(filename):
    """
    Load sentences from a text file.

    Supports:
    - one sentence per line
    - sentences separated by blank lines

    Empty lines are ignored.
    """

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' in:\n{Path.cwd()}"
        )

    # utf-8-sig also handles files saved with a UTF-8 BOM
    with open(path, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    sentences = []

    for line in lines:
        sentence = line.strip()

        if sentence:
            sentences.append(sentence)

    return sentences


def main():

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    kafka_sentences = load_sentences("ykafka.txt")
    normal_sentences = load_sentences("nkafka.txt")

    print(f"Kafkaesque sentences: {len(kafka_sentences)}")
    print(f"Regular sentences: {len(normal_sentences)}")

    # --------------------------------------------------
    # Validate datasets
    # --------------------------------------------------

    if len(kafka_sentences) == 0:
        raise ValueError(
            "\nykafka.txt contains no readable sentences.\n"
            "Check that the file is in the same folder as "
            "train_model.py and that it contains text."
        )

    if len(normal_sentences) == 0:
        raise ValueError(
            "\nnkafka.txt contains no readable sentences.\n"
            "Check that the file is in the same folder as "
            "train_model.py and that it contains text."
        )

    # --------------------------------------------------
    # Create features and labels
    # --------------------------------------------------

    X = kafka_sentences + normal_sentences

    # 1 = Kafkaesque
    # 0 = Regular
    y = (
        [1] * len(kafka_sentences)
        + [0] * len(normal_sentences)
    )

    print(f"\nTotal samples: {len(X)}")
    print(f"Kafka class: {y.count(1)}")
    print(f"Regular class: {y.count(0)}")

    # --------------------------------------------------
    # Split training and testing data
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_features=10000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Regular",
                "Kafkaesque"
            ]
        )
    )

    # --------------------------------------------------
    # Retrain on all data
    # --------------------------------------------------

    print("Retraining using the complete dataset...")

    model.fit(X, y)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_filename = "kafka_classifier.joblib"

    joblib.dump(model, model_filename)

    print(f"\nModel successfully saved as: {model_filename}")


if __name__ == "__main__":
    main()