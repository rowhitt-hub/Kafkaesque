# Kafkaesque

A small machine learning project that classifies whether a passage feels **Kafkaesque** or not. It uses **Logistic Regression** over TF-IDF word and character n-gram features, trained on two datasets of 150 sentences each.

The project includes a **Streamlit** app with a simple, terminal-inspired interface for making predictions.

---

## Features

* Binary text classification: **KAFKA** vs **NOT KAFKA**
* Logistic Regression classifier
* TF-IDF features:

  * Word n-grams `(1, 2)`
  * Character n-grams `(2, 5)`
* Trains from two plain-text datasets:

  * `ykafka.txt` — 150 Kafkaesque sentences
  * `nkafka.txt` — 150 regular sentences
* Saves the trained model to `kafka_classifier.pkl`
* Streamlit UI with:

  * Text area for pasting a passage
  * Prediction label
  * Confidence bar and percentage

---

## Demo

```text
┌───────────────────────────────────────────────┐
│                                               │
│             Kafka Classifier                  │
│                                               │
│  Paste a passage below                        │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │ The door was still closed. Gregor       │  │
│  │ looked at it for a long time...         │  │
│  │                                         │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│              [ Analyze Text ]                 │
│                                               │
│  ───────────────────────────────────────────  │
│                                               │
│  Prediction                                   │
│                                               │
│  KAFKA                                        │
│                                               │
│  Confidence                                   │
│  █████████████████░░░  87%                    │
│                                               │
└───────────────────────────────────────────────┘
```

---

## Project Structure

```text
.
├── app.py                  # Streamlit front end
├── train_model.py          # Trains and saves the classifier
├── requirements.txt        # Python dependencies
├── ykafka.txt              # 150 Kafkaesque sentences
├── nkafka.txt              # 150 regular sentences
└── kafka_classifier.pkl    # Generated after training
```

> `kafka_classifier.pkl` is created when you run `train_model.py`. It is not included by default.

---

## Requirements

* Python 3.9+
* `streamlit`
* `scikit-learn`
* `joblib`

### Install Dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit scikit-learn joblib
```

---

## Data Format

Both dataset files should contain one sentence per entry, separated by blank lines:

```text
Sentence one goes here.

Sentence two goes here.

Sentence three goes here.
```

The training script also supports a fallback where each non-empty line is treated as one sentence.

### Expected Files

| File         | Description                        |
| ------------ | ---------------------------------- |
| `ykafka.txt` | Kafkaesque sentences               |
| `nkafka.txt` | Regular / non-Kafkaesque sentences |

The included datasets are expected to contain **150 sentences each**.

---

## Training the Model

From the project folder, run:

```bash
python train_model.py
```

This will:

1. Load `ykafka.txt` and `nkafka.txt`
2. Build a TF-IDF + Logistic Regression pipeline
3. Print an evaluation report on a 20% hold-out set
4. Retrain the classifier on the full dataset
5. Save the trained model as `kafka_classifier.pkl`

### Example Output

```text
Loaded 150 kafkaesque sentences.
Loaded 150 regular sentences.

Evaluation on 20% hold-out set:
              precision    recall  f1-score   support

     regular       0.90      0.87      0.88        30
       kafka       0.88      0.90      0.89        30

    accuracy                           0.88        60
   macro avg       0.89      0.88      0.89        60
weighted avg       0.89      0.88      0.89        60

Saved model to kafka_classifier.pkl
```

> Actual metrics may vary depending on the dataset and scikit-learn version.

---

## Running the App

After training the model, start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

Paste a passage into the text area and click **Analyze Text**.

The app will display:

* **Prediction:** `KAFKA` or `NOT KAFKA`
* **Confidence:** a progress bar and percentage

If the model file is missing, the app will display an error instructing you to run `train_model.py` first.

---

## How It Works

### 1. Feature Extraction

The input text is converted into numerical features using two TF-IDF vectorizers.

#### Word TF-IDF

* `ngram_range=(1, 2)`
* English stop words removed
* `min_df=2`
* `sublinear_tf=True`

#### Character TF-IDF

* `analyzer="char_wb"`
* `ngram_range=(2, 5)`
* `min_df=2`
* `sublinear_tf=True`

The two feature sets are combined using `FeatureUnion`.

---

### 2. Classification

A `LogisticRegression` classifier is trained with:

* `max_iter=1000`
* `class_weight="balanced"`
* `solver="liblinear"`
* `random_state=42`

The classifier learns patterns that distinguish the Kafkaesque examples from the regular examples in the training dataset.

---

### 3. Prediction

For a new passage, the model:

1. Converts the text into TF-IDF features.
2. Passes those features to the Logistic Regression classifier.
3. Calculates the probability for each class.
4. Selects the class with the higher probability.
5. Displays the predicted class and its probability as the confidence score.

---


## Troubleshooting

### `FileNotFoundError: Missing data file`

Make sure both dataset files are in the same folder as `train_model.py`:

```text
ykafka.txt
nkafka.txt
```

---

### `Model not found. Run train_model.py first.`

The Streamlit application requires `kafka_classifier.pkl`.

Train the model first:

```bash
python train_model.py
```

Then start the application:

```bash
streamlit run app.py
```

---

### `streamlit` Command Not Found

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

### Poor Predictions

If the classifier produces unexpected predictions:

* Check that the datasets are clean and correctly labeled.
* Increase the size of the training dataset.
* Add more diverse examples.
* Tune the TF-IDF parameters.
* Experiment with different n-gram ranges.
* Try alternative linear classification models.

Keep in mind that **"Kafkaesque" is subjective**. The classifier does not have an objective understanding of Kafka's writing style; it learns statistical patterns from the examples provided during training.

---

## Limitations

This is a small experimental text-classification project.

* The training dataset contains only 300 sentences.
* The concept of "Kafkaesque" is inherently subjective.
* The model does not understand literature, philosophy, or Kafka's work in a human sense.
* Predictions depend heavily on the quality and diversity of the training data.
* The confidence score represents the model's estimated probability, not a guarantee that the passage is actually Kafkaesque.

For a production-quality classifier, a much larger and carefully curated dataset would be required.

---

## Notes

This project is primarily intended for **educational and experimental use**.

It demonstrates a complete, lightweight machine-learning workflow:

```text
Text Data
    ↓
TF-IDF Feature Extraction
    ↓
Word + Character Features
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Streamlit Interface
```

---

## License

This project is provided for educational and experimental use.

If you plan to distribute or publish the project, consider adding an appropriate open-source license such as MIT.
