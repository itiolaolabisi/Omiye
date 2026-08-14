from pathlib import Path
import ast
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. INPUT DATA
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\US3R\Downloads\New folder\combined_dataset_cleaned.csv"
)


# ============================================================
# 2. OUTPUT FOLDERS
# ============================================================

OUTPUT_FOLDER = INPUT_FILE.parent / "language_id_results"
OUTPUT_FOLDER.mkdir(exist_ok=True)

MODEL_FOLDER = OUTPUT_FOLDER / "trained_model"
MODEL_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Columns:")
print(df.columns.tolist())

print("\nNumber of rows:", len(df))


# ============================================================
# 4. CONVERT TOKEN LISTS BACK INTO SENTENCES
# ============================================================

def tokens_to_text(value):
    tokens = ast.literal_eval(value)
    return " ".join(tokens)


df["text_for_model"] = df["Tokens"].apply(
    tokens_to_text
)

df = df[
    df["text_for_model"].str.strip() != ""
].copy()


# ============================================================
# 5. LABELS
# ============================================================

print("\nClass distribution:")
print(df["labels"].value_counts())


# ============================================================
# 6. X AND Y
# ============================================================

X = df["text_for_model"]
y = df["labels"]


# ============================================================
# 7. TRAIN / DEV / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training:", len(X_train))
print("Development:", len(X_dev))
print("Test:", len(X_test))


# ============================================================
# 8. TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=False,
    sublinear_tf=True,
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_dev_tfidf = vectorizer.transform(X_dev)

X_test_tfidf = vectorizer.transform(X_test)

print(
    "Vocabulary size:",
    len(vectorizer.vocabulary_)
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training complete.")


# ============================================================
# 10. SAVE TRAINED MODEL AND VECTORIZER
# ============================================================

print("\n" + "=" * 60)
print("SAVING TRAINED MODEL")
print("=" * 60)

model_file = MODEL_FOLDER / "language_identifier.pkl"

vectorizer_file = MODEL_FOLDER / "tfidf_vectorizer.pkl"

labels_file = MODEL_FOLDER / "labels.pkl"

joblib.dump(
    model,
    model_file
)

joblib.dump(
    vectorizer,
    vectorizer_file
)

joblib.dump(
    list(model.classes_),
    labels_file
)

print("Model saved to:")
print(model_file)

print("\nVectorizer saved to:")
print(vectorizer_file)

print("\nLabels saved to:")
print(labels_file)


# ============================================================
# 11. DEVELOPMENT SET
# ============================================================

dev_predictions = model.predict(
    X_dev_tfidf
)

dev_accuracy = accuracy_score(
    y_dev,
    dev_predictions
)

dev_macro_f1 = f1_score(
    y_dev,
    dev_predictions,
    average="macro"
)

print("\n" + "=" * 60)
print("DEVELOPMENT RESULTS")
print("=" * 60)

print(
    "Accuracy:",
    round(dev_accuracy, 4)
)

print(
    "Macro-F1:",
    round(dev_macro_f1, 4)
)


# ============================================================
# 12. FINAL TEST
# ============================================================

test_predictions = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro Precision:",
    round(precision, 4)
)

print(
    "Macro Recall:",
    round(recall, 4)
)

print(
    "Macro F1:",
    round(macro_f1, 4)
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_FOLDER / "classification_report.csv"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Language Identifier - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. OVERALL METRICS PLOT
# ============================================================

metrics = {
    "Accuracy": accuracy,
    "Macro Precision": precision,
    "Macro Recall": recall,
    "Macro F1": macro_f1
}

plt.figure()

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Language Identifier Performance"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "overall_metrics.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. PER-CLASS F1
# ============================================================

class_report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)

class_names = [
    label
    for label in labels
]

f1_values = [
    class_report[label]["f1-score"]
    for label in class_names
]

plt.figure()

plt.bar(
    class_names,
    f1_values
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "F1 Score by Language"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "f1_by_language.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": test_predictions
})

test_results.to_csv(
    OUTPUT_FOLDER / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel artifacts:")
print(model_file)

print(vectorizer_file)

print(labels_file)

print("\nResults:")
print(OUTPUT_FOLDER)

print("\nThe trained model can now be reused without retraining.")
