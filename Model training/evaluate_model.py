# evaluate_model.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate
import argparse

def load_data():
    # Load test data
    VTPAN_test = pd.read_csv('/content/drive/MyDrive/IM/VTPAN_csv/VTPAN-test.csv')

    # One-hot encoding the labels in testset
    def to_sentiment(label):
        if label == 'predator':
            return 1
        elif label == 'non-predator':
            return 0

    VTPAN_test['label'] = VTPAN_test['label'].apply(to_sentiment)
    X_test = list(VTPAN_test["segment"])
    y_test = list(VTPAN_test["label"])

    return X_test, y_test

def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained("xmlRoberta_GenData_Double")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    return model, tokenizer

def tokenize_data(tokenizer, X_test):
    return tokenizer(X_test, truncation=True, padding=True, max_length=512)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def evaluate_model():
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--all", action="store_true", help="Run all evaluation methods.")
    parser.add_argument("--f1", action="store_true", help="Evaluate using F1 score.")
    parser.add_argument("--classification_report", action="store_true", help="Generate classification report.")
    parser.add_argument("--confusion_matrix", action="store_true", help="Generate confusion matrix.")
    parser.add_argument("--roc", action="store_true", help="Generate ROC curve.")

    args = parser.parse_args()

    X_test, y_test = load_data()
    model, tokenizer = load_model_and_tokenizer()
    test_encodings = tokenize_data(tokenizer, X_test)
    test_dataset = Dataset(test_encodings, y_test)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    if args.all or args.f1:
        metric = evaluate.load("f1")
        macro_f1 = metric.compute(predictions=preds, references=y_test, average="macro")
        print("Macro F1 Score:", macro_f1)

    if args.all or args.classification_report:
        class_names = ['non predator', 'predator']
        print(classification_report(y_test, preds, target_names=class_names))

    if args.all or args.confusion_matrix:
        class_names = ['non predator', 'predator']
        cm = confusion_matrix(y_test, preds)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        show_confusion_matrix(df_cm)
        plt.show()

    if args.all or args.roc:
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

def show_confusion_matrix(cm):
    hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    evaluate_model()
