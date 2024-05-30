# train_model.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Read Data
CompleteSet = pd.read_csv("/content/CompleteDataset_OneHot")
VTPAN_train = pd.read_csv('/content/drive/MyDrive/IM/VTPAN_csv/VTPAN-train.csv')
VTPAN_test = pd.read_csv('/content/drive/MyDrive/IM/VTPAN_csv/VTPAN-test.csv')

# Split dataset into training and validation (70:30)
X_train, X_val, y_train, y_val = train_test_split(CompleteSet['Chats'], CompleteSet['labels'], stratify=CompleteSet['labels'], test_size=0.30, random_state=42, shuffle=True)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Tokenize data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=512)

# Create torch dataset
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

train_dataset = Dataset(train_encodings, y_train.tolist())
val_dataset = Dataset(val_encodings, y_val.tolist())

# Define Trainer
training_args = TrainingArguments(
    output_dir="xmlRoberta_GenData_Double",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model
trainer.push_to_hub()
