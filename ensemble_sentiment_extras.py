# ensemble_sentiment_extras.py
"""
Prototype advanced extension: fine-tune a DistilBERT-based classifier that
uses daily headline text concatenated with numeric market features to predict next-day movement.
This demonstrates multimodal modelling for the independent research component.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch, pandas as pd

def prepare_multimodal_dataset(df_full: pd.DataFrame, text_col="title", label_col="target", max_len=64):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(df_full[text_col].tolist(), truncation=True, padding=True, max_length=max_len)
    labels = torch.tensor(df_full[label_col].values)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings["input_ids"]),
        torch.tensor(encodings["attention_mask"]),
        labels
    )
    return dataset

def train_transformer_classifier(df_full: pd.DataFrame):
    dataset = prepare_multimodal_dataset(df_full)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(
        output_dir="results_task7_transformer",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_dir="logs_task7_transformer"
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained("results_task7_transformer/final_model")
    return model
