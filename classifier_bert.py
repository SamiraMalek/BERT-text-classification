from datasets import Dataset, DatasetDict
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_path = '/scratch/sxm6547/CSE_584/Midterm_project/classifier/dataset/train.csv'
train_df = pd.read_csv(train_path)  # Replace with your dataset file
val_path = '/scratch/sxm6547/CSE_584/Midterm_project/classifier/dataset/val.csv'
val_df = pd.read_csv(val_path)  # Replace with your validation dataset file


# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create a dataset dictionary
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': val_dataset
})

# Prepare and tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

# Setup evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load pretrained model and evaluate model after each epoch
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=8)
model.to(device)  # Move the model to the GPU

# Set training arguments
training_args = TrainingArguments(
    output_dir="/scratch/sxm6547/CSE_584/Midterm_project/classifier/test_trainer",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    logging_dir="/scratch/sxm6547/CSE_584/Midterm_project/classifier/log",  # Directory for storing logs
    logging_strategy="steps",  # Log training metrics at regular intervals
    logging_steps=1,  # Log every 10 steps (adjustable)
    per_device_train_batch_size=32,  # Batch size for training
    per_device_eval_batch_size=32,  # Batch size for evaluation
    num_train_epochs=10,  # Number of epochs
    weight_decay=0.01,
    load_best_model_at_end=True,  # Load the best model after training
    save_strategy="epoch",  # Save model after every epoch
    metric_for_best_model="accuracy",  # Metric to optimize for
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

# Save the model and the tokenizer
model_save_path = "/scratch/sxm6547/CSE_584/Midterm_project/classifier/model"
tokenizer_save_path = "/scratch/sxm6547/CSE_584/Midterm_project/classifier/tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)