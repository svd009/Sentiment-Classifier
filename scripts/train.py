from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def fine_tune_model():
    # Load IMDB dataset
    dataset = load_dataset("imdb")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Preprocess function
    def preprocess(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Training arguments (simplified)
    training_args = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(500)),  # Reduced to 500 for speed
        eval_dataset=dataset["test"].shuffle(seed=42).select(range(200)),
        tokenizer=tokenizer
    )

    # Train and save
    trainer.train()

    model.save_pretrained("./models")
    tokenizer.save_pretrained("./models")

    print("✅ Model and tokenizer saved to ./models")

if __name__ == "__main__":
    fine_tune_model()
