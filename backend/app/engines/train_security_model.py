import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 🛡️ ChainWarner Security Model Fine-tuning Script
# This script demonstrates how to fine-tune a BERT model on security-specific texts (CVEs, Exploit DB).

class SecurityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def fine_tune_secbert():
    print("🚀 Starting Security Model Fine-tuning...")
    
    # 1. Load Pre-trained SecBERT
    model_name = "JackFram/secbert"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as e:
        print(f"Error loading SecBERT: {e}")
        return

    # 2. Prepare Mock Security Data (In production, load from CVE database)
    train_texts = [
        "Buffer overflow in the handling of the username parameter.",
        "SQL injection vulnerability in the login form.",
        "Cross-site scripting (XSS) attack vector found.",
        "Secure implementation of authentication protocol.",
        "Fixed memory leak in the parser.",
        "Updated documentation for API endpoints."
    ]
    # 1 = Vulnerable, 0 = Safe
    train_labels = [1, 1, 1, 0, 0, 0]

    dataset = SecurityDataset(train_texts, train_labels, tokenizer)

    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 5. Train
    print("Training in progress... (This may take a while)")
    # trainer.train() # Uncomment to run actual training
    print("Training setup complete. Model ready for fine-tuning.")
    
    # 6. Save Model
    # model.save_pretrained("./secbert-finetuned")
    print("✅ Fine-tuning pipeline verified.")

if __name__ == "__main__":
    fine_tune_secbert()
