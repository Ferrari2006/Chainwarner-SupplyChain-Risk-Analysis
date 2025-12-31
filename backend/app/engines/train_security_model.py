from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import random

# ==========================================
# 🛡️ ChainWarner Security Model Trainer
# ==========================================
# This script demonstrates how to fine-tune the 'JackFram/secbert' model
# on a custom security dataset (CVE descriptions + Risk Labels).
# ==========================================

class SecurityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    print("🚀 Starting Security BERT Fine-tuning...")
    
    # 1. Load Pre-trained Security Model
    # We use JackFram/secbert which is pre-trained on cybersecurity texts
    model_name = "JackFram/secbert" 
    print(f"Loading base model: {model_name}")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except:
        print("⚠️  Network error or model not found. Using 'bert-base-uncased' as fallback.")
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Prepare Synthetic Security Dataset (Demo)
    # In production, load from 'cve_dataset.csv'
    texts = [
        "Buffer overflow in the XML parser allows remote attackers to execute arbitrary code.",
        "Cross-site scripting (XSS) vulnerability in the login page.",
        "The application is secure and follows best practices.",
        "Fixed a minor typo in the documentation.",
        "SQL injection vulnerability in the user search component.",
        "Updated dependency versions to latest stable release.",
        "Critical remote code execution vulnerability discovered in core module.",
        "Added unit tests for the new feature."
    ]
    # 1 = High Risk, 0 = Low Risk
    labels = [1, 1, 0, 0, 1, 0, 1, 0]

    print(f"Training on {len(texts)} samples...")
    
    # 3. Tokenize
    train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    train_dataset = SecurityDataset(train_encodings, labels)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        no_cuda=True # Force CPU for compatibility
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset 
    )

    # 6. Train
    print("Training started...")
    trainer.train()
    print("✅ Fine-tuning complete!")
    
    # 7. Save Model
    save_path = "./secbert_finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
