import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

# Read your CSV file and split it into train and test sets
print("Preparing Dataset...\n")
df = pd.read_csv('HATEX_binaryclass.csv')
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Dataset preparation done!\n")
# Load a pre-trained BERT model and tokenizer
print("Preparing Model...\n")
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels (modify as needed)
print("Model preparation done!\n")
# Tokenize and preprocess the text data

print("Tokenizing...\n")
max_length = 64
X_train_encoded = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
X_test_encoded = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')

y_train_encoded = torch.tensor(y_train.values)
y_test_encoded = torch.tensor(y_test.values)
print("Tokenization done!\n")
# Create PyTorch DataLoader for efficient batching
batch_size = 32
train_data = TensorDataset(X_train_encoded['input_ids'], X_train_encoded['attention_mask'], y_train_encoded)
train_loader = DataLoader(train_data, batch_size=batch_size)
valid_data = TensorDataset(X_train_encoded['input_ids'], X_train_encoded['attention_mask'], y_train_encoded)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

best_val_accuracy = 0.0

print("Training...\n")
# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started...\n")
    model.train()
    total_loss = 0.0
    total_batch = len(train_loader)
    current_batch = 1
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if current_batch % 10 == 0 or current_batch == total_batch-1:
            print(f"Epoch:{epoch+1} - Batch# {current_batch}/{total_batch} completed!")
        current_batch = current_batch + 1
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

    model.eval()
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_targets, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model checkpoint based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model_bertweet_HATEX_binary.pt')


# Load the best model checkpoint for testing
model.load_state_dict(torch.load('best_model_bertweet_HATEX_binary.pt'))

# Evaluation
print("Evaluating...\n")
model.eval()
y_pred = []
with torch.no_grad():
    for batch in DataLoader(TensorDataset(X_test_encoded['input_ids'], X_test_encoded['attention_mask']), batch_size=batch_size):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.argmax(dim=1)
        y_pred.extend(logits.cpu().numpy())

y_pred = np.array(y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n------------------------------")
print("BerTweet - HATEX Binary Class")
print("------------------------------\n")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
