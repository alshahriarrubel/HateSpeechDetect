import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

##########################################
# Dataset AHSD_binaryclass               #
##########################################

# Read your CSV file and split it into train and test sets
print("Preparing Dataset AHSD_binaryclass...\n")
df = pd.read_csv('AHSD_binaryclass.csv')
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Dataset preparation done!\n")
# Load a pre-trained BERT model and tokenizer
print("Preparing Model...\n")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels (modify as needed)
print("Model preparation done!\n")
# Tokenize and preprocess the text data

print("Tokenizing...\n")
max_length = 64
X_test_encoded = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
y_test_encoded = torch.tensor(y_test.values)
print("Tokenization done!\n")
# Create PyTorch DataLoader for efficient batching
batch_size = 32

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Load the best model checkpoint for testing
model.load_state_dict(torch.load('best_model_bert_Merged_binary.pt'))

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

print("\n\n------------------------------")
print("BerT - AHSD Binary Class")
print("------------------------------\n")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


##########################################
# Dataset MHS_binaryclass                #
##########################################

# Read your CSV file and split it into train and test sets
print("Preparing Dataset MHS_binaryclass...\n")
df = pd.read_csv('MHS_binaryclass.csv')
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Dataset preparation done!\n")
# Load a pre-trained BERT model and tokenizer
print("Preparing Model...\n")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels (modify as needed)
print("Model preparation done!\n")
# Tokenize and preprocess the text data

print("Tokenizing...\n")
max_length = 64
X_test_encoded = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
y_test_encoded = torch.tensor(y_test.values)
print("Tokenization done!\n")
# Create PyTorch DataLoader for efficient batching
batch_size = 32

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Load the best model checkpoint for testing
model.load_state_dict(torch.load('best_model_bert_Merged_binary.pt'))

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

print("\n\n------------------------------")
print("BerT - MHS Binary Class")
print("------------------------------\n")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


##########################################
# Dataset HATEX_binaryclass              #
##########################################

# Read your CSV file and split it into train and test sets
print("Preparing Dataset HATEX_binaryclass...\n")
df = pd.read_csv('HATEX_binaryclass.csv')
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Dataset preparation done!\n")
# Load a pre-trained BERT model and tokenizer
print("Preparing Model...\n")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels (modify as needed)
print("Model preparation done!\n")
# Tokenize and preprocess the text data

print("Tokenizing...\n")
max_length = 64
X_test_encoded = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
y_test_encoded = torch.tensor(y_test.values)
print("Tokenization done!\n")
# Create PyTorch DataLoader for efficient batching
batch_size = 32

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Load the best model checkpoint for testing
model.load_state_dict(torch.load('best_model_bert_Merged_binary.pt'))

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

print("\n\n------------------------------")
print("BerT - HATEX Binary Class")
print("------------------------------\n")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
