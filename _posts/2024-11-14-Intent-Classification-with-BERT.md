---
title: "🚀 Boost Your Chatbot's IQ: Intent Classification with BERT and PyTorch in Python"
date: 2024-11-14
last_modified_at: 2024-11-14
tags:
    - NLP
header:
image: /assets/images/headers/2023-02-05-Intent-Classification-with-BERT.png
teaser: /assets/images/headers/2023-02-05-blog-portfolio-with-mm-header.jpg
---

## Table of Contents
- [Introduction](#introduction)
- [Why Use BERT for Intent Classification?](#why-use-bert-for-intent-classification)
- [Getting Started: Setting Up the Environment](#getting-started-setting-up-the-environment)
- [Data Preparation for Intent Classification](#data-preparation-for-intent-classification)
- [Building the Model](#building-the-model)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Conclusion](#conclusion)
- [Code Summary](#code-summary)

---

## Introduction

In the world of chatbots, **intent classification** is crucial. It determines how accurately your bot understands the user’s request and responds appropriately. Leveraging the power of **BERT** (Bidirectional Encoder Representations from Transformers) in **PyTorch** allows us to boost our chatbot’s IQ by accurately categorizing intents. In this blog, we'll walk through how to use BERT and PyTorch to build a simple yet powerful intent classifier in Python.

## Why Use BERT for Intent Classification?

BERT has transformed NLP by enabling models to capture deep contextual relationships within text. Its **transformer architecture** allows BERT to excel in understanding the nuances of human language, making it perfect for tasks like intent classification, where subtle differences in user queries can drastically alter the response.

## Getting Started: Setting Up the Environment

To get started, you'll need to install the following libraries:

```bash
pip install torch transformers
```

## Data Preparation for Intent Classification

For this tutorial, let’s assume a small dataset with various user intents, such as <span style="background-color:grey">greeting</span>, <span style="background-color:grey">weather</span>, <span style="background-color:grey">news</span>, and  <span style="background-color:grey">goodbye</span>.

Here’s an example dataset:

```python
import pandas as pd

data = {
    'text': [
        "Hello, how are you?",
        "What's the weather like today?",
        "Tell me the latest news.",
        "Goodbye, see you later."
    ],
    'intent': ['greeting', 'weather', 'news', 'goodbye']
}
df = pd.DataFrame(data)
```

For encoding our intent classes, we’ll use label encoding:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['intent'] = label_encoder.fit_transform(df['intent'])
```

## Building the Model
### Step 1: Load BERT Tokenizer and BERT Model

The **BERT tokenizer** converts our text inputs into the format required for BERT, which includes:

-    Token IDs
-    Attention Masks

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
```

### Step 2: Create PyTorch Dataset and DataLoader

Using PyTorch’s DataLoader, we’ll create mini-batches of data for more efficient training:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=32, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

dataset = IntentDataset(df['text'].tolist(), df['intent'].tolist())
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### Step 3: Define the Intent Classifier Model

We'll define a classifier model that uses BERT embeddings followed by a dense layer for classification.

```python
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(IntentClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(last_hidden_state)
        return logits

model = IntentClassifier(n_classes=len(df['intent'].unique()))
```

## Training the Model
### Step 1: Set Up Loss and Optimizer

We'll use **cross-entropy loss** as our loss function and **Adam** as our optimizer.

```python
from torch.optim import Adam

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-5)
```

### Step 2: Training Loop

Here’s the training loop, where we’ll feed our data to the model, calculate the loss, and optimize the parameters.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for epoch in range(3):  # Set number of epochs
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} Loss: {loss.item()}")
```

## Evaluating the Model

To evaluate the performance of our intent classifier, we’ll use metrics like accuracy. This helps us understand how well our model classifies user intents after training.

```python
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=label_encoder.classes_)
    
    return accuracy, report

# Call the evaluation function
accuracy, report = evaluate_model(model, dataloader)
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
```

The classification report provides a detailed breakdown, including:

-    Precision: How accurate the positive predictions are.
-    Recall: How well the classifier identifies all positive instances.
-    F1-Score: The balance between precision and recall.
A well-balanced F1-score across classes indicates a robust model that can classify intents effectively.

## Conclusion
By using BERT and PyTorch, we’ve created a powerful intent classifier for a chatbot, allowing it to accurately interpret user requests. This foundational model can be further improved by fine-tuning on a larger, domain-specific dataset, or by applying transfer learning techniques.

# Code Summary
Here's a summary of the main steps covered:

## 1. Import dependencies
## 2. Load tokenizer and model
## 3. Preprocess and encode data
## 4. Define IntentClassifier model
## 5. Set up DataLoader, criterion, and optimizer
## 6. Train model
## 7. Evaluate model

With just a few lines of code, you can upgrade your chatbot’s intelligence, making it smarter and more responsive.
