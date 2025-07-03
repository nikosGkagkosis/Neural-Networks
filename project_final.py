from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import time
from torch.optim import Adam
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/nfc_project/news-classification.csv')

# Combine title and content into a single text column
df['text'] = df[['title', 'content']].apply(lambda x: ' . '.join(x.astype(str)), axis=1)

# Clean the text
pattern = r'[^a-zA-Z,\d]'
pattern2 = r'[0123456789]'
df['text'] = df['text'].apply(lambda text: re.sub(pattern, ' ', str(text)))
df['text'] = df['text'].apply(lambda text: re.sub(pattern2, ' ', str(text)))

# Split data into input (x) and labels (y1 and y2)
x = df['text']
y1 = df['category_level_1']
y2 = df['category_level_2']

# Tokenization and label encoding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
unique_labels = df['category_level_1'].unique()
labels = {label: idx for idx, label in enumerate(unique_labels)}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category_level_1']]
        self.texts = [tokenizer.encode_plus(text,
                                            padding='max_length', max_length=512, truncation=True,
                                            return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

# Split dataset into train, validation, and test sets
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8 * len(df)), int(.9 * len(df))])

# Define the classifier model
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, len(unique_labels))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        output = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = output.last_hidden_state[:, 0, :]  # Using the CLS token representation
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

# Training function
def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

# Define the number of epochs and learning rate
EPOCHS = 8
LR = 4e-6

# Train the model
model = BertClassifier()
train(model, df_train, df_val, LR, EPOCHS)
# Accessing the final trained parameter values of the model
print("Final trained parameter values:")
for name, param in model.named_parameters():
    print(f'Parameter name: {name}')
    print(f'Parameter value: {param}')

# Evaluation function
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    total_time = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            start_time = time.time()
            output = model(input_id, mask)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    print(f'Average Inference Time per Sample: {total_time / len(test_data): .3f} seconds')

# Evaluate the model
evaluate(model, df_test)
