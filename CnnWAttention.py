import torch
import torch.nn as nn
import torchtext
import argparse
import warnings
import numpy as np
import random
import csv
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn

class CNNWithAttention(nn.Module):
    def __init__(self, embeddings, num_filters=100, filter_sizes=(3, 4, 5), d_model=128, num_heads=4, num_classes=2, dropout=0.2):
        super().__init__()
        vocab_size, embedding_dim = embeddings.size()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs, padding=fs // 2)
            for fs in filter_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(50)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)
        self.cnn_to_attention = nn.Linear(num_filters * len(filter_sizes), d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout applied globally

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Convolutional and pooling layers
        embedded = embedded.permute(0, 2, 1)  # Convert to (batch_size, channels, seq_len) for Conv1d
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [self.pool(c) for c in conved]
        cnn_features = torch.cat(pooled, dim=1)  # Concatenate along the channel dimension
        cnn_features = self.dropout(cnn_features)  # Apply dropout after convolutional features

        # Transform to attention input
        cnn_features = cnn_features.permute(0, 2, 1)  # Convert to (batch_size, seq_len, features)
        attention_input = self.cnn_to_attention(cnn_features)

        # Attention mechanism
        attention_output, _ = self.attention(attention_input, attention_input, attention_input)
        global_features = attention_output.mean(dim=1)

        # Fully connected layer with dropout
        global_features = self.dropout(global_features)  # Apply dropout before fully connected layer
        return self.fc(global_features)


def main():
    parser = argparse.ArgumentParser(description="IMDB")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default = 3, type= int)
    parser.add_argument("--csv_output", default="cnn_attention_metrics.csv", type=str)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(args.seed)

    TEXT = torchtext.data.Field(lower=True, batch_first=True, fix_length=256)
    LABEL = torchtext.data.LabelField(dtype=torch.long)
    train_txt, test_txt = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_txt, vectors=torchtext.vocab.GloVe(name="6B", dim=100), max_size=25_000)
    LABEL.build_vocab(train_txt)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_txt, test_txt), batch_size=args.batchsize, device=device
    )

    model = CNNWithAttention(
        embeddings=TEXT.vocab.vectors,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        d_model=512,
        num_heads=8,
        num_classes=2,
        dropout=.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = .0001)

    # Initialize CSV
    with open(args.csv_output, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy", "Time per Epoch (s)", "Total Elapsed Time (s)"])

    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss, epoch_acc = 0, 0

        for batch in train_iter:
            optimizer.zero_grad()
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += (predictions.argmax(1) == batch.label).sum().item()

        epoch_acc /= len(train_iter.dataset)
        train_loss = epoch_loss / len(train_iter)
        train_acc = epoch_acc

        with torch.no_grad():
            model.eval()
            test_loss, test_acc = 0, 0

            for batch in test_iter:
                predictions = model(batch.text)
                loss = criterion(predictions, batch.label)
                test_loss += loss.item()
                test_acc += (predictions.argmax(1) == batch.label).sum().item()

            test_acc /= len(test_iter.dataset)
            test_loss = test_loss / len(test_iter)

        epoch_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - total_start_time

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.3f}, Train Accuracy = {train_acc:.3f}, Test Loss = {test_loss:.3f}, Test Accuracy = {test_acc:.3f}, Time = {epoch_time:.2f}s")

        # Write metrics to CSV
        with open(args.csv_output, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc, epoch_time, total_elapsed_time])


if __name__ == "__main__":
    main()
