import math
import argparse
import torch
import torch.nn as nn
import torchtext
import warnings
import time
import csv


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Net(nn.Module):
    def __init__(self, embeddings, nhead=8, dim_feedforward=512, num_layers=6, dropout=0.1):
        super().__init__()
        vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)  # Dropout layer added
        self.classifier = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)  # Dropout after positional encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)  # Dropout before classification
        return self.classifier(x)

def main():
    parser = argparse.ArgumentParser(description="IMDB")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default=4, type=int)
    parser.add_argument("--csv_output", default="transformer_metrics.csv", type=str)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning)

    TEXT = torchtext.data.Field(lower=True, batch_first=True, fix_length=256)
    LABEL = torchtext.data.LabelField(dtype=torch.long)
    train_txt, test_txt = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_txt, vectors=torchtext.vocab.GloVe(name="6B", dim=50), max_size=25_000)
    LABEL.build_vocab(train_txt)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_txt, test_txt), batch_size=args.batchsize, device=device
    )

    model = Net(TEXT.vocab.vectors, nhead=10, dim_feedforward=512, num_layers=6, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=.0001)

    # Initialize CSV
    with open(args.csv_output, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy", "Time per Epoch (s)", "Total Elapsed Time (s)"])

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        epoch_acc = 0

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

            test_loss /= len(test_iter)
            test_acc /= len(test_iter.dataset)

        epoch_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - start_time

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.3f}, Train Accuracy = {train_acc:.3f}, "
              f"Test Loss = {test_loss:.3f}, Test Accuracy = {test_acc:.3f}, Time = {epoch_time:.2f}s")

        # Write to CSV
        with open(args.csv_output, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc, epoch_time, total_elapsed_time])


if __name__ == "__main__":
    main()
