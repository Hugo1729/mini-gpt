import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from pytorch_gpt import GPT


class TextDataset(Dataset):
    def __init__(self, file_path, block_sz):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = sorted(list(set(text)))
        self.vocab_size = len(self.tokens)
        self.itos = {i : s for i, s in enumerate(self.tokens)}
        self.stoi = {s : i for i, s in enumerate(self.tokens)}

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_sz = block_sz

    def __len__(self):
        return (len(self.data) - 1) // self.block_sz

    def __getitem__(self, idx):
        # get a chunk of text starting at index idx*block_sz that is one longer than block_sz
        start = idx * self.block_sz
        chunk = self.data[start : start + self.block_sz + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y

@torch.no_grad()
def eval_loss(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += loss.item()

    model.train()

    return total_loss / len(dataloader)

def train(embed_dim, num_heads, block_sz, num_blocks, batch_size, learning_rate,
          epochs, dataset, train_split=0.9, random_seed=42):
    

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    print(f"Device: {device}")

    train_sz = int(len(dataset) * train_split)
    val_sz = len(dataset) - train_sz

    trainset, valset = random_split(dataset, [train_sz, val_sz], generator=torch.Generator().manual_seed(random_seed))

    trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size, shuffle=True, drop_last=True)

    model = GPT(embed_dim,
              dataset.vocab_size,
              num_heads,
              block_sz,
              num_blocks).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Starting training loop")
    print(f"Training Dataloader Length is {len(trainloader)}")
    print(f"Validation Dataloader Length is {len(valloader)}")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx % 100 == 99):
                print(f"Epoch {epoch+1}/{epochs} | batch {batch_idx+1}/{len(trainloader)} | loss {loss.item():.4f}")

        train_loss = total_loss / len(trainloader)
        val_loss = eval_loss(model, valloader, device, criterion)

        print(f"### Done with epoch {epoch+1}/{epochs} | Average Train Loss: {train_loss:.4f} | Average Val Loss: {val_loss:.4f} ###")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return model, train_losses, val_losses