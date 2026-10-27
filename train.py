import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

def train(embed_dim,
    num_heads,
    block_sz,
    num_blocks,
    batch_size,
    learning_rate,
    epochs,
    input_file):
    

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    print(f"Device: {device}")

    dataset = TextDataset(input_file, block_sz)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    model = GPT(embed_dim,
              dataset.vocab_size,
              num_heads,
              block_sz,
              num_blocks).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Starting training loop")
    print(f"Dataloader Length is {len(dataloader)}")

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx % 100 == 99):
                print(f"Epoch {epoch+1}/{epochs} | batch {batch_idx+1}/{len(dataloader)} | loss {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)

        print(f"### Done with epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f} ###")
        losses.append(avg_loss)

    return model, losses