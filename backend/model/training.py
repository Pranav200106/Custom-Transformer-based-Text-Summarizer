import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from tqdm import tqdm

def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(texts, vocab_size=1000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(preprocess_text(text))
    
    word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for i, (word, _) in enumerate(word_counts.most_common(vocab_size - 4)):
        word2idx[word] = i + 4
    return word2idx

class TextDataset(Dataset):
    def __init__(self, texts, summaries, word2idx, max_len=128):
        self.texts = texts
        self.summaries = summaries
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = [self.word2idx.get(token, 1) for token in preprocess_text(self.texts[idx])]
        tgt = [self.word2idx.get(token, 1) for token in preprocess_text(self.summaries[idx])]
        
        src = [2] + src + [3]
        tgt = [2] + tgt + [3]
        
        src = src[:self.max_len] + [0] * (self.max_len - len(src))
        tgt = tgt[:self.max_len] + [0] * (self.max_len - len(tgt))
        
        return torch.tensor(src), torch.tensor(tgt)

def train_model(model, dataloader, epochs=5, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            from .transformer import create_masks
            src_mask, tgt_mask = create_masks(src, tgt_input)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask.to(device), tgt_mask.to(device))
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
