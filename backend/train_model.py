import torch
import os
from torch.utils.data import DataLoader
from model.transformer import Transformer
from model.training import build_vocab, TextDataset, train_model
from datasets import load_dataset

def main():
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train") # Use full training split
    
    texts = [item['article'] for item in dataset]
    summaries = [item['highlights'] for item in dataset]
    
    # Increase vocabulary size
    word2idx = build_vocab(texts + summaries, vocab_size=10000) 
    vocab_size = len(word2idx)
    
    dataset = TextDataset(texts, summaries, word2idx)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Increase batch size and shuffle
    
    model = Transformer(vocab_size=vocab_size)

    # Ensure a fresh start for training with new data and vocab
    print("Starting training from scratch with updated dataset and vocabulary.")
    
    train_model(model, dataloader) # Remove start_epoch and best_val_loss as we are starting fresh

if __name__ == '__main__':
    main()
