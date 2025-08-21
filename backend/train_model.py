import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer
from model.training import build_vocab, TextDataset, train_model
from datasets import load_dataset

def main():
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    
    texts = [item['article'] for item in dataset]
    summaries = [item['highlights'] for item in dataset]
    
    word2idx = build_vocab(texts + summaries)
    vocab_size = len(word2idx)
    
    dataset = TextDataset(texts, summaries, word2idx)
    dataloader = DataLoader(dataset, batch_size=2)
    
    model = Transformer(vocab_size=vocab_size)
    
    train_model(model, dataloader)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx
    }, 'backend/model/checkpoints/best_model.pth')

if __name__ == '__main__':
    main()
