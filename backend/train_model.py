import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer
from model.training import build_vocab, TextDataset, train_model

def main():
    texts = ["this is a test sentence", "this is another one"]
    summaries = ["a test", "another one"]
    
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
    }, 'backend/model/checkpoints/simple_model.pth')

if __name__ == '__main__':
    main()
