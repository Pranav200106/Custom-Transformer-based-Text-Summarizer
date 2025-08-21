import torch
import os
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

    # Load latest checkpoint if available
    checkpoint_dir = 'backend/model/checkpoints'
    latest_checkpoint_path = None
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.startswith('model_epoch_') and f.endswith('.pth'):
                epoch_num = int(f.split('_')[2].split('.')[0])
                if latest_checkpoint_path is None or epoch_num > int(os.path.basename(latest_checkpoint_path).split('_')[2].split('.')[0]):
                    latest_checkpoint_path = os.path.join(checkpoint_dir, f)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path} for continued training...")
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu') # Load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    train_model(model, dataloader, start_epoch=start_epoch, best_val_loss=best_val_loss)

if __name__ == '__main__':
    main()
