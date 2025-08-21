import torch
import os
from .transformer import Transformer, create_masks
from .training import preprocess_text

def summarize(text, model_path='checkpoints/best_model.pth', max_len=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Construct absolute path to the model checkpoint
    base_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_model_path = os.path.join(base_dir, model_path)
    
    checkpoint = torch.load(absolute_model_path, map_location=device)
    word2idx = checkpoint['word2idx']
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    
    model = Transformer(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    src = [word2idx.get(token, 1) for token in preprocess_text(text)]
    src = [2] + src + [3]
    src = torch.tensor(src).unsqueeze(0).to(device)
    
    tgt = torch.tensor([[2]]).to(device)
    
    for _ in range(max_len):
        src_mask, tgt_mask = create_masks(src, tgt)
        
        with torch.no_grad():
            output = model(src, tgt, src_mask.to(device), tgt_mask.to(device))
        
        next_token = output.argmax(2)[:, -1].item()
        if next_token == 3:
            break
        
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
        
    summary_tokens = [idx2word.get(i, '<UNK>') for i in tgt.squeeze().tolist()]
    return " ".join(summary_tokens[1:])
