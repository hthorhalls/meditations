import torch
import torch.nn as nn
import torch.nn.functional as F


dropout = 0.3

class FeedForward(nn.Module): 
    
    def __init__(self, embedding_size): 
        super().__init__()
        self.linear = nn.Linear(embedding_size, 4 * embedding_size)
        self.relu = nn.ReLU()
        self.second_linear = nn.Linear(4 * embedding_size, embedding_size)
        self.m = nn.Sequential(self.linear, self.relu, self.second_linear, nn.Dropout(dropout))
        
        
    def forward(self, x): 
        return self.m(x)


class Head(nn.Module):

    def __init__(self, head_size, context_size, embedding_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        
        # This is a buffer and not a regular parameter since we don't need to update this during backprop
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)
        
    # x is of size B, T, embedding_size
    # output is B, T, head_size
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x) 
        
        # Our attention pattern, scaled by sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 

        # Masking tokens in the future out, -inf will make the softmax normalization well behaved
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        # probabilistic normalization of the attention scores 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # weighted aggregation of v 
        v = self.value(x) # B, T, head_size
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, context_size, embedding_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, context_size, embedding_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, context_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, context_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
        
        
# Let's train! 
class AureliusGPT(nn.Module):
    
    def __init__(self, vocab_size, context_size, embedding_size, n_head, n_layer, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.token_position_embedding_table = nn.Embedding(context_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, n_head=n_head, context_size=context_size) for _ in range(n_layer)])
        self.lm_head = nn.Linear(embedding_size, vocab_size)
        self.ln_f = nn.LayerNorm(embedding_size)
        self.context_size = context_size
        self.device = device

        
    # idx is a (B, T) sized tensor 
    def forward(self, idx, targets=None):
        B, T = idx.shape # Get the batch size and the current context size
        
        tok_emb = self.token_embedding_table(idx) # B, T, embed_size
        
        # get all the position embeddings for each position in our context
        pos_emb = self.token_position_embedding_table(torch.arange(T, device=self.device)) # B, T, embed_size
        x = tok_emb + pos_emb # B, T, embed_size
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def sample(self, idx, length, tokenizer):
        
        # idx: (B, T)
        for _ in range(length):
            idx_cond = idx[:, -self.context_size:] # make sure we are only inputting context up to context_size
            logits, loss = self(idx_cond) # logits is a (B, T, C) tensor with raw output of the model for each token
            logits = logits[:, -1, :] # Let's only select the prediction for the last token
            probs = F.softmax(logits, dim=-1) # Get our probabilities (B, C) vector
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    