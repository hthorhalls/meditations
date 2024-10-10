import torch 
from tokenizer import Tokenizer
from model import AureliusGPT

device = 'cuda' if torch.cuda.is_available() else 'mps'

""" Hyper Params """
context_size = 256
embedding_size = 384
batch_size = 64
eval_iters = 200
learning_rate = 1e-4
max_iters = 4000
eval_interval = 500
n_layer = 6
n_head = 6

torch.manual_seed(17)

# Read in our dataset, tokenize and split into train/test
with open('meditations.txt', 'r') as f:
    data = f.read()
n = len(data)
tokenizer = Tokenizer(data)
tokenized_data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
train_data = tokenized_data[:int(n*0.9)]
val_data = tokenized_data[int(n*0.9):]

print(f'Num training tokens: {train_data.size(dim=0)}')
print(f'Num test tokens: {val_data.size(dim=0)}')
print(f'Vocab size: {tokenizer.vocab_size}')


"""
Training harness functions 
"""

# Generates (B, T) tensors for x and y
# Each batch is a sequence of length T of tokens where y_i is x_i offset by 1
def get_batch(data, batch_size): 
    ix = torch.randint(len(data)-context_size, (batch_size, )) 
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    splits = {'train': train_data, 'val': val_data }
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

model = AureliusGPT(tokenizer.vocab_size, context_size, embedding_size, n_head, n_layer, device)
m = model.to(device)
x, y = get_batch(train_data, batch_size)


print(sum(p.numel() for p in m.parameters()), 'parameters')
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Grab a batch 
    xb, yb = get_batch(train_data, batch_size)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long,  device=device)
open('out.txt', 'w').write(tokenizer.decode(m.sample(context, length=1000, tokenizer=tokenizer)[0].tolist())) # Not very intelligble but doesn't just output shit