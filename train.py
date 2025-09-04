import torch
import torch.nn as nn
from torch.nn import functional as F


# =============== Hyperparameters =============== 
batch_size = 64 # How many independent sequence will we process in parallel
block_size = 256 # what is the maximum context length for prediction?
n_embd = 384
n_head = 6
n_layer = 6

learning_rate = 3e-4

max_iters = 5000
eval_interval = 500
eval_iters = 200
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # FOR GPT2 small 124M params
# block_size = 1024   # still manageable
# n_embd = 768       # slightly under 768
# n_head = 12       # divides 704 exactly? Actually 11×64 = 704, perfect
# n_layer = 12       # GPT-2 small style




torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()



# Making a vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# vocab_size = 50000
# Creating a tokenizer

stoi={}
for i,ch in enumerate(chars):
    stoi[ch] = i

itos ={}
for i,ch in enumerate(chars):
    itos[i]=ch


def encode(s):
    lis =[]
    for c in s:
        lis.append(stoi[c])
    return lis

def decode(l):
    lis=[]
    for i in l:
        lis.append(itos[i])
    return ''.join(lis)

# print(encode('hi there'))
# print(decode(encode('hi there')))

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



def get_batch(split):
  data= train_data if split=='train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model=m
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y =  X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out
    


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out 
    


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non_lin   earity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)



class Block(nn.Module):
    """ Transformer block : communicates followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd : embedding dimension, n_head: the number of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    

    def forward(self, idx, targets=None):
        
        B,T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
         
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Getting the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmaxx to get probabilities
            probs = F.softmax(logits, dim=1)
            # samlpe from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Defining Optimizer
m = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)



# Trianing Loop
for steps in range(0):
    xb,yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps%eval_interval==0:
        losses = estimate_loss()
        print(f"steps {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
# print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))




def check_total_parameters(print_all_details=False):
    """This is code to check the total parameters of the model"""
    model=m
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {total_params:,}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    if print_all_details:
        for name, param in model.named_parameters():
            print(f"{name:40} {param.numel():,}")


check_total_parameters(print_all_details=False)
