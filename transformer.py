#tokenize our inputs
# aka turn the inputs into a sequence of integers/floats according to a vocabulary
import torch

# Experience hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3 #self-attention can't tolerate a high lr
eval_iters = 200
n_embed = 32

with open('input.txt','r', encoding = 'utf-8') as f :
    text = f.read()

print("lenght of dataset in characters : ", len(text))

chars = sorted(list(set(text)))

vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Tokenizing
stoi = { ch:i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s] # str -> list[int]
decode = lambda l : ''.join([itos[i] for i in l]) # list[int] -> str

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

print(data[:1000])


# Train / Validation splitt of the dataset
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

train_data[:block_size+1] #targets for each position

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    """ Done to make the transformer network be used to see context, all the way from as little as one to mny"""
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target : {target}")

# Creating the dataloaders | to allow us to process multiple chunks batches at the same time

torch.manual_seed(1337)
# batch_size =4 # n fo independent sequences will we process in parallel
# block_size = 8 # max-context len for preds

def get_batch(split : str):
    #generate a small batch of data of inputs x an targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

xb, yb = get_batch('train')

print('inputs: ')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)
print(yb)

print('---------------')

for b in range (batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target : {target}")

class Evaluator:
  """
  Class that will output the evaluation @ every given interval in the parameters:

  Params:
    model : nn.Module ; the model itself (should i pass the params?)
    eval_every : int ; the frequency at which we will do our evaluation
  """
  def __init__(self, model, eval_every : int, eval_iters : int):
    self.model = model
    self.eval_every = eval_every
    self.eval_iters = eval_iters

  @torch.no_grad()
  def estimate_loss(self):
    """ Averages the loss over multiple batches, so we don't get a very noisy loss, it also uses the loss returned by the model (at prediction)
    """
    out = {}
    self.model.eval()
    for split in ['train', 'val']:
      losses = torch.zeros(self.eval_iters)
      for k in range(self.eval_iters):
        X, Y = get_batch(split)
        _, loss = self.model(X,Y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    self.model.train()

    return out


# =============
# NN - Bigram Language model
# ==============

from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_embed),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.net(x)


class BigramLanguageModel(nn.Module):
  """ Bigram as in pair of two, the model only every looks at pairs liek the current and the next character; meaning that it doesnt have this notion of context of the previous tokens; aka no memory"""

  def __init__(self):
    super().__init__()
    # each token reads of the logits for the next token from a lookup table  -  aka a embedding table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    # pass from a single comm channel to 4 ;
    self.sa_heads = MultiHeadAttention(4, n_embed//4) # we have 4 heads from each C we will get 8 overall after cat we will get 32 C
    self.lm_head = nn.Linear(n_embed, vocab_size)


  def forward(self, idx, targets = None):
    B,T = idx.shape
    #idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) #(B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T)) #integers from 0 to T-1 that will be embedded through the table -> (T,C)
    x = tok_emb + pos_emb #(B,T, C) holds the token id + positions
    x = self.sa_heads(x) # apply the head of attention
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None :
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # stretch out the array
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      #crop the idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      logits , _ = self(idx_cond)
      # focus on the last time step
      logits = logits[:, -1, :] # becomes (B,C)
      # apply softmax to get P
      probs = F.softmax(logits, dim=-1)
      # sample from distrib, aka the vocab in the distrib over the vocabulary | aka roll a dice
      # to see whats the next char
      idx_next = torch.multinomial(probs, num_samples= 1) #(B,1)
      #append
      idx = torch.cat((idx,idx_next), dim=1) # (B, T+1)
    return idx

class Head(nn.Module):
  def __init__(self, head_size) -> None:
     super().__init__()
     self.key = nn.Linear(n_embed, head_size, bias = False)
     self.query = nn.Linear(n_embed, head_size, bias = False)
     self.value = nn.Linear(n_embed, head_size, bias = False)

     self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    # compute the attention scores
    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)

    out = wei @ v # (B, T ,T ) @ (B, T, C) -> (BTC)
    return out

class MultiHeadAttention(nn.Module):
  """
  Multiple head of self-attentions running in parallel
  """

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))

  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim = -1) # cat over the channel dimension

model = BigramLanguageModel()
logits, loss = model(xb,yb)
print(logits.shape)
print(loss)

# Inital generation to see the model before training;
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens= 100)[0].tolist()))

# ============
# training loop \ cmmon?
# ==========

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
evaluator = Evaluator(model, eval_interval, eval_iters)
for iter in range(max_iters):

  # at every 'eval_interval' : evaluate the loss on both sets
  if iter % eval_interval == 0:
    losses = evaluator.estimate_loss()
    print(f"Step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # get a batch of data
  xb, yb = get_batch('train')

  # eval the loss
  logits, loss = model(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

# Generate from the model
context = torch.zeros((1,1), dtype = torch.long)
print(decode(model.generate(context, max_new_tokens= 500)[0].tolist()))
