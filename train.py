import torch
import torch.nn as nn
from torch.nn import functional as F

# let's build a GPT from scratch!
# following Andrej Karpathy's video tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1299s

# the tiny shakespeare dataset is roughly 40,000 lines of text, taken from a variety of Shakespeare's plays
with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))
# -> 1115389

print(text[:1000])

chars = sorted(list(set(text)))
print(''.join(chars))
# ->  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

vocab_size = len(chars)
print(vocab_size)
# -> 65

# create a tokenizer to encode the input chars as an array of integers (and vice versa)

# string to integer
stoi = {ch: i for i, ch in enumerate(chars)}
# integer to string
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hello world!'))
print(decode(encode('hello world!')))
# -> [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42, 2]
# -> hello world!

# encode the entire tiny shakespeare dataset and wrap it in a PyTorch tensor:
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# split the data into a training set (90%) and a validation set (10%)
# used to help us understand the extent to which our network is overfitting
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# let's start training our transformer
# we don't want to always feed in the full dataset at once
# instead, lets train it on randomly sampled chunks of the data, one at a time

# define a max length for these randomly sampled chunks
block_size = 8

chunk = train_data[:block_size+1]
print(chunk)

x = train_data[:block_size]
y = train_data[1:block_size+1]

# block size also sets the max context length: how many input chars can our network use to produce output?
# the range will be 1 -> block_size
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

# in addition to defining a block size, let's define a batch size, which is the number of chunks that are
# fed into the network at once. this allows it to process multiple blocks in parallel (for efficiency reasons)

# just setting a manual seed for the random number generator inside torch
torch.manual_seed(1337)

batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generate batch_size number random indices into the data set
    ix = torch.randint(len(data - block_size), (batch_size,))
    # stack the input chunks into a tensor (4 x 8)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # stack the output chunks into a tensor (4 x 8)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print('outputs:')
print(yb.shape)
print(yb)
print('-------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()}, the target is {target}")

# now we have our transformer, let's start building a neural network!


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
