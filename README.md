# micro micro micro gpt

A character-level language model written in PyTorch, built as a learning exercise around the
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper.

It's built in two steps:

1. **`bigram.py`**: a bigram language model. It's the simplest possible baseline, it predicts the next
   character from the current one only.
2. **`transformer.py`**: a **Transformer(decoder-only)** (the GPT-style part of the paper's
   architecture). It uses the self-attention mechanism so every character can look at the characters that came before it.

Both models are trained on the same data with the same training loop, so the difference in results
comes only from the architecture.

---

## Dataset & tokenization

- **Data:** `input.txt` is the *Tiny Shakespeare* dataset, ~1.1M characters.
- **Tokenizer:** character level. Each unique character in the text becomes a token, which gives a
  vocabulary of 65 tokens.
  - `encode("hi")` → `[46, 47]`, `decode([46, 47])` → `"hi"`
- **Batches:** `get_batch()` picks random chunks of `block_size` characters. The target `y` is the
  input `x` shifted one character to the right. One chunk therefore holds `block_size` training
  examples: predict char 2 from char 1, char 3 from chars 1–2, and so on. The model learns to predict
  with any amount of context, from 1 character up to `block_size`.

---

## Part 1: Bigram language model (`bigram.py`)

"Bigram" means a pair of two, the model only looks at the **current character** to guess the
**next one**. It has no memory of anything earlier in the sequence.

### How it works

The whole model is one `nn.Embedding(vocab_size, vocab_size)` lookup table:

```
token id -> row of the vocab_table (65 numbers) -> logits for the next character
```

- **forward:** look up the row for each input token -> logits `(B, T, 65)` -> cross-entropy against the targets.
- **generate:** start from a token, take the logits of the last position, apply softmax, sample the
  next character, append it, and repeat.

### Configuration

| batch_size | block_size | max_iters | learning rate | parameters |
|---|---|---|---|---|
| 4 | 8 | 3000 | 1e-3 (AdamW) | 4,225 (65 × 65) |

### Results

```
Step 0    : train loss 4.6470, val loss 4.6401
Step 900  : train loss 4.0542, val loss 4.0868
Step 1800 : train loss 3.5906, val loss 3.6351
Step 2700 : train loss 3.2670, val loss 3.3483
```

Sample output (Before training):

```
WhomngjOUCakGkREMyLEFqd s!fJ hHE.VFhe.'XAnyoain
BIqLo&$oMP$ gGCzPit.e releso.'IIU-JfooOho; and,Oy erarcug!QXDorm&OmKis ZKin-is
```

We can see that the model picks up a few common pairs ("Th", "he", " and,"), but with only one character of context it
can't really build words (or anything that makes sense). The tokens need to 'talk to each other', and that's what the Transformer adds.

---

## Part 2: Decoder-only Transformer (`transformer.py`)

The bigram lookup table becomes a stack of Transformer blocks. The main idea is **"communicate, then compute"**: tokens first exchange information through self-attention, then each token processes what it received on its own through a feed-forward network.

The original paper describes an encoder–decoder model that's used for translation, but this project only implements the decoder, like most modern LLMs do: there's no encoder and no cross-attention, only masked self-attention over the text generated so far.

### Architecture

```
add_a_funny_picture
```

### Things Learned;

- **Token + positional embeddings:** self-attention has no built-in sense of order, so that's why a learned
  position embedding is added to each token embedding. (so in this case words can have a different meaning depending on where they are)
- **`Head` (one self-attention head):**
  - each token produces a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what I'll share if you attend to me").
  - attention scores `q @ kᵀ` are scaled to keep the softmax from one-hot'ing the tokens.
  - a **causal mask** (which is the lower-triangular matrix `tril`) sets future positions to `-inf`, so a token can only look at the past tokens, not the future ones.
  - the result after softmax is used to take a weighted sum of the values.
- **`MultiHeadAttention`:** `n_head` heads run in parallel, each with `head_size = n_embed / n_head`.
  Their outputs are concatenated and passed through a linear projection.
- **`FeedForward`:** `Linear(n_embed -> 4·n_embed) -> ReLU -> Linear(4·n_embed _> n_embed)`, with the 4x inner size taken from the paper.
- **`Block`:** attention + feed-forward, each wrapped in a **residual connection**
  (`x = x + f(x)`), which keeps gradients flowing through a deep network.
- **Generation:** same as the bigram, but the context is cropped to the last `block_size`
  tokens, because the position table only has `block_size` entries.

### Differences from the paper

- **Decoder only**, no encoder or cross-attention.
- **Pre-LayerNorm:** LayerNorm is applied *before* attention and feed-forward, not after the
  residual add as in the original paper. This is the modern GPT-2-style layout and trains more
  stably. There's also a final LayerNorm before the output layer.
- **Dropout** (set to 20%) on the attention weights and after the feed-forward network, to reduce overfitting.
- **Learned positional embeddings** instead of the paper's sinusoidal ones.

### Configuration

| n_embed | n_head | n_layer | block_size | batch_size | dropout | max_iters | learning rate | dataset |
|---|---|---|---|---|---|---|---|---|
| 384 | 6 | 6 | 256 | 64 | 0.2 | 5000 | 1e-3 (AdamW) | Tiny Shakespeare |

This configuration has **~10.8M parameters** (10,788,929). Training takes a while, especially
on CPU.

### Results

Initial loss and an untrained sample (random characters):

```
tensor(4.2283, grad_fn=<NllLossBackward0>)

nsYsNmPaRl:$:eSMNMdS:jcNg!SfzcCbhjpJ JdQ:i!B;!yLEfRWlI
seDP,zRY
YX AhDwHbjFg,,hE$bRuQTM'A:sSSb&OT!N$
```

Training:

```
Step 0 : train loss 4.2174, val loss 4.2155
Step 300 : train loss 2.1874, val loss 2.2208
Step 600 : train loss 2.0519, val loss 2.1295
Step 900 : train loss 1.9798, val loss 2.0980
Step 1200 : train loss 1.9400, val loss 2.0642
Step 1500 : train loss 1.9077, val loss 2.0269
Step 1800 : train loss 1.8710, val loss 1.9847
Step 2100 : train loss 1.8628, val loss 1.9851
Step 2400 : train loss 1.8417, val loss 1.9937
Step 2700 : train loss 1.8273, val loss 1.9637
Step 3000 : train loss 1.7982, val loss 1.9479
Step 3300 : train loss 1.7953, val loss 1.9467
Step 3600 : train loss 1.7752, val loss 1.9342
Step 3900 : train loss 1.7727, val loss 1.9309
Step 4200 : train loss 1.7706, val loss 1.9165
Step 4500 : train loss 1.7626, val loss 1.8943
Step 4800 : train loss 1.7528, val loss 1.9026
```

Sample after training:

```
First theit won from have this quickers,
When do madan his life, bid.

CLARENCE

ESCROKE:
Comil that I was you.

LUCIUS:
What's more:
Lies aper yet, once,
By Rome forlar.

WARWICK:
Aways,
To that less and heaver, letter of the so? he stosmen, if then!

First Romea, he keeps this did fape, have she
will poition.

DUKE VINCENTIO:
Things a juwas hope?

DUKE VINCE.

SAMP ELIA:
Warrail, Capulian,
If to greast, baline's lad cheere's in this wifes acbrow-lible.

HESRY VI

ANGEL:
You, what keep'd theys
```

The output is still far from a real Shakespeare play, but it's a big improvement over the gibberish output the previous model gave
(the bigram). It has learned the structure of a play: character names in capitals followed by `:`, line breaks, punctuation, real words ("life", "Rome", "hope", "letter") and some phrases that almost make sense.

---

## Bigram vs Transformer

| | Bigram | Transformer |
|---|---|---|
| Context | 1 character | up to 256 characters |
| Parameters | ~4K | ~10.8M |
| Final val loss | ~3.35 | ~1.90 |
| Output | random-looking characters | play-like structure, real words |

---

## Running

Requires Python and PyTorch.

```bash
uv pip install torch    # or pip add torch
python bigram.py        # fast, runs in seconds
python transformer.py   # slow on CPU
```

Both scripts print the dataset info, a sample from the untrained model, the train/val loss every
300 steps, and a 500-character sample at the end.

---

## Next steps

- **KV cache** to speed up generation: reuse the keys and values of past tokens instead of
  recomputing attention over the whole context for every new character.
- GPU support (`.to(device)`) so the bigger configuration trains in reasonable time.
- A sub-word tokenizer (e.g. BPE) instead of single characters (like other models use e.g ChatGPT).
- Save and load model checkpoints.

---

## Conclusion

Overall it was a very interesting project. It took some time to learn it all, but I'm very glad I
took it on, it wasn't easy to learn it all as it had a lot of new things but despite all of that I've learned a lot.

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
