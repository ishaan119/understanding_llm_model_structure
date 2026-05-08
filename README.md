# What Actually Happens When You Talk to a Language Model

I wanted to explore what exactly does a model consists of how does the flow really work. So I went to hugging face and started exploring the 
qwen model. https://huggingface.co/Qwen/Qwen2.5-1.5B

I saw a model card and lots of files and got curious even though at high level I understand how things works what exactly are the files and get into the details i.e  between "I type a question" and "it gives me an answer" — what are the moving parts?

So I opened the files that ship with the model and traced the path a sentence takes from the moment you type it to the moment the model produces a response. This post is that trace.

---

## Where Does a Model Even Live?

When you run:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

...it downloads files to a shared cache at `~/.cache/huggingface/hub/`. Not in your project folder. This is deliberate — multiple projects can share the same 3 GB of weights without re-downloading them every time.

Inside that cache, the model is just **7 files**. That's it. Seven files and you have a working language model:

| File | What it is | Size |
|------|-----------|------|
| `config.json` | Architecture blueprint | tiny |
| `model.safetensors` | **All the learned weights** | **~3 GB** |
| `generation_config.json` | Default generation settings | tiny |
| `tokenizer_config.json` | Tokenizer metadata + chat template | small |
| `tokenizer.json` | Compiled tokenizer (fast version) | ~7 MB |
| `vocab.json` | Vocabulary — 151,665 token↔ID mappings | ~2.8 MB |
| `merges.txt` | BPE merge rules | ~1.7 MB |

Here's what `config.json` actually looks like (the architecture blueprint):

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "hidden_size": 1536,
  "num_hidden_layers": 28,
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "intermediate_size": 8960,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16"
}
```

And `generation_config.json` (default generation settings):

```json
{
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.1
}
```

Rather than go through the files one by one (which is how I first approached it and got confused), it's easier to understand them by following the path a sentence takes through the model.

---

## The Question That Organizes Everything

Say I type:

> "I am anxious about tomorrow."

The model responds with something. What happened in between?

```
"I am anxious about tomorrow."
      │
      ▼  Step 1: TOKENIZE — split text into pieces, convert to integer IDs
      │
      ▼  Step 2: EMBED — turn each ID into a 1,536-dim vector
      │
      ▼  Step 3: TRANSFORM — pass vectors through 28 layers
      │
      ▼  Step 4: PREDICT — produce probabilities over the next token
      │
      ▼  Step 5: REPEAT — pick a token, append it, go back to Step 1
```

Each of these steps involves a different piece of the model. Let me walk through them.

---

## Step 1: Tokenize — Text Becomes Integers

The model does not see text. It sees integers. The tokenizer is what converts between them.

```
"I am anxious about tomorrow."
         │
         ▼
    [40, 1079, 37000, 911, 11424, 13]
```

Six integers. Same input → same output, every time. No neural network involved. It's a deterministic system that lives *outside* the model.

It does this in two sub-steps.

### 1a. Split the text into subword pieces

The tokenizer doesn't work at the word level. It works at the *subword* level — pieces of words. "Tomorrow" might be one piece. "Stoicism" might be three. "你好" might be one.

It decides how to split using rules stored in `merges.txt`. The file has 151,387 rules, and each rule says "merge these two pieces into one." Here's what the actual file looks like — the first 15 lines:

```
Ġ Ġ
ĠĠ ĠĠ
i n
Ġ t
ĠĠĠĠ ĠĠĠĠ
e r
ĠĠ Ġ
o n
Ġ a
r e
a t
s t
e n
o r
Ġt h
```

Each line is one rule. `Ġ` represents a space. So `Ġ Ġ` means "merge space + space into a double-space token." `i n` means "merge 'i' + 'n' into 'in'." `ĠĠĠĠ ĠĠĠĠ` means "merge four-space + four-space into eight-space" — Python indentation.

The rules are in priority order. Rule 1 gets applied first, rule 151,387 last.

### How were these rules created?

This is the part that clicked for me. The rules weren't hand-written. They were **learned by a simple algorithm** called BPE (Byte Pair Encoding). Here's how it works:

1. Start with a corpus of text (billions of words).
2. Split everything into individual characters: `['h', 'e', 'l', 'l', 'o']`
3. Count every adjacent pair. Find the most common one. Say it's `('l', 'l')`.
4. Merge that pair everywhere in the corpus. Now `"hello"` is `['h', 'e', 'll', 'o']`.
5. Write down the rule: `l l` (this becomes line 1 of merges.txt).
6. Repeat from step 3. Next most common pair might be `('h', 'e')` → merge → write rule.
7. Do this 151,387 times. Stop.

That's it. The algorithm just keeps finding the most frequent pair and merging it. Common words like "the" get merged all the way into a single token early on. Rare words like "prosoche" never fully merge — they stay as fragments.

The engineers at Alibaba ran this on their training corpus once, saved the rules to `merges.txt`, and that file never changes again. When you use the tokenizer, it just applies these rules — no learning happens at inference time.

### How tokenization works on a word

When the tokenizer sees `"should"`:

1. Start as individual characters: `['s', 'h', 'o', 'u', 'l', 'd']`
2. Scan the rules top to bottom. First matching rule: `s h` → apply: `['sh', 'o', 'u', 'l', 'd']`
3. Next match: `o u` → `['sh', 'ou', 'l', 'd']`
4. Next match: `l d` → `['sh', 'ou', 'ld']`
5. Next match: `ou ld` → `['sh', 'ould']`
6. Next match: `sh ould` → `['should']`
7. Done — one token.

Same input always gives same output. Deterministic.

### 1b. Look up each piece in vocab.json

Once the text is split into pieces, each piece gets mapped to an integer ID using `vocab.json`. Here's a sample of what's actually in that file:

```json
{
  "!": 0,
  ".": 13,
  "I": 40,
  "Ġthe": 279,
  "Ġof": 315,
  "Ġam": 1079,
  "ism": 2142,
  "Ġanxious": 37000,
  "ĠMarcus": 35683
}
```

(The real file has 151,643 entries. These are just a few I pulled out.)

Notice: `Ġam` means "space + am" — the space is part of the token. `ĠMarcus` means "space + Marcus." This is how the tokenizer handles word boundaries — the space before a word is glued to the word itself.

### How was vocab.json created?

It's the output of the same BPE process. Every time the algorithm merges a pair, the result becomes a new vocabulary entry. After 151,387 merges, you have 151,643 tokens (256 base bytes + 151,387 merged tokens). Each gets assigned an integer ID. Save to JSON. Done.

So `merges.txt` and `vocab.json` are two views of the same process:
- `merges.txt` = the rules (how to split)
- `vocab.json` = the results (what the pieces are called)

Qwen's vocabulary is 151,665 entries because it needs to cover English, Chinese, Japanese, Korean, and code. Other models have smaller vocabularies:

| Model | Vocab size |
|-------|-----------|
| Qwen2.5 | 151k |
| LLaMA 3 | 128k |
| GPT-4 | 100k |
| Mistral | 32k |

The size of the vocabulary shapes what the model is *efficient* at. "你好" is one token in Qwen. In a 32k English-centric tokenizer, it would be 4-6 bytes. Meanwhile, a rare word like "Stoicism" splits into three tokens in Qwen (`['St', 'o', 'icism']`) because it wasn't common enough in the training data to deserve its own token.

### What I verified

I wrote a small script to confirm this works the way I thought:

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

tok.tokenize("anxious")        # → [' anxious']
tok.tokenize("Stoicism")       # → ['St', 'o', 'icism']
tok.tokenize("memento mori")   # → [' m', 'emento', ' mor', 'i']
tok.tokenize("你好")            # → ['你好']
```

**What the tokenizer is, in one sentence:** a deterministic function that turns text into integers using a learned vocabulary. It lives outside the model. It never changes.

---

## Step 2: Embed — Integers Become Vectors

Now we have a list of integer IDs: `[40, 1079, 37000, 911, 11424, 13]`. But the transformer layers inside the model work on vectors (lists of floating-point numbers), not integers. So we need to convert each ID into a vector.

This is the job of the **embedding layer**, which is the very first layer of the neural network. And here's the thing that took me a while to really accept:

The embedding layer is not a function. It's just a matrix.

Specifically, a matrix of shape `[151,936 × 1,536]` — one row for every possible token in the vocabulary, and each row is a 1,536-dimensional vector. "Embedding" a token means:

```
Token ID 40 → row 40 of the matrix → a 1,536-dim vector
```

Here's what that actually looks like. These are real numbers from the model:

```
Embedding matrix shape: (151936, 1536)

Row 40 ("I"), first 8 of 1536 dims:
  [+0.0286, +0.0003, -0.0123, +0.0439, -0.0405, -0.0060, +0.0008, +0.0126]

Row 1079 (" am"), first 8 of 1536 dims:
  [+0.0159, +0.0063, +0.0212, +0.0068, +0.0134, +0.0305, +0.0009, -0.0029]

Row 37000 (" anxious"), first 8 of 1536 dims:
  [-0.0063, +0.0227, -0.0454, +0.0315, -0.0243, -0.0075, +0.0055, +0.0197]
```

Each row is 1,536 numbers like these. The full matrix has 151,936 such rows. That's 151,936 × 1,536 = **233 million numbers** just for the embedding layer — about 15% of the model's total parameters are in this one lookup table.

The numbers themselves look random, but they're not. During pre-training, the model learned to arrange them so that words with similar meanings have similar vectors. You can't see that from 8 numbers, but across all 1,536 dimensions, "happy" and "joyful" would be close together, "king" and "queen" would be close together.

That's it. Row indexing. No computation, no nonlinearity, no "neural net doing its thing." Just: here's an integer, here's the row with that index, return it.

I wanted to be sure of this, so I checked:

```python
embedding_matrix = model.get_input_embeddings().weight   # shape [151936, 1536]
ids = torch.tensor([[40, 1079, 37000]])

via_layer  = model.get_input_embeddings()(ids)   # the "official" way
via_manual = embedding_matrix[ids]                # just row-indexing

torch.equal(via_layer, via_manual)
# True
```

They're bit-identical. The embedding layer is just `matrix[ids]`.

So where's the intelligence? It's in what the rows *contain*. During pre-training, the model learned to arrange the rows so that semantically similar tokens ended up with similar vectors. "King" and "queen" are near each other. "Paris" and "France" are near each other. The lookup is trivial; the arrangement is everything.

### A way to think about it

Imagine a dictionary. It has two parts:
- **Page numbers** — determined by alphabetical order.
- **Definitions** — what's printed on each page.

The tokenizer is the system that tells you page numbers. "Given the word 'anxious', what page is it on?" → Page 37,000.

The embedding matrix is the definitions. "Turn to page 37,000. Read what's there." → A 1,536-number fingerprint that captures what "anxious" means.

Both were compiled once and saved. The page numbers (vocabulary) are strictly fixed. The definitions (embedding vectors) are part of the model's weights — they were learned during pre-training.

---

## Step 3: Transform — 28 Layers of Attention and MLPs

Now we have 6 vectors (one per token), each 1,536 dimensions. These get passed through the main body of the model: **28 transformer layers stacked on top of each other.**

Each layer does the same two things:

### Self-attention

Each token "looks at" the other tokens in the sequence and decides which ones are relevant to it. When processing the token for "anxious," attention lets it pay strong attention to "tomorrow" (because that's what the anxiety is about) and maybe weak attention to "I" (less informative).

This is what lets language models understand context. A word's meaning depends on the words around it, and attention is the mechanism that lets each word be influenced by the others.

Inside each attention block are four weight matrices. Here are the actual shapes from layer 0 of our model:

```
model.layers.0.self_attn.q_proj.weight  →  (1536, 1536)  = 2.4 million numbers
model.layers.0.self_attn.k_proj.weight  →  (256, 1536)   = 393k numbers
model.layers.0.self_attn.v_proj.weight  →  (256, 1536)   = 393k numbers
model.layers.0.self_attn.o_proj.weight  →  (1536, 1536)  = 2.4 million numbers
```

These are the "query", "key", "value", and "output" projections:
- **q_proj** — "what am I looking for?"
- **k_proj** — "what do I contain?"
- **v_proj** — "what do I offer?"
- **o_proj** — "merge the attention results back"

I'm not going to derive the attention math here — entire textbooks exist for that. What matters for our purposes is that each layer has these four matrices, and 28 layers × 4 matrices = **112 attention matrices** in total.

### MLP (feed-forward network)

After attention, each token's vector goes through an MLP — a small neural network that transforms the representation. Here are the actual shapes:

```
model.layers.0.mlp.gate_proj.weight  →  (8960, 1536)  = 13.8 million numbers
model.layers.0.mlp.up_proj.weight    →  (8960, 1536)  = 13.8 million numbers
model.layers.0.mlp.down_proj.weight  →  (1536, 8960)  = 13.8 million numbers
```

The MLP expands the 1,536-dim vector to 8,960 dimensions (more room to compute), applies a nonlinearity, and projects back down to 1,536. Three matrices, about 41 million parameters per layer just for the MLP.

### Stack 28 of these

Each layer refines the representation a little further. By layer 28, each token's vector has been transformed from "what this word means in isolation" into "what this word means in the full context of the sentence." The deeper layers tend to encode more abstract concepts; the shallow layers handle more local patterns.

All of this is happening on matrices of numbers. The transformer never "understands" anything in the human sense. It just does billions of matrix multiplications, and out the other side comes something that looks an awful lot like language.

---

## Step 4: Predict — Vector Becomes a Probability Distribution

After 28 layers, we have a final vector for the last token in the sequence. This vector is supposed to contain all the information the model needs to guess what token comes next.

To turn this vector into an actual prediction, we multiply it by an **output matrix** that has one row per possible token:

```
final_vector (1,536 dims)  ×  output_matrix (1,536 × 151,936)  =  151,936 numbers
```

Each of those 151,936 numbers is a "logit" — a raw score for how likely that token is to come next. Apply a softmax, and you get a probability distribution. Sample from it (using settings like temperature and top-p, which live in `generation_config.json`), and you get the next token.

Here's a detail that surprised me when I learned it: **the output matrix is the same matrix as the embedding matrix, just transposed.** This is what `"tie_word_embeddings": true` means in the config. The model uses one matrix for two jobs:

- At the start, converting integer IDs → vectors (embedding)
- At the end, converting vectors → probabilities over IDs (prediction)

This is elegant. If "king" embeds to vector V, then a final vector close to V should predict "king" with high probability. Using the same matrix in both directions enforces that consistency — and saves 233 million parameters.

---

## Step 5: Loop

The model predicts one token, appends it to the sequence, and goes back to Step 1. Tokenize, embed, transform, predict, sample, repeat. This continues until the model produces a special `<|im_end|>` stop token or hits the max length.

Generating a response of 300 words takes roughly 400 loops of this process.

---

## The Special Tokens and the Chat Template

There's one more piece I need to cover: how the model knows who's talking.

Qwen reserves 22 token IDs (151643–151664) for structural markers — they're not words, they're signals. The important ones:

```
<|im_start|>   — "a new message is starting"
<|im_end|>     — "this message is done"
<|endoftext|>  — end of document / padding
```

Conversations get wrapped in these:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
I am anxious about tomorrow.<|im_end|>
<|im_start|>assistant
```

When the model sees `<|im_start|>assistant\n`, it knows "my turn to respond." When it generates `<|im_end|>`, it stops.

This format is baked into a Jinja2 template stored in `tokenizer_config.json`. You don't write it manually — you call:

```python
messages = [
    {"role": "user", "content": "I am anxious about tomorrow."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

...and it produces the properly-wrapped string. Then this string goes into Step 1 and the whole pipeline kicks off.

---

## The Files, Revisited

Now the seven files make sense. Each one has a specific role in the pipeline:

| File | What it's for |
|------|---------------|
| `merges.txt` | Step 1a: rules for splitting text into subword pieces |
| `vocab.json` | Step 1b: dictionary mapping pieces → integer IDs |
| `tokenizer.json` | Step 1: compiled fast version of the above (Rust) |
| `tokenizer_config.json` | Step 5: chat template + tokenizer settings |
| `model.safetensors` | Steps 2–4: **all** the learned weights (embedding matrix, 28 layers, output projection) |
| `config.json` | The model's shape — tells PyTorch how to instantiate it before loading weights |
| `generation_config.json` | Step 4: default sampling settings |

Three files for tokenization. One giant file for the neural network. Three small config files for everything else. That's a language model.

---

## What I Actually Understand Now

Before I looked at any of this, if you'd asked me "what is a language model?" I'd have said "a neural network trained on a lot of text." Technically true. Not useful.

Here's what I'd say now:

A language model is a stack of matrices that operates on integer sequences. A separate system — the tokenizer — converts text to integers on the way in, and integers back to text on the way out. The model's first layer does a lookup (embedding: integer → vector). Its last layer does a reverse lookup (prediction: vector → probability over integers). In between, 28 layers of attention and MLPs refine each token's representation based on the other tokens around it.

The tokenizer is deterministic and lives outside the model. It was built once and frozen. The model's weights — embedding matrix, attention projections, MLP weights — were all learned during pre-training on trillions of tokens. Together, they form a system that, given any prefix of text, can produce a surprisingly good guess at what comes next.

That's it. That's the whole thing.

The complexity isn't in the architecture — it's in the scale (1.54 billion parameters) and the training (trillions of tokens). The architecture itself is simple enough to fit on a napkin.
