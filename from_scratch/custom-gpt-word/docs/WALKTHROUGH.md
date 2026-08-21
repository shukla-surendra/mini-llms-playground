# From words to generated words

`custom-gpt-word` teaches one job: predict the next token. Suppose a sampled corpus
window is:

```text
x: the cat sat on the mat
y: cat sat on the mat .
```

The target is `x` shifted left by one. At each position the model receives all tokens
up to that position and is scored against the token that followed it in the corpus.
It learns six predictions in this one short example, not only the final `.`.

## What `make dry-run` proves

The dry-run creates exactly the same `TextData`, `GPT`, and `AdamW` objects as training,
then executes these operations once:

```text
corpus -> WordTokenizer -> ids -> x/y batch
                                  |
                                  v
             GPT(x) -> logits -> cross_entropy(logits, y) -> loss
                                                          |
                                                          v
                                                   loss.backward()
                                                          |
                                                          v
                                                   optimizer.step()
```

The printed shapes have the following meanings:

| Shape | Meaning |
|---|---|
| `x: (16, 12)` | 16 examples, each containing 12 integer token ids |
| `y: (16, 12)` | the correct next ids for all 192 input positions |
| `logits: (16, 12, V)` | a score for each of the `V` vocabulary tokens at each position |

For a perfectly uniform guess the loss is `log(V)`; an untrained neural network's
random logits can start above or below that reference. The loss should fall over a full training
run. A non-zero embedding gradient norm and a changed weight show that backpropagation
and the optimizer update both occurred. The process then terminates, discarding its
in-memory model, so no `checkpoints/` directory is written.

## Inside one Transformer block

`model.py` takes embedding vectors shaped `(B, T, C)`, where `B` is batch size, `T` is
tokens in the context window, and `C` is embedding width.

1. `qkv(x)` produces Queries, Keys, and Values. Each asks what a position needs,
   describes what it contains, or carries the information to transfer.
2. `q @ k.transpose(...)` gives every earlier/later token pair a compatibility score.
3. The lower-triangular mask changes all future scores to negative infinity. Softmax
   makes those weights exactly zero, so the model cannot see the answer token.
4. `softmax(scores) @ v` mixes previous Value vectors according to the learned weights.
5. A residual connection adds that result back to the input; a second residual adds an
   MLP transformation. Three blocks refine the representation three times.

Finally, `lm_head` turns each final vector into one logit per vocabulary item. Cross
entropy rewards high logits for the observed next token and penalizes the others.

## Why generation is a loop

At training time the model predicts every next token in parallel because the target
sequence is already known. At generation time only the prompt is known. The loop is:

1. Run the current prompt through GPT.
2. Keep the final position's vocabulary logits.
3. Divide by temperature, softmax, and randomly sample one id.
4. Append it to the prompt and repeat.

Once the prompt exceeds 12 tokens, `generate.py` supplies only its most recent 12 ids
to the model. That is the visible version of a context-window limit. A real serving
system also uses a KV cache to avoid recomputing earlier attention work; this learning
project intentionally leaves that optimization out.
