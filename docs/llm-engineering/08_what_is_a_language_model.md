# What Is a Language Model, Really

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 1 — Foundations. Builds on
[Chapter 7](07_history_how_we_got_here.md)'s history — this chapter is the precise
mechanism behind the "next-token prediction" objective named there.

## In Plain English

A language model is a machine that has learned to play one very specific game extremely
well: given some text, guess what word (or word-piece) comes next. That's it. It doesn't
"know facts" the way a database does, it doesn't "understand" in the way a person does —
it has learned an extremely good statistical sense of what text is likely to follow other
text, from having seen enormous amounts of real text. Everything an LLM appears to do —
answer questions, write code, hold a conversation — emerges from repeatedly playing this
one guessing game, one token at a time.

## The First-Principles Explanation

### The core objective: P(next token | everything so far)

Formally, a language model learns a probability distribution:

```
P(token_t | token_1, token_2, ..., token_{t-1})
```

In words: given every token that came before position `t`, what's the probability of
each possible next token? A vocabulary might contain 50,000+ possible tokens (see
[Chapter 9](09_tokenization.md)), and the model's job at every position is to output a
probability for *every single one of them* — a full probability distribution, not just
one guess.

### Autoregressive generation: the loop that builds text one token at a time

"Autoregressive" means the model's own past outputs become part of its future input.
Generation is a loop:

```
1. Start with a prompt (some initial tokens)
2. Ask the model: given these tokens, what's the probability of each possible next token?
3. Pick one token (how you pick — greedy, sampling — matters a lot; see below)
4. Append it to the sequence
5. Go back to step 2, now with one more token of context
6. Repeat until done
```

This loop, and nothing more, is how every word of every LLM response gets produced — one
token, chosen from a probability distribution, appended, repeat.

### Where does the probability distribution actually come from?

The Transformer architecture (full mechanism in [Chapter 10](10_transformer_architecture.md))
processes the input tokens and produces, for the last position, a vector of raw scores
called **logits** — one score per vocabulary token, higher meaning "more likely." A
**softmax** function converts these raw scores into an actual probability distribution
(all values between 0 and 1, summing to 1):

```
probability_i = exp(logit_i) / sum(exp(logit_j) for all j in vocabulary)
```

### How a token actually gets chosen: greedy vs. sampling

Given the probability distribution, there are genuinely different strategies for picking
the next token, each with real trade-offs:

- **Greedy decoding** — always pick the single highest-probability token. Deterministic
  (same prompt always produces the same output), but often repetitive, since it never
  takes a "reasonable but not top" option.
- **Sampling** — pick a token *randomly*, weighted by the probability distribution
  (higher-probability tokens are more likely to be picked, but not guaranteed). Produces
  more varied, natural-sounding text, at the cost of losing determinism.
- **Temperature** — a knob that reshapes the distribution before sampling: dividing
  logits by a temperature `< 1` makes the distribution sharper (closer to greedy, more
  confident/repetitive); `> 1` flattens it (more random, more diverse, more likely to go
  off the rails).
- **Top-k / top-p (nucleus) sampling** — restrict sampling to only the k highest-
  probability tokens (top-k), or the smallest set of tokens whose cumulative probability
  exceeds p (top-p) — both exist to avoid sampling from the "long tail" of very
  low-probability, often nonsensical tokens.

## Grounded in This Repo's Code

Every concept above is directly implemented, in a fully readable way, in
[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py):

```python
# The core generation loop (simplified from the actual generate() function, ~line 338)
for _ in range(max_new_tokens):
    window = ids[:, -ctx_len:]          # the "everything so far" context
    logits = model(window)              # raw scores, one per vocab token, for every position
    next_logits = logits[:, -1, :]      # we only care about the LAST position's prediction
    if do_sample:
        next_token = sample_next_token(next_logits)   # sampling, defined below
    else:
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # greedy
    ids = torch.cat([ids, next_token], dim=1)   # autoregressive: append, then repeat
```

And `sample_next_token` (line 197) implements temperature, top-k, *and* top-p exactly as
described above, in about 15 lines:

```python
def sample_next_token(logits, temperature=0.9, top_k=40, top_p=0.95):
    logits = logits / temperature          # temperature reshaping
    vals, idx = torch.topk(logits, top_k, dim=-1)    # top-k restriction
    probs = torch.softmax(vals, dim=-1)              # softmax: logits -> probabilities
    # ... top-p (nucleus) filtering on top of that ...
    chosen = torch.multinomial(probs, num_samples=1) # weighted random sampling
    return idx.gather(-1, chosen)
```

There's also a fourth technique in the same file worth naming: `apply_repetition_penalty`
(line 215) — it reduces the probability of tokens that already appeared recently in the
generated sequence, a practical fix for a very real problem (small, undertrained models
especially tend to loop) that isn't part of the "pure" theory above but matters a lot in
practice.

## Deep-Dive: Why "Just Predicting the Next Word" Produces Such Varied Behavior

This is the single most counterintuitive fact about LLMs, worth sitting with: a model
trained on *only* "predict the next token" ends up able to answer questions, write code,
and reason through problems — none of which were explicitly labeled as separate tasks
during training. The explanation: to predict the next token *well* across a large,
diverse corpus of real text (which includes questions being answered, code being written,
problems being reasoned through), the model has to implicitly learn the *patterns*
underlying all of those activities — predicting the next token of a working code snippet
requires something that behaves like understanding code; predicting the next token of a
well-reasoned argument requires something that behaves like following logic. The single
objective is simple; what it takes to get *good* at that objective, at scale, is not.

## Try It Yourself

- In [`from_scratch/custom-gpt-153m/`](../../from_scratch/custom-gpt-153m/), after
  training (even briefly) via `./scripts/workflow.sh train`, run
  `./scripts/workflow.sh infer` and watch the model generate text one token at a time
  conceptually — every word that appears is the result of one pass through the loop
  above.
- Edit `sample_next_token`'s default `temperature` value in `tiny_llm.py`, re-run
  inference, and compare outputs — a lower temperature should feel more repetitive/
  conservative, a higher one more chaotic. This turns "temperature reshapes the
  distribution" from an abstract claim into something you directly observe.

## Common Misconceptions

- **"The model looks up an answer."** There's no lookup — every token is generated fresh
  from a probability distribution computed on the fly; there's no database of facts being
  queried, which is also the root explanation for why models can confidently generate
  incorrect information (hallucination) — it's still just predicting plausible-sounding
  next tokens, with no built-in mechanism to check truth.
- **"Greedy decoding is 'more correct' than sampling."** Neither is inherently more
  correct — they're different trade-offs. Greedy is deterministic and often more focused
  but can loop/repeat; sampling is more varied but non-deterministic. Production systems
  often use sampling with carefully tuned temperature/top-p specifically because
  greedy's repetitiveness is a worse user experience than controlled randomness.
- **"Temperature 0 and greedy decoding are different things."** They're mathematically
  equivalent in the limit — dividing by a temperature approaching 0 makes the
  distribution collapse entirely onto the single highest-probability token, which is
  exactly what greedy/argmax picks directly.

## Practice Questions

1. Why does a language model need to output a full probability distribution over the
   *entire* vocabulary at every position, rather than just its single best guess?
2. Explain, in terms of the autoregressive loop, why a language model has no way to
   "revise" an earlier token once it's been generated and appended.
3. A model set to `temperature=2.0` starts producing incoherent text. Explain what's
   happening to the probability distribution that causes this.

## Key Terms

- **Logits**: raw, unnormalized scores output by the model, one per vocabulary token,
  before softmax converts them into probabilities.
- **Softmax**: the function that converts a vector of raw scores into a valid probability
  distribution (all values in [0,1], summing to 1).
- **Autoregressive**: a generation process where each new output becomes part of the
  input for the next step.
- **Greedy decoding**: always selecting the single highest-probability next token.
- **Sampling**: selecting the next token randomly, weighted by its probability.
- **Temperature**: a scaling factor applied to logits before sampling, controlling how
  sharp or flat the resulting probability distribution is.
- **Top-k / top-p (nucleus) sampling**: restricting sampling to a subset of the most
  likely tokens, to avoid sampling from the unreliable long tail of the distribution.
- **Hallucination**: a model generating fluent, plausible-sounding but factually
  incorrect text — a direct consequence of there being no built-in fact-checking
  mechanism in the next-token-prediction objective.
