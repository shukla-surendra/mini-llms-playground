# Continual Training Across Many Datasets, No GPU Required

Companion to [`CONTINUING_TRAINING_ON_NEW_DATA.md`](CONTINUING_TRAINING_ON_NEW_DATA.md)
(which covers switching to *one* new dataset) — this doc is for the broader goal: **keep
training the same model, indefinitely, across a growing sequence of different datasets,
on a machine with no GPU**, without your setup getting harder to manage as the number of
datasets grows.

## Why this is actually a good fit for this project's constraints, not a limitation

This model is ~5.85M parameters ([`ARCHITECTURE.md`](ARCHITECTURE.md)), and real,
observed MPS throughput is ~5-6 steps/second ([`TRAINING.md`](TRAINING.md#throughput-on-apple-silicon-mps))
— a 1,000-step training round takes roughly 3-4 minutes. **Small, frequent training
rounds on modest hardware is exactly what this model size is suited for** — you don't
need a GPU to add a new dataset round every so often; you need a workflow that doesn't
degrade as rounds accumulate. That's what this doc is.

## The two real problems that show up specifically when you keep doing this repeatedly

1. **Forgetting compounds across rounds.** [`CONTINUING_TRAINING_ON_NEW_DATA.md`](CONTINUING_TRAINING_ON_NEW_DATA.md#does-it-retain-previous-knowledge-yes-partially--its-a-spectrum-not-yesno)
   covers forgetting for a single old→new transition. Across *many* rounds, simply mixing
   "the immediately previous dataset" into each new round isn't enough — round 3 could
   still gradually erode round 1's patterns if round 1 is never revisited again, several
   rounds later.
2. **No way to roll back a bad round.** Every training call writes to the same fixed
   checkpoint filenames. If dataset round 4 turns out to hurt quality more than it helps
   (a real risk — not every dataset addition is guaranteed to be net-positive), there was
   previously no way to get back to "the model as it was after round 3" without having
   manually copied files yourself.

Both are now addressed with two small additions to this project.

## Tool #1: `scripts/build_replay_mix.py` — mixing in old data without re-tokenizing anything

Since [`CONTINUING_TRAINING_ON_NEW_DATA.md`'s `--reuse-tokenizer`](CONTINUING_TRAINING_ON_NEW_DATA.md#the-fix---reuse-tokenizer)
guarantees every dataset round shares the same token vocabulary, old datasets' already-
tokenized `train.bin` files remain valid and meaningful **forever** — there's no need to
re-download or re-tokenize old text to use it as replay data. `build_replay_mix.py`
exploits this directly, working purely on the token arrays already on disk:

```bash
python scripts/build_replay_mix.py \
  --new data_round3/train.bin \
  --replay-from data/train.bin \
  --replay-from data_round2/train.bin \
  --replay-fraction 0.3 \
  --out data_round3/train_with_replay.bin \
  --seed 42
```

This samples a random contiguous chunk from **each** `--replay-from` source (pass it
multiple times to replay from every prior round, not just the last one — directly
solving the "forgetting compounds" problem above), sized so the combined output hits your
target `--replay-fraction` (30% replay / 70% new data by default), and writes one merged
`train.bin` ready to train on. Real, tested output:

```
[new] data_round3/train.bin: 397,678 tokens
[replay] data/train.bin: sampled 132,559 tokens
[replay] data_round2/train.bin: sampled 132,559 tokens
[done] wrote 662,796 tokens -> data_round3/train_with_replay.bin (actual replay fraction: 40.0%)
```

**What `--replay-fraction` to use**: start around `0.2`-`0.3` (20-30% of each round's
training data is old-round replay) — high enough to meaningfully counteract forgetting,
low enough that the new dataset still dominates what the round actually teaches the
model. Per [`CONTINUING_TRAINING_ON_NEW_DATA.md`'s measurement technique](CONTINUING_TRAINING_ON_NEW_DATA.md#you-can-actually-measure-forgetting-not-just-worry-about-it),
you can check whether this fraction is enough for your specific datasets by evaluating
against an old round's `val.bin` after training — raise the fraction if old-round loss is
still climbing too much.

## Tool #2: checkpoint snapshots — a rollback point before every new round

```bash
# Before starting a new dataset round:
make snapshot NAME=round2-shakespeare

# ...train on the new round...
DATA_DIR=data_round2/train_with_replay.bin STEPS=6000 make train-resume

# If round 2 made things worse, roll back:
make restore-snapshot NAME=round2-shakespeare
```

`make snapshot` copies your **current** `tinystories_gpt_checkpoint.pt`,
`tinystories_gpt_checkpoint_latest.pt`, and `logs/` into
`checkpoints_archive/<NAME>/` — a labeled point-in-time copy, before the new round's
training can touch the active checkpoint files. `make restore-snapshot` copies them back,
making that archived state the active checkpoint again — the next `make train-resume`
continues from exactly that point, as if the intervening round never happened.
`make list-snapshots` shows every archived round. None of this is committed to git (the
repo's root `.gitignore` already excludes `*.pt` and `logs` at any depth, including
inside `checkpoints_archive/`) — these are local safety checkpoints, not project history.

## The full recommended cadence, per new dataset

```bash
# 1. Snapshot BEFORE touching anything, so this round is reversible
make snapshot NAME=round<N>-<short-dataset-name>

# 2. Prepare the new dataset, reusing the existing tokenizer (never train a new one)
python prepare_dataset.py --dataset <new-dataset> \
  --reuse-tokenizer data/tokenizer.json --out-dir data_round<N>

# 3. Build a replay mix from ALL prior rounds (not just the last one)
python scripts/build_replay_mix.py \
  --new data_round<N>/train.bin \
  --replay-from data/train.bin --replay-from data_round2/train.bin ... \
  --replay-fraction 0.3 \
  --out data_round<N>/train_with_replay.bin

# 4. Train a modest number of steps for this round (a few thousand is enough per round
#    at this model size — see docs/HOW_MUCH_TRAINING_IS_ENOUGH.md for reading the
#    train/val curve to judge when THIS round is done)
STEPS=<current_step_count + a_few_thousand> make train-resume

# 5. Check quality didn't regress — both by loss (against old rounds' val sets, per
#    CONTINUING_TRAINING_ON_NEW_DATA.md's measurement technique) and by actually reading
#    generated samples (make infer)

# 6. If it regressed badly: make restore-snapshot NAME=round<N>-<short-dataset-name>
#    If it's good: move on to round N+1
```

## Why this stays manageable as rounds accumulate, unlike naive repeated fine-tuning

The two failure modes this workflow specifically avoids, compounding round over round:

- **Without replay mixing**: each round only "remembers" what the immediately prior
  round taught it, so early-round knowledge silently erodes a little more with every
  subsequent round — after enough rounds, the model may retain almost nothing of round 1.
  Replaying from *every* prior round each time directly prevents this decay.
- **Without snapshots**: a single bad dataset round permanently degrades the model, with
  no way back short of retraining everything from round 1 again — a real cost at any
  scale, and a discouraging one for a low-resource, keep-going-indefinitely workflow.
  Snapshots make every round a reversible experiment instead of a one-way commitment.
