# Model artifacts, open weights, and open source

## A model checkpoint is not a runnable model

A `.pt`, `.pth`, `.bin`, or `.safetensors` file generally stores learned tensors. Those
tensors are the **weights**, not the complete executable model.

To turn text into generated text, a deployment normally needs all of these pieces:

| Artifact | Purpose | What fails without it? |
| --- | --- | --- |
| Model class / architecture code | Defines layers, attention, cache behavior, and forward pass | Weight tensors have nowhere to load or execute. |
| Model configuration | Defines layer counts, dimensions, heads, vocabulary size, position limits, and architecture name | The runtime cannot construct the correct empty model. |
| Weights | Learned parameter values | The model has random/uninitialized behavior. |
| Tokenizer | Converts text to token IDs and back | The model cannot reliably accept or produce human text. |
| Special-token definitions | Defines BOS, EOS, padding, and other control tokens | Generation may not stop or format correctly. |
| Chat template, for chat models | Formats role messages into the model's trained prompt syntax | Instruction-following/chat behavior degrades or breaks. |

Think of a checkpoint as the values in a program's variables. The architecture code and
configuration are the program that creates those variables and defines the computation.

```text
user text
  → tokenizer
  → token IDs
  → model class instantiated from config
  → weights loaded into that model class
  → generated token IDs
  → tokenizer
  → assistant text
```

## Can weights alone be used?

Sometimes, but only when the missing information can be recovered reliably.

| What is available? | Can it run? | What is needed next? |
| --- | --- | --- |
| Weights plus known standard architecture and original tokenizer | Usually | Recreate the exact config and load compatible architecture code. |
| Hugging Face model directory with config/tokenizer/weights | Usually | Use Transformers or vLLM if the architecture is supported. |
| Weights plus custom model code but no config | Possibly | Recover dimensions and all behavior from code/checkpoint. |
| Raw weights only, unknown architecture/tokenizer | Not reliably | Obtain original artifacts or perform difficult reverse engineering. |

Tensor names and shapes can provide clues—such as layer count or hidden size—but they do
not reliably reveal all behavior. Important details such as RoPE scaling, normalization,
attention layout, activation functions, vocabulary ordering, and special tokens can make
an apparently loaded model produce incorrect output.

For vLLM specifically, a recognized or registered model architecture is required in
addition to the checkpoint. `--load-format pt` selects a PyTorch weight-file loader; it
does not infer a model class from arbitrary tensors. See
[CUSTOM_PYTORCH_MODEL_INTEGRATION.md](CUSTOM_PYTORCH_MODEL_INTEGRATION.md).

## Open-weight, open-source, and reproducible models

These terms are often used loosely. Separating them avoids overstating what is available.

| Term | Usually means | Does it guarantee training can be reproduced? |
| --- | --- | --- |
| **Open-weight** | Trained model weights can be downloaded under stated terms. | No. |
| **Open-source model/project** | Model implementation or surrounding code is available under an open-source license. | No. Code alone is insufficient. |
| **Open-data model** | Training data is available or fully disclosed under usable terms. | Not by itself. Training recipe/compute still matter. |
| **Fully reproducible open model** | Weights, code, data or usable data recipe, preprocessing, training method, hyperparameters, and evaluation are available. | Closest practical meaning to yes, subject to hardware and nondeterminism. |

Missing training data does not make a model's code “closed source.” It does mean that
outsiders cannot fully audit or reproduce the training process. Likewise, a permissive
weight license does not prove that all source code or training data is available.

## How to describe TinyLlama precisely

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` has publicly downloadable weights under Apache 2.0
and associated project code/documentation. It is therefore commonly called an open model
or open-source LLM in everyday discussion.

For technical, legal, or governance writing, a more precise description is:

> An Apache-2.0 open-weight model with publicly available project code and documentation.

Use “fully reproducible” only after confirming that the complete training-data rights,
data-processing pipeline, training recipe, evaluation assets, and relevant environment
details are available.

## Deployment checklist

Before serving an externally obtained model, verify:

1. The weight license permits the intended use and redistribution.
2. The exact model revision and checksums are recorded.
3. The model config matches the architecture and tensors.
4. The tokenizer and special tokens come from the same release family.
5. The chat template is present and appropriate for instruction-tuned checkpoints.
6. Custom code is reviewed before using `trust_remote_code`.
7. The model is loaded with a known runtime and tested with deterministic prompts.

## Related guides

- [PyTorch checkpoint versus safetensors](CHECKPOINT_FORMATS.md)
- [Custom PyTorch model integration with vLLM](CUSTOM_PYTORCH_MODEL_INTEGRATION.md)
- [vLLM serving guide](VLLM_SERVING_GUIDE.md)
