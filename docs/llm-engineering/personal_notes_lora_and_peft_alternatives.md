> **Personal notes** — written before [Chapter 17](17_lora_and_qlora.md) existed in this
> curriculum. Kept here as-is (original file, not rewritten) since its content is real,
> accurate, and a genuinely useful quick-reference companion — its PEFT-alternatives
> table and the image-generation cross-domain example were folded into
> [Chapter 17's "LoRA's Neighbors in the PEFT Family" section](17_lora_and_qlora.md#deep-dive-loras-neighbors-in-the-peft-family)
> directly. Read Chapter 17 first for the full first-principles treatment (the actual
> math, grounded in this repo's real training code); this file is a condensed standalone
> summary, useful for a fast refresher.

# LoRA and Alternatives for LLM Fine-Tuning

## 1. What is LoRA?

**LoRA** stands for **Low-Rank Adaptation**. It is a technique for fine-tuning large AI models efficiently without updating all of the model's original parameters.

### Core idea

A large pretrained model may contain billions of parameters. Instead of modifying all of them:

- The original model weights are **frozen**.
- Small trainable matrices are added to selected layers.
- Only these small matrices are trained.
- The resulting adapter can be loaded on top of the original model.

Conceptually:

```text
Original model weights (frozen)
            +
      LoRA adapter
            ↓
   Adapted/fine-tuned model
```

LoRA represents the weight update approximately as:

```text
ΔW = A × B
```

where `A` and `B` are much smaller matrices and the rank is deliberately kept small.

### Why LoRA is useful

- Much lower training cost than full fine-tuning
- Lower GPU memory requirements
- Small adapter files
- Multiple adapters can be used with the same base model
- Practical for customizing large language models

### Simple analogy

Think of a large reference book.

- **Full fine-tuning:** Rewrite the entire book.
- **LoRA:** Keep the book unchanged and create a small supplement containing the changes.

The supplement is much smaller but can substantially change how the model behaves.

---

## 2. Alternatives to LoRA

LoRA is part of a broader family called **Parameter-Efficient Fine-Tuning (PEFT)**.

Important alternatives include:

| Technique | Basic idea | Memory/compute | Typical use |
|---|---|---:|---|
| **LoRA** | Train low-rank matrices | Low | General-purpose fine-tuning |
| **QLoRA** | Quantized base model + LoRA | Very low | Large models on limited GPUs |
| **Adapters** | Insert small trainable modules | Low | Multiple task-specific adaptations |
| **Prefix Tuning** | Learn virtual prefix vectors | Very low | Task adaptation |
| **Prompt Tuning** | Learn trainable prompt embeddings | Very low | Lightweight specialization |
| **IA³** | Learn scaling vectors for activations | Very low | Extremely parameter-efficient tuning |
| **BitFit** | Train only bias parameters | Very low | Minimal adaptation |
| **Full Fine-Tuning** | Update all model parameters | Very high | Maximum adaptation |

---

## 3. How much of the model gets modified?

A useful mental model is:

```text
Full Fine-Tuning
        ↓
Modify millions/billions of model weights
        ↓
LoRA / Adapters
        ↓
Modify small trainable components
        ↓
Prefix / Prompt Tuning
        ↓
Modify learned embeddings
        ↓
Prompt Engineering
        ↓
Modify nothing in the model
```

The further down the list you go, the less of the original model you need to train.

---

## 4. LoRA vs QLoRA

**QLoRA** is especially important when working with large language models.

### LoRA

```text
FP16/BF16 Base Model
        +
LoRA adapters
        ↓
Fine-tuned model
```

### QLoRA

```text
4-bit Quantized Base Model
        +
LoRA adapters
        ↓
Fine-tuned model
```

QLoRA keeps the base model quantized, substantially reducing GPU memory requirements while still using LoRA adapters for training.

---

## 5. LoRA in Image Generation

LoRA is not limited to LLMs.

It is also widely used with image-generation models such as Stable Diffusion.

A LoRA can teach a base model things such as:

- A particular visual style
- A character
- A person
- A clothing style
- A particular object
- Specific poses or concepts

The same base model can therefore be combined with different adapters.

---

## 6. LoRA vs Other Approaches

### Full Fine-Tuning

All model parameters are updated.

**Advantages**
- Maximum flexibility
- Can produce strong task specialization

**Disadvantages**
- Very expensive
- High GPU memory requirements
- Large resulting model
- More difficult to maintain multiple specialized versions

### LoRA

Only small adapter parameters are trained.

**Advantages**
- Cheap
- Small adapters
- Easy to swap between tasks
- Lower GPU requirements

**Disadvantages**
- Adds an adaptation mechanism
- May not provide the same flexibility as full fine-tuning for every task

### Prompt Engineering

No model parameters are changed.

You simply change the instructions sent to the model.

**Advantages**
- No training
- No additional model storage
- Extremely simple

**Disadvantages**
- Cannot fundamentally teach the model new behavior as reliably as fine-tuning
- Prompt effectiveness can be task-dependent

### RAG

RAG (**Retrieval-Augmented Generation**) is different from fine-tuning.

Instead of changing model weights, external information is retrieved and provided to the model at inference time.

```text
Documents
    ↓
Embedding / Index
    ↓
Retriever
    ↓
Relevant context
    ↓
LLM
    ↓
Answer
```

RAG is generally better when the goal is to give a model access to changing or private knowledge rather than teach it a new behavior.

---

## 7. What to Learn First for LLM Infrastructure

If the goal is understanding **LLM GPU infrastructure and operations**, you do not need to master every PEFT method initially.

A practical learning order is:

```text
1. Full Fine-Tuning
        ↓
2. LoRA
        ↓
3. QLoRA
        ↓
4. Distributed Fine-Tuning
   ├── FSDP
   └── DeepSpeed
        ↓
5. GPU Memory / Communication
   ├── CUDA
   ├── NCCL
   ├── NVLink
   └── InfiniBand
        ↓
6. LLM Inference / Serving
```

> Note: the sibling `platform-lab` repo's `gpu_infrastructure/` track covers steps 4-6 of
> this exact roadmap in depth — Phase 7 (FSDP/DeepSpeed/ZeRO), Phase 3 (NCCL/RDMA/
> InfiniBand), and Phase 5 (LLM serving) respectively.

### The two most important PEFT techniques

For practical LLM work, prioritize:

**LoRA → QLoRA**

Then learn distributed training and GPU communication concepts.

---

## 8. Key Mental Model

The simplest way to remember the whole topic:

```text
                 MODEL CUSTOMIZATION
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Change weights    Add adapters     Change input
        │                │                │
   Full Fine-Tune    LoRA / QLoRA    Prompt / RAG
        │                │
   Expensive        Parameter-efficient
```

**LoRA = small trainable changes to a frozen model.**

**QLoRA = LoRA + a quantized base model.**

**Full fine-tuning = update the whole model.**

**RAG = don't change the model; give it relevant external information.**
