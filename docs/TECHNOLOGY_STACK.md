# Technology Stack

This document provides an overview of all technologies used in the F5 AI Technical Assistant Training Lab, explaining what each component does and why it was chosen.

---

## Table of Contents
- [Overview](#overview)
- [Base Model](#base-model)
- [Quantization & Memory Optimization](#quantization--memory-optimization)
- [Fine-Tuning Stack](#fine-tuning-stack)
- [RAG Stack](#rag-stack)
- [Core ML Infrastructure](#core-ml-infrastructure)
- [Environment](#environment)
- [Version Summary](#version-summary)

---

## Overview

This project demonstrates two approaches to creating a domain-specific AI assistant:

1. **RAG (Retrieval-Augmented Generation)**: Enhances a base model with external knowledge at inference time
2. **Fine-Tuning with QLoRA**: Trains the model to internalize domain knowledge and terminology

The tech stack was selected to run entirely on Google Colab's free tier (T4 GPU with 16GB VRAM).

---

## Base Model

### TinyLlama/TinyLlama-1.1B-Chat-v1.0
| Attribute | Value |
|-----------|-------|
| Parameters | 1.1 billion |
| Architecture | LLaMA-based decoder-only transformer |
| Context Length | 2,048 tokens |
| Training Data | 3 trillion tokens (SlimPajama, StarCoder) |
| License | Apache 2.0 |

**What it does**: TinyLlama is a compact large language model that provides strong performance relative to its size. The chat-tuned version has been instruction-fine-tuned for conversational use.

**Why we use it**:
- Small enough to fit in Colab's free T4 GPU memory (~2GB when 4-bit quantized)
- Fast inference and training times for a workshop setting
- Good baseline capabilities for demonstrating improvement via RAG/fine-tuning
- Open license allows unrestricted use

---

## Quantization & Memory Optimization

### BitsAndBytes
| Component | Purpose |
|-----------|---------|
| `load_in_4bit` | Reduces model weights from 16-bit to 4-bit |
| `nf4` quantization | Normalized float 4-bit format optimized for neural networks |
| `double_quant` | Quantizes the quantization constants for additional savings |

**What it does**: BitsAndBytes enables loading and running large models in reduced precision, dramatically cutting memory requirements while maintaining most of the model's capabilities.

**Why we use it**:
- Reduces TinyLlama from ~4.4GB to ~2GB VRAM
- Enables fine-tuning on consumer GPUs
- Minimal quality degradation with NF4 quantization

### Accelerate
**What it does**: Hugging Face's library for distributed and mixed-precision training. Handles device placement, gradient accumulation, and multi-GPU setups automatically.

**Why we use it**: Required by Transformers for efficient model loading with `device_map="auto"`.

---

## Fine-Tuning Stack

### PEFT (Parameter-Efficient Fine-Tuning)
**What it does**: Instead of updating all model weights, PEFT methods add small trainable adapters while freezing the base model. This dramatically reduces:
- Memory needed for training
- Storage for saved models
- Risk of catastrophic forgetting

**Key concepts**:
- **LoRA (Low-Rank Adaptation)**: Decomposes weight updates into low-rank matrices
- **Adapter size**: Controlled by `r` (rank) parameter - higher = more capacity but more memory

### QLoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=16,           # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM"    # Decoder-only language model
)
```

**Why we use it**: QLoRA combines 4-bit quantization with LoRA adapters, enabling fine-tuning of billion-parameter models on a single consumer GPU.

### TRL (Transformer Reinforcement Learning)
**What it does**: Hugging Face's library for training language models with various objectives including supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and direct preference optimization (DPO).

**Component used**: `SFTTrainer` - Supervised Fine-Tuning Trainer

**Why we use SFTTrainer**:
- Handles chat template formatting automatically
- Integrates seamlessly with PEFT/LoRA
- Manages gradient accumulation and mixed precision
- Provides training callbacks and logging

### Unsloth
**What it does**: Optimized kernels and training loops that accelerate fine-tuning by 2x while reducing memory usage by up to 60%.

**Key optimizations**:
- Custom CUDA kernels for attention and MLP layers
- Fused operations to reduce memory bandwidth
- Gradient checkpointing optimizations

**Why we use it**: Makes training feasible within Colab's session time limits and memory constraints.

---

## RAG Stack

### LangChain
**What it does**: Framework for building applications with LLMs. Provides abstractions for:
- Document loading and processing
- Text splitting and chunking
- Vector store integration
- Retrieval chains

**Components used**:
| Component | Purpose |
|-----------|---------|
| `DirectoryLoader` | Load all .txt files from a directory |
| `TextLoader` | Parse plain text files |
| `RecursiveCharacterTextSplitter` | Split documents into overlapping chunks |

**Why we use it**: Industry-standard framework with extensive documentation and integrations.

### ChromaDB
**What it does**: Open-source vector database that stores embeddings and enables similarity search. Runs entirely in-memory or persisted to disk.

**Key features**:
- No external server required (embedded mode)
- Automatic embedding management
- Metadata filtering
- Multiple distance metrics (cosine, L2, IP)

**Why we use it**:
- Zero configuration - works out of the box
- No API keys or external services needed
- Fast enough for demo/workshop purposes
- Persists to disk for reuse across sessions

### Sentence-Transformers
**What it does**: Library for computing dense vector embeddings of text using transformer models. These embeddings capture semantic meaning, enabling similarity search.

**Model used**: `all-MiniLM-L6-v2`
| Attribute | Value |
|-----------|-------|
| Dimensions | 384 |
| Max Sequence | 256 tokens |
| Size | ~80MB |
| Speed | Very fast |

**Why we use it**:
- Free and runs locally (no API costs)
- Good balance of quality and speed
- Small enough to load alongside the LLM
- Well-suited for technical documentation retrieval

### RAG Pipeline Flow

The RAG pipeline has two phases: **indexing** (one-time setup) and **retrieval** (every query).

#### Phase 1: Indexing (One-Time Setup)

Before we can answer questions, we must prepare our knowledge base:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   F5 Documentation Files                                         │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│   │ ssl_offload  │ │ irules_guide │ │ load_balance │  ...       │
│   │    .txt      │ │    .txt      │ │    .txt      │            │
│   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│          │                │                │                     │
│          └────────────────┼────────────────┘                     │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │  Text Splitter  │                             │
│                  │  (500 chars,    │                             │
│                  │   50 overlap)   │                             │
│                  └────────┬────────┘                             │
│                           │                                      │
│          Why? Documents are too long for LLM context.            │
│          We split into chunks that fit and overlap               │
│          to preserve context at boundaries.                      │
│                           │                                      │
│                           ▼                                      │
│          ┌────────────────────────────────┐                      │
│          │  Chunks (e.g., 45 total)       │                      │
│          │  "To configure SSL offload..." │                      │
│          │  "Create a Client SSL prof..." │                      │
│          │  "iRules use TCL syntax..."    │                      │
│          └────────────────┬───────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │ Embedding Model │                             │
│                  │ (all-MiniLM-L6) │                             │
│                  └────────┬────────┘                             │
│                           │                                      │
│          Why? Computers can't understand text directly.          │
│          Embeddings convert text → numbers (vectors)             │
│          where similar meanings = similar numbers.               │
│                           │                                      │
│                           ▼                                      │
│          ┌────────────────────────────────┐                      │
│          │  Vectors (384 dimensions each) │                      │
│          │  [0.23, -0.45, 0.12, ...]      │                      │
│          │  [0.67, -0.21, 0.89, ...]      │                      │
│          │  [-0.15, 0.33, 0.44, ...]      │                      │
│          └────────────────┬───────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │    ChromaDB     │                             │
│                  │  (Vector Store) │                             │
│                  └─────────────────┘                             │
│                                                                  │
│          ChromaDB stores vectors + original text.                │
│          Think of it as a searchable index where                 │
│          "search" means "find similar vectors."                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 2: Retrieval (Every Query)

When a user asks a question:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Question: "How do I configure SSL offloading?"           │
│                           │                                      │
│                           ▼                                      │
│            ┌───────────────────────────┐                         │
│            │     Embedding Model       │                         │
│            │     (all-MiniLM-L6)       │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│       Why embed the question? To search ChromaDB, we need        │
│       to compare vectors. The question must become a vector      │
│       so we can find chunks with similar vectors.                │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │   Question Vector         │                         │
│            │   [0.25, -0.42, 0.15...]  │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │       ChromaDB           │                         │
│            │   Similarity Search       │                         │
│            │   (cosine distance)       │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│       How it works: ChromaDB compares the question vector        │
│       against ALL stored chunk vectors using cosine similarity.  │
│       Vectors pointing in similar directions = similar meaning.  │
│                          │                                       │
│       Example distances:                                         │
│       • "SSL offloading steps..." → 0.15 (very similar!)        │
│       • "Create Client SSL..."    → 0.22 (similar)              │
│       • "Round Robin algorithm"   → 0.89 (not similar)          │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │   Top-K Results (k=3)     │                         │
│            │                           │                         │
│            │   1. "To configure SSL    │                         │
│            │      offloading, first    │                         │
│            │      import your cert..." │                         │
│            │                           │                         │
│            │   2. "Create a Client     │                         │
│            │      SSL profile under    │                         │
│            │      Local Traffic..."    │                         │
│            │                           │                         │
│            │   3. "Attach the SSL      │                         │
│            │      profile to your      │                         │
│            │      virtual server..."   │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│       We retrieve the original TEXT (not vectors) of the        │
│       most similar chunks. These become our context.             │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │    Prompt Construction    │                         │
│            │                           │                         │
│            │  "You are an F5 expert.   │                         │
│            │   Use this context:       │                         │
│            │                           │                         │
│            │   [Retrieved chunks...]   │                         │
│            │                           │                         │
│            │   Question: How do I      │                         │
│            │   configure SSL...?"      │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│       Why inject context? The LLM doesn't "know" F5 docs.        │
│       By putting relevant text in the prompt, we give it         │
│       the information needed to answer accurately.               │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │       TinyLlama LLM       │                         │
│            │    (generates response)   │                         │
│            └─────────────┬─────────────┘                         │
│                          │                                       │
│       The LLM reads the context and question, then generates     │
│       an answer. It's essentially "open-book" - the model        │
│       synthesizes an answer FROM the provided context.           │
│                          │                                       │
│                          ▼                                       │
│            ┌───────────────────────────┐                         │
│            │   Generated Response      │                         │
│            │                           │                         │
│            │   "To configure SSL       │                         │
│            │    offloading on BIG-IP:  │                         │
│            │    1. Import your SSL     │                         │
│            │       certificate...      │                         │
│            │    2. Create a Client     │                         │
│            │       SSL profile..."     │                         │
│            └───────────────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Why This Architecture?

| Problem | Solution |
|---------|----------|
| LLMs have knowledge cutoffs and gaps | RAG injects current/domain knowledge at query time |
| Can't search text by meaning with keywords | Embeddings enable semantic search ("SSL setup" finds "certificate configuration") |
| Full documents don't fit in LLM context | Chunking + retrieval finds just the relevant parts |
| LLMs can hallucinate facts | Grounding in retrieved documents improves accuracy |
| Updating LLM knowledge requires retraining | Just update the document store - no retraining needed |

#### Key Insight: Two Different Models

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   EMBEDDING MODEL                    LLM (TinyLlama)            │
│   (all-MiniLM-L6-v2)                                            │
│                                                                  │
│   Purpose: Convert text → vectors    Purpose: Generate text     │
│                                                                  │
│   Input:  "SSL offloading"           Input:  Full prompt with   │
│   Output: [0.23, -0.45, ...]                 context + question │
│           (384 numbers)              Output: Natural language   │
│                                              answer              │
│                                                                  │
│   Used for: SEARCHING                Used for: ANSWERING        │
│   (finding relevant docs)            (generating response)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

The embedding model and LLM serve completely different purposes. The embedding model is small and fast (80MB) - optimized for creating searchable representations. The LLM is larger (2GB quantized) - optimized for understanding and generating language.

---

## Core ML Infrastructure

### PyTorch
**What it does**: Deep learning framework providing tensors, automatic differentiation, and GPU acceleration. The foundation for all model operations.

**Why we use it**: Industry standard, required by Transformers and all other ML libraries in the stack.

### Transformers (Hugging Face)
**What it does**: Library providing pre-trained models, tokenizers, and training utilities for NLP tasks.

**Components used**:
| Component | Purpose |
|-----------|---------|
| `AutoModelForCausalLM` | Load decoder-only language models |
| `AutoTokenizer` | Load model-specific tokenizers |
| `BitsAndBytesConfig` | Configure quantization settings |
| `pipeline` | High-level inference API |
| `TrainingArguments` | Configure training hyperparameters |

**Why we use it**: Central hub for accessing models and standardized training loops.

### Tokenizer

**What it does**: A tokenizer converts human-readable text into numbers (token IDs) that the model can process, and converts the model's output back into text.

#### Why Tokenizers Are Necessary

Neural networks only understand numbers, not text. The tokenizer is the translator between human language and the model's numeric world:

```
┌─────────────────────────────────────────────────────────────────┐
│                     TOKENIZATION PROCESS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Text: "Configure SSL offloading"                        │
│                          │                                       │
│                          ▼                                       │
│                    ┌───────────┐                                 │
│                    │ Tokenizer │                                 │
│                    └─────┬─────┘                                 │
│                          │                                       │
│                          ▼                                       │
│   Step 1: Split into subwords (tokens)                          │
│                                                                  │
│   "Configure" → ["Con", "fig", "ure"]                           │
│   "SSL"       → ["SS", "L"]                                     │
│   "offloading"→ ["off", "load", "ing"]                          │
│                                                                  │
│   Why subwords? The model can't store every possible word.      │
│   Instead, it learns ~32,000 common subwords and combines       │
│   them. This handles new/rare words like "BIG-IP" or "iRule".   │
│                          │                                       │
│                          ▼                                       │
│   Step 2: Convert to token IDs (numbers)                        │
│                                                                  │
│   ["Con", "fig", "ure", "SS", "L", "off", "load", "ing"]        │
│              ↓                                                   │
│   [1128, 2500, 545, 5765, 43, 1283, 2613, 292]                  │
│                                                                  │
│   Each subword maps to a unique ID in the vocabulary.           │
│   These IDs are what the neural network actually processes.     │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │           TinyLlama Model               │                   │
│   │   (processes token IDs, outputs new IDs)│                   │
│   └─────────────────────────────────────────┘                   │
│                          │                                       │
│                          ▼                                       │
│   Step 3: Decode output IDs back to text                        │
│                                                                  │
│   [1762, 2891, 445, ...] → "To configure SSL offloading..."    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Tokenizer Concepts

| Concept | Explanation |
|---------|-------------|
| **Vocabulary** | Fixed set of ~32,000 tokens the model knows. TinyLlama uses the LLaMA tokenizer vocabulary. |
| **Subword tokenization** | Words are split into pieces. "unhappiness" → ["un", "happiness"] or ["un", "hap", "pi", "ness"] |
| **Special tokens** | Control tokens like `<s>` (start), `</s>` (end), `<pad>` (padding). Used for formatting. |
| **Token IDs** | Integer indices into the vocabulary. "hello" might be token ID 12345. |
| **Encoding** | Text → token IDs (what we send to the model) |
| **Decoding** | Token IDs → text (what we show to users) |

#### Why Each Model Needs Its Own Tokenizer

Different models use different vocabularies and tokenization algorithms:

```python
# TinyLlama tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Same text, different tokenization per model:
# GPT-2:     "Hello" → [15496]           (1 token)
# LLaMA:     "Hello" → [10994]           (1 token, different ID)
# BERT:      "Hello" → [7592]            (1 token, different ID)
```

The model's weights are trained expecting specific token IDs. Using the wrong tokenizer would be like speaking French to someone who only understands Japanese - the IDs would map to wrong meanings.

#### Context Length and Token Counting

Models have maximum context lengths measured in **tokens, not characters**:

```
TinyLlama context limit: 2,048 tokens

Example token counts:
• "Hello"                           →  1 token
• "SSL offloading configuration"    →  4 tokens
• A typical paragraph (100 words)   → ~130 tokens
• This entire documentation file    → ~3,500 tokens (too long!)
```

This is why RAG chunks documents - we need retrieved context + question + response to fit within the token limit.

#### Tokenizer in Our Code

```python
# Load the tokenizer that matches our model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Set padding token (required for batch processing)
tokenizer.pad_token = tokenizer.eos_token

# Encoding: text → tokens
input_ids = tokenizer.encode("What is SSL offloading?")
# Result: [1, 1724, 338, 17122, 1283, 13789, 29973]

# Decoding: tokens → text
text = tokenizer.decode([1, 1724, 338, 17122, 1283, 13789, 29973])
# Result: "<s> What is SSL offloading?"

# The pipeline handles this automatically:
pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### Datasets (Hugging Face)
**What it does**: Library for loading, processing, and sharing datasets. Provides memory-efficient data handling via Apache Arrow.

**Why we use it**: Efficient loading of JSONL training data with automatic batching.

---

## Environment

### Google Colab
**What it does**: Free cloud-based Jupyter notebook environment with GPU access.

**Resources (Free Tier)**:
| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA T4 (16GB VRAM) |
| RAM | ~12GB system memory |
| Disk | ~100GB temporary storage |
| Session | Up to 12 hours (may disconnect earlier) |

**Why we use it**:
- Free GPU access for all students
- No local setup required
- Pre-installed CUDA drivers
- Easy notebook sharing

### NVIDIA T4 GPU
**What it does**: Data center GPU optimized for inference and light training workloads.

**Specifications**:
| Spec | Value |
|------|-------|
| CUDA Cores | 2,560 |
| Tensor Cores | 320 |
| Memory | 16GB GDDR6 |
| FP16 Performance | 65 TFLOPS |

**Why it works for this lab**:
- 16GB VRAM sufficient for 4-bit quantized TinyLlama + training
- Tensor cores accelerate mixed-precision training
- Available free on Colab

---

## Version Summary

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `transformers` | 4.44.0 | Model loading and training |
| `accelerate` | ≥0.33.0 | Distributed training utilities |
| `peft` | ≥0.12.0 | Parameter-efficient fine-tuning |
| `bitsandbytes` | ≥0.43.0 | Quantization |
| `trl` | ≥0.9.0 | SFTTrainer |
| `unsloth` | ≥2024.8 | Training acceleration |
| `langchain` | ≥0.2.0 | RAG framework |
| `langchain-community` | ≥0.2.0 | Document loaders |
| `langchain-huggingface` | ≥0.0.3 | HuggingFace integrations |
| `chromadb` | ≥0.5.0 | Vector database |
| `sentence-transformers` | ≥3.0.0 | Embedding model |
| `datasets` | ≥2.20.0 | Data loading |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Colab (T4 GPU)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    TinyLlama 1.1B                        │    │
│  │              (4-bit quantized via BitsAndBytes)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                    │                 │
│           ▼                                    ▼                 │
│  ┌─────────────────────┐            ┌─────────────────────┐     │
│  │    Fine-Tuning      │            │        RAG          │     │
│  │  ┌───────────────┐  │            │  ┌───────────────┐  │     │
│  │  │   Unsloth     │  │            │  │  LangChain    │  │     │
│  │  │   + QLoRA     │  │            │  │  + ChromaDB   │  │     │
│  │  │   + PEFT      │  │            │  │  + MiniLM     │  │     │
│  │  │   + TRL       │  │            │  └───────────────┘  │     │
│  │  └───────────────┘  │            └─────────────────────┘     │
│  └─────────────────────┘                                        │
│           │                                    │                 │
│           ▼                                    ▼                 │
│  ┌─────────────────────┐            ┌─────────────────────┐     │
│  │   LoRA Adapter      │            │   Vector Store      │     │
│  │   (~20MB saved)     │            │   (F5 docs indexed) │     │
│  └─────────────────────┘            └─────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Further Reading

- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
