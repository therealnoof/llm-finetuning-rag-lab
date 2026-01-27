# LLM Fine-Tuning & RAG Lab
## Data Flow Design Document

**A Detailed Guide to Understanding How Data Moves Through the System**

For Beginners and Technical Learners

Version 1.0 | January 2026

---

## Table of Contents

1. [Introduction to Data Flow](#1-introduction-to-data-flow)
2. [The Big Picture: Two Main Pipelines](#2-the-big-picture-two-main-pipelines)
3. [Pipeline A: Document Ingestion (Indexing Phase)](#3-pipeline-a-document-ingestion-indexing-phase)
   - 3.1 [Document Loading](#31-document-loading)
   - 3.2 [Text Chunking](#32-text-chunking)
   - 3.3 [Embedding Generation](#33-embedding-generation)
   - 3.4 [Vector Storage](#34-vector-storage)
4. [Pipeline B: Query Processing (Inference Phase)](#4-pipeline-b-query-processing-inference-phase)
   - 4.1 [Query Tokenization](#41-query-tokenization)
   - 4.2 [Query Embedding](#42-query-embedding)
   - 4.3 [Semantic Search](#43-semantic-search-vector-similarity)
   - 4.4 [Context Assembly](#44-context-assembly)
   - 4.5 [LLM Processing](#45-llm-processing-tinyllama)
   - 4.6 [Response Generation](#46-response-generation-autoregressive-decoding)
5. [Inside the Neural Networks: Matrix Operations](#5-inside-the-neural-networks-matrix-operations)
6. [Fine-Tuning Data Flow](#6-fine-tuning-data-flow)
7. [Hardware Mapping: CPU vs GPU Operations](#7-hardware-mapping-cpu-vs-gpu-operations)
8. [Complete Data Flow Diagram](#8-complete-data-flow-diagram)
9. [Glossary](#9-glossary)

---

## 1. Introduction to Data Flow

This document explains exactly how data moves through the LLM Fine-Tuning and RAG Lab system. We will trace every step from the moment a user types a question to when they receive an answer, explaining what happens to the data at each stage.

### Why Understanding Data Flow Matters

Understanding data flow helps you:

- **Debug problems**: When something goes wrong, you can identify which stage failed
- **Optimize performance**: Know which operations are slow and why
- **Make informed decisions**: Choose the right tools and configurations
- **Explain AI systems**: Communicate how these systems work to others

### Key Concepts to Keep in Mind

Before diving in, understand these fundamental ideas:

- **Text to Numbers**: Computers cannot understand words directly. Every piece of text must be converted to numbers before processing.
- **Vectors**: A vector is simply a list of numbers. In AI, vectors represent the "meaning" of text in a mathematical form.
- **Matrix Operations**: Neural networks work by multiplying matrices (grids of numbers). This is where the actual "thinking" happens.
- **GPU vs CPU**: GPUs can perform thousands of matrix operations simultaneously, making them much faster for AI workloads.

---

## 2. The Big Picture: Two Main Pipelines

The RAG system has two distinct data pipelines that work together:

### Pipeline A: Document Ingestion (Happens Once)

This pipeline processes your F5 documentation and stores it in a searchable format. It runs BEFORE any user asks a question - typically during setup.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  F5 Docs (.txt)  ──►  Chunking  ──►  Embedding  ──►  ChromaDB  │
│                                                                 │
│  [Raw Text]         [Smaller     [384 numbers    [Stored for   │
│                      pieces]      per chunk]      searching]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline B: Query Processing (Happens Every Question)

This pipeline handles each user question in real-time, finds relevant context, and generates an answer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query ──► Embed ──► Search ──► Combine ──► LLM ──► Answer│
│                                                                 │
│  "How do I     [384     [Find      [Query +   [Generate  [Text │
│   configure     nums]    similar    Context]   response]  out] │
│   pools?"               chunks]                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How the Pipelines Connect

Pipeline A creates the "knowledge base" that Pipeline B searches. Without Pipeline A running first, Pipeline B would have nothing to search.

| Pipeline | When It Runs | How Often | Duration |
|----------|--------------|-----------|----------|
| A: Document Ingestion | Lab setup / data updates | Once (or when docs change) | 1-2 minutes |
| B: Query Processing | Every user question | Many times per session | 2-5 seconds per query |

---

## 3. Pipeline A: Document Ingestion (Indexing Phase)

Let's trace exactly what happens to your F5 documentation files as they are processed and stored.

### 3.1 Document Loading

#### What Happens

The system reads raw text files from disk into memory.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | LangChain TextLoader / DirectoryLoader |
| Hardware | CPU + Disk I/O |
| Memory | System RAM |
| Location | Your compute environment (Colab or AWS) |

#### Data Transformation

```
BEFORE (on disk):
  data/f5_docs/load_balancing.txt
  [Binary file on storage]

AFTER (in memory):
  Document object:
    - page_content: "Load balancing distributes incoming..."
    - metadata: {source: "load_balancing.txt"}
```

#### Technical Details

- **File Encoding**: UTF-8 text is decoded into Python string objects
- **Memory Usage**: Each character uses 1-4 bytes depending on the character
- **I/O Operation**: This is a blocking disk read operation
- **No GPU**: This step uses only CPU and disk; no neural network involved yet

---

### 3.2 Text Chunking

#### What Happens

Large documents are split into smaller, overlapping pieces called "chunks". This is necessary because embedding models have a maximum input length, and smaller chunks allow for more precise retrieval.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | LangChain RecursiveCharacterTextSplitter |
| Hardware | CPU only |
| Memory | System RAM |
| Algorithm | String manipulation (no neural network) |

#### Chunking Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| chunk_size | 500 characters | Maximum length of each chunk |
| chunk_overlap | 50 characters | How much consecutive chunks share |
| separators | ["\n\n", "\n", " ", ""] | Where to prefer splitting (paragraph, line, word, character) |

#### Data Transformation Example

```
BEFORE (one large document, 2000 characters):
  "Load balancing distributes incoming network traffic across
   multiple servers. This ensures no single server bears too
   much demand. By spreading the work evenly, load balancing
   improves application responsiveness... [continues for 2000 chars]"

AFTER (multiple chunks, ~500 characters each with 50 char overlap):

  Chunk 1 (chars 1-500):
    "Load balancing distributes incoming network traffic across
     multiple servers. This ensures no single server bears too
     much demand. By spreading the work evenly..."

  Chunk 2 (chars 451-950):  ◄── Note: starts 50 chars before chunk 1 ends
    "By spreading the work evenly, load balancing improves
     application responsiveness. It also increases availability
     of applications and websites..."

  Chunk 3 (chars 901-1400):
    "...and websites. F5 BIG-IP offers several load balancing
     algorithms including Round Robin, Least Connections..."
```

#### Why Overlap?

Overlap ensures that if important information spans a chunk boundary, it appears in at least one complete chunk. Without overlap, a sentence split across two chunks might lose its meaning in both.

---

### 3.3 Embedding Generation

#### What Happens

Each text chunk is converted into a vector (list of 384 numbers) that represents its semantic meaning. This is where the first neural network processing occurs.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | Sentence-Transformers (all-MiniLM-L6-v2) |
| Hardware | GPU (preferred) or CPU |
| Memory | GPU VRAM (if GPU) or System RAM |
| Model Size | ~80MB loaded into memory |
| Output | 384-dimensional float32 vector per chunk |

#### Step-by-Step Process

**Step 1: Tokenization (CPU)**

The chunk text is split into subword tokens. Each token is mapped to an integer ID from the vocabulary.

```
Input:  "Load balancing distributes traffic"
Tokens: ["Load", "bal", "##ancing", "distributes", "traffic"]
IDs:    [4391,   8347,  2739,       21854,         4026]
```

**Step 2: Token Embedding Lookup (GPU)**

Each token ID retrieves a vector from the embedding table. This is a simple table lookup, not computation.

```
Token ID 4391 ("Load") ──► [0.023, -0.156, 0.089, ...] (384 dims)
Token ID 8347 ("bal")  ──► [0.178, 0.042, -0.201, ...] (384 dims)
... and so on for each token
```

**Step 3: Transformer Encoding (GPU - Heavy Computation)**

Token vectors pass through 6 transformer layers. Each layer performs attention and feed-forward operations. This is where matrix multiplications happen.

```
Layer 1: Attention + Feed-Forward ──► Updated token vectors
Layer 2: Attention + Feed-Forward ──► More refined vectors
...
Layer 6: Attention + Feed-Forward ──► Final token vectors
```

**Step 4: Mean Pooling (GPU)**

All token vectors are averaged into a single vector. This produces one 384-dim vector representing the entire chunk.

```
Token 1 vector: [0.1, 0.2, 0.3, ...]
Token 2 vector: [0.2, 0.1, 0.4, ...]
Token 3 vector: [0.3, 0.3, 0.2, ...]
─────────────────────────────────────
Mean (average): [0.2, 0.2, 0.3, ...] ◄── Final chunk embedding
```

**Step 5: Normalization (GPU)**

The vector is normalized to unit length (length = 1). This makes cosine similarity computation simpler later.

#### Matrix Operations in Embedding

During transformer encoding, these matrix multiplications occur per layer:

| Operation | Matrix Sizes | Multiplications |
|-----------|--------------|-----------------|
| Query projection | [seq_len x 384] × [384 x 384] | ~147,456 × seq_len |
| Key projection | [seq_len x 384] × [384 x 384] | ~147,456 × seq_len |
| Value projection | [seq_len x 384] × [384 x 384] | ~147,456 × seq_len |
| Attention scores | [seq_len x 384] × [384 x seq_len] | ~147,456 × seq_len |
| Feed-forward 1 | [seq_len x 384] × [384 x 1536] | ~589,824 × seq_len |
| Feed-forward 2 | [seq_len x 1536] × [1536 x 384] | ~589,824 × seq_len |

For a 50-token chunk through 6 layers: approximately **50 million multiply-add operations**. This is why GPUs are so valuable - they can do millions of these in parallel.

---

### 3.4 Vector Storage

#### What Happens

The embedding vectors are stored in ChromaDB along with the original text and metadata. ChromaDB also builds an index structure for fast similarity search.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | ChromaDB |
| Hardware | CPU + Disk |
| Storage | Local filesystem (./chroma_db/) |
| Index Type | HNSW (Hierarchical Navigable Small World) |

#### Data Stored Per Chunk

```
ChromaDB Record:
┌─────────────────────────────────────────────────────────────────┐
│ ID: "chunk_001"                                                 │
│                                                                 │
│ Embedding: [0.0234, -0.1567, 0.0891, 0.2341, ..., -0.0123]     │
│            (384 float32 values = 1,536 bytes)                  │
│                                                                 │
│ Document: "Load balancing distributes incoming network..."     │
│           (original text, up to 500 characters)                │
│                                                                 │
│ Metadata: {                                                    │
│   "source": "load_balancing.txt",                              │
│   "chunk_index": 0                                             │
│ }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

#### HNSW Index Structure

ChromaDB builds a special index called HNSW that allows finding similar vectors without comparing against every single stored vector. Think of it like a smart filing system:

- **Layer 0 (bottom)**: Contains all vectors, connected to nearby neighbors
- **Layer 1**: Contains a subset of vectors, more sparsely connected
- **Layer 2+**: Even sparser, allowing quick navigation across the space
- **Search**: Start at top layer, navigate down to find nearest neighbors

This reduces search time from O(n) to approximately O(log n), meaning searching 10,000 vectors is nearly as fast as searching 100.

---

## 4. Pipeline B: Query Processing (Inference Phase)

Now let's trace what happens when a user asks: "How do I configure a health monitor on F5 BIG-IP?"

### 4.1 Query Tokenization

#### What Happens

The user's question is converted from a string of characters into a sequence of token IDs that the model can process. This happens twice: once for the embedding model and once for the LLM.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | Tokenizer (from Hugging Face) |
| Hardware | CPU only |
| Memory | System RAM |
| Algorithm | BPE (Byte-Pair Encoding) or similar |

#### Tokenization Process

```
User Input (string):
  "How do I configure a health monitor on F5 BIG-IP?"

Step 1 - Split into subwords:
  ["How", "do", "I", "configure", "a", "health", "monitor", "on",
   "F", "5", "BIG", "-", "IP", "?"]

Step 2 - Convert to IDs (lookup in vocabulary):
  [1128, 171, 146, 12876, 143, 1192, 4669, 187, 401, 245, 8724, 118, 5765, 136]

Step 3 - Add special tokens:
  [101, 1128, 171, 146, 12876, 143, 1192, 4669, 187, 401, 245, 8724, 118, 5765, 136, 102]
        ▲                                                                            ▲
       [CLS]                                                                       [SEP]
       start                                                                        end
```

#### Why Subword Tokenization?

- **Handles unknown words**: "BIG-IP" splits into known pieces even if never seen as a whole
- **Efficient vocabulary**: ~32,000 tokens can represent any text
- **Captures morphology**: "configuring" shares tokens with "configure"

---

### 4.2 Query Embedding

#### What Happens

The tokenized query passes through the same embedding model used for documents, producing a 384-dimensional vector. This vector will be used to find similar document chunks.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | Sentence-Transformers (all-MiniLM-L6-v2) |
| Hardware | GPU (preferred) or CPU |
| Input | 16 token IDs (from tokenization) |
| Output | 384 float32 values |
| Time | ~5-10ms on GPU, ~50-100ms on CPU |

#### Process (Same as Document Embedding)

```
Token IDs: [101, 1128, 171, ..., 102]
           │
           ▼
    ┌──────────────────────┐
    │  Embedding Lookup    │  ◄── Each ID gets a 384-dim vector
    └──────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Transformer Layer 1 │  ◄── Attention + Feed-Forward
    └──────────────────────┘
           │
           ▼
         (... 5 more layers ...)
           │
           ▼
    ┌──────────────────────┐
    │     Mean Pooling     │  ◄── Average all token vectors
    └──────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │    Normalization     │  ◄── Scale to unit length
    └──────────────────────┘
           │
           ▼
    Query Vector: [0.0891, -0.2341, 0.1567, ..., 0.0234]
                  (384 dimensions)
```

#### Critical Point: Same Model, Same Space

The query MUST be embedded with the same model that embedded the documents. Different models produce incompatible vector spaces - comparing vectors from different models is meaningless, like comparing distances in miles vs kilometers without conversion.

---

### 4.3 Semantic Search (Vector Similarity)

#### What Happens

ChromaDB compares the query vector against all stored document vectors to find the most similar chunks. This uses cosine similarity - measuring the angle between vectors.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | ChromaDB with HNSW index |
| Hardware | CPU (ChromaDB doesn't use GPU for search) |
| Algorithm | Approximate Nearest Neighbor via HNSW |
| Output | Top K most similar chunks (default K=4) |

#### Cosine Similarity Calculation

Cosine similarity measures how "aligned" two vectors are:

```
                    A · B         (dot product of vectors)
  similarity = ─────────────── = ──────────────────────────────
                ||A|| × ||B||    (product of their lengths)

  Since vectors are normalized (length = 1), this simplifies to:

  similarity = A · B = A[0]×B[0] + A[1]×B[1] + ... + A[383]×B[383]
```

#### Example Similarity Calculation

```
Query vector (simplified to 4 dims): [0.5, 0.5, 0.5, 0.5]

Document chunk 1 ("health monitors..."): [0.6, 0.4, 0.5, 0.5]
  Similarity = 0.5×0.6 + 0.5×0.4 + 0.5×0.5 + 0.5×0.5
             = 0.30 + 0.20 + 0.25 + 0.25 = 1.00 (very similar!)

Document chunk 2 ("SSL certificates..."): [-0.2, 0.8, -0.3, 0.4]
  Similarity = 0.5×(-0.2) + 0.5×0.8 + 0.5×(-0.3) + 0.5×0.4
             = -0.10 + 0.40 + -0.15 + 0.20 = 0.35 (less similar)
```

#### Search Results

```
Top 4 chunks retrieved (sorted by similarity):

  1. [sim=0.89] "Health monitors verify that pool members can
                 process traffic. BIG-IP checks server health..."

  2. [sim=0.84] "To configure a health monitor: 1. Navigate to
                 Local Traffic > Monitors. 2. Click Create..."

  3. [sim=0.79] "Monitor types include HTTP, HTTPS, TCP, ICMP...
                 Each has specific configuration options..."

  4. [sim=0.71] "Pool members are marked up or down based on
                 monitor responses. Timeouts and intervals..."
```

---

### 4.4 Context Assembly

#### What Happens

LangChain combines the retrieved document chunks with the original question into a formatted prompt that instructs the LLM how to respond.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | LangChain PromptTemplate + RetrievalQA |
| Hardware | CPU only (string formatting) |
| Input | Query string + retrieved chunk strings |
| Output | Formatted prompt string |

#### Prompt Template

```
┌─────────────────────────────────────────────────────────────────┐
│ Use the following context to answer the question. If you don't │
│ know the answer based on the context, say you don't know.      │
│                                                                 │
│ Context:                                                        │
│ {context}                                                       │
│                                                                 │
│ Question: {question}                                            │
│                                                                 │
│ Answer:                                                         │
└─────────────────────────────────────────────────────────────────┘
```

#### Assembled Prompt (Actual)

```
┌─────────────────────────────────────────────────────────────────┐
│ Use the following context to answer the question. If you don't │
│ know the answer based on the context, say you don't know.      │
│                                                                 │
│ Context:                                                        │
│ Health monitors verify that pool members can process traffic.  │
│ BIG-IP checks server health using configurable monitors...     │
│                                                                 │
│ To configure a health monitor: 1. Navigate to Local Traffic >  │
│ Monitors. 2. Click Create. 3. Select monitor type...           │
│                                                                 │
│ Monitor types include HTTP, HTTPS, TCP, ICMP. Each has         │
│ specific configuration options including intervals...          │
│                                                                 │
│ Pool members are marked up or down based on monitor responses. │
│ Timeouts and intervals determine how quickly failures...       │
│                                                                 │
│ Question: How do I configure a health monitor on F5 BIG-IP?    │
│                                                                 │
│ Answer:                                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.5 LLM Processing (TinyLlama)

#### What Happens

The assembled prompt is tokenized and processed through TinyLlama's 22 transformer layers. This is the most computationally intensive step, involving billions of matrix operations.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | TinyLlama-1.1B (via Unsloth) |
| Hardware | GPU (T4 with 16GB VRAM) |
| Model Size | ~2GB in 4-bit quantized form |
| Layers | 22 transformer layers |
| Attention Heads | 32 heads per layer |

#### Step 1: Prompt Tokenization

```
Prompt (~500 characters) ──► Tokenizer ──► ~150-200 token IDs

Token IDs: [1, 4803, 278, 1494, 3030, 304, 1234, ...]
           (each ID is an integer referencing the vocabulary)
```

#### Step 2: Initial Embedding

```
Each token ID ──► Embedding table lookup ──► 2048-dim vector

Token ID 4803 ──► [0.0123, -0.0456, 0.0789, ..., 0.0234]
                  (2048 values, one for each hidden dimension)

Result: Matrix of shape [seq_len × 2048]
        e.g., [175 tokens × 2048 dimensions]
```

#### Step 3: Transformer Layers (×22)

Each of the 22 layers performs these operations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONE TRANSFORMER LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [175 × 2048] matrix                                    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MULTI-HEAD ATTENTION                        │   │
│  │                                                          │   │
│  │  1. Project to Q, K, V (3 matrix multiplications)       │   │
│  │     Input × W_Q = Query   [175 × 2048] × [2048 × 2048] │   │
│  │     Input × W_K = Key     [175 × 2048] × [2048 × 2048] │   │
│  │     Input × W_V = Value   [175 × 2048] × [2048 × 2048] │   │
│  │                                                          │   │
│  │  2. Split into 32 attention heads (64 dims each)        │   │
│  │                                                          │   │
│  │  3. Compute attention scores (Q × K^T for each head)    │   │
│  │     [175 × 64] × [64 × 175] = [175 × 175] per head     │   │
│  │                                                          │   │
│  │  4. Apply softmax (normalize scores to probabilities)   │   │
│  │                                                          │   │
│  │  5. Multiply by Values (attention × V)                  │   │
│  │     [175 × 175] × [175 × 64] = [175 × 64] per head     │   │
│  │                                                          │   │
│  │  6. Concatenate heads and project                       │   │
│  │     [175 × 2048] × [2048 × 2048] = [175 × 2048]        │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼ (add residual connection + normalize)                │
│         │                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FEED-FORWARD NETWORK                        │   │
│  │                                                          │   │
│  │  1. Expand: [175 × 2048] × [2048 × 5632] = [175 × 5632]│   │
│  │  2. Activation function (SiLU)                          │   │
│  │  3. Contract: [175 × 5632] × [5632 × 2048] = [175×2048]│   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼ (add residual connection + normalize)                │
│                                                                 │
│  Output: [175 × 2048] matrix (same shape, refined content)    │
└─────────────────────────────────────────────────────────────────┘
```

#### Computation Scale

| Operation | Per Layer | All 22 Layers |
|-----------|-----------|---------------|
| Q, K, V projections | ~2.1B multiply-adds | ~46.2B |
| Attention scores | ~0.5B multiply-adds | ~11B |
| Attention output | ~0.5B multiply-adds | ~11B |
| Feed-forward | ~4.0B multiply-adds | ~88B |
| **TOTAL** | **~7.1B multiply-adds** | **~156B per forward pass** |

The T4 GPU can perform approximately 65 trillion operations per second (65 TFLOPS at FP16), making these billions of operations complete in milliseconds.

---

### 4.6 Response Generation (Autoregressive Decoding)

#### What Happens

The LLM generates text one token at a time. After processing the prompt, it predicts the most likely next token, adds it to the sequence, and repeats until done.

#### Where It Occurs

| Aspect | Details |
|--------|---------|
| Component | TinyLlama output layer + sampling |
| Hardware | GPU for computation, CPU for sampling |
| Output | One token ID per iteration |
| Speed | ~10-30 tokens/second on T4 |

#### Generation Process

```
AUTOREGRESSIVE GENERATION:

Step 1: Process full prompt through 22 layers
        Output: hidden state for last token position [1 × 2048]

Step 2: Project to vocabulary
        [1 × 2048] × [2048 × 32000] = [1 × 32000] (logits)
                                       ▲
                                       └── One score per vocabulary word

Step 3: Convert logits to probabilities (softmax)
        logits:       [2.3, -1.2, 0.8, 5.1, ...]
        probabilities: [0.02, 0.001, 0.01, 0.85, ...]
                                          ▲
                                          └── "To" has 85% probability

Step 4: Sample next token
        With temperature=0.7, sample from distribution
        Selected: token ID 1763 ("To")

Step 5: Append token and repeat
        New input: [...prompt..., "To"]
        Generate next: "configure"
        Generate next: "a"
        Generate next: "health"
        ... continue until <EOS> or max length
```

#### KV Cache Optimization

Without optimization, generating 100 tokens would require processing the entire sequence 100 times. The KV Cache stores intermediate Key and Value computations so they don't need to be recalculated:

```
Without KV Cache:
  Token 1:   Process [prompt]                    → 156B ops
  Token 2:   Process [prompt + token1]           → 157B ops
  Token 3:   Process [prompt + token1 + token2]  → 158B ops
  ...
  Token 100: Process [prompt + 99 tokens]        → 255B ops
  TOTAL: ~20 trillion operations

With KV Cache:
  Token 1:   Process [prompt], cache K,V         → 156B ops
  Token 2:   Process [token1] only, use cache    → 1.5B ops
  Token 3:   Process [token2] only, use cache    → 1.5B ops
  ...
  Token 100: Process [token99] only, use cache   → 1.5B ops
  TOTAL: ~300 billion operations (66× faster!)
```

---

## 5. Inside the Neural Networks: Matrix Operations

All neural network computation reduces to matrix multiplication. Understanding this helps demystify what's happening inside the models.

### What is Matrix Multiplication?

Matrix multiplication combines two grids of numbers to produce a new grid. Each element in the output is a dot product (multiply corresponding elements and sum):

```
Simple Example (2×2 matrices):

    [1  2]       [5  6]       [1×5+2×7  1×6+2×8]     [19  22]
    [3  4]   ×   [7  8]   =   [3×5+4×7  3×6+4×8]  =  [43  50]

    Matrix A     Matrix B           Calculation        Result
     (2×2)        (2×2)                                 (2×2)
```

### Matrix Operations in Embedding

Converting text to vectors:

```
Input: 5 tokens, each represented as one-hot vector (size 32000)

    [0,0,1,0,...,0]     ┐
    [0,0,0,0,...,1]     │
    [0,1,0,0,...,0]     ├── Input matrix [5 × 32000]
    [0,0,0,1,...,0]     │
    [1,0,0,0,...,0]     ┘

         ×

    Embedding table [32000 × 384]  ◄── Learned during training

         =

    [0.12, -0.34, 0.56, ..., 0.23]    ┐
    [0.45, 0.67, -0.89, ..., -0.12]   │
    [-0.23, 0.45, 0.78, ..., 0.34]    ├── Token embeddings [5 × 384]
    [0.89, -0.12, 0.34, ..., 0.56]    │
    [0.34, 0.78, -0.45, ..., -0.67]   ┘
```

### Matrix Operations in Attention

Computing which tokens should attend to which:

```
Step 1: Create Query, Key, Value

    Input [5 × 384]  ×  W_query [384 × 384]  =  Query [5 × 384]
    Input [5 × 384]  ×  W_key [384 × 384]    =  Key [5 × 384]
    Input [5 × 384]  ×  W_value [384 × 384]  =  Value [5 × 384]

Step 2: Compute attention scores

    Query [5 × 384]  ×  Key^T [384 × 5]  =  Scores [5 × 5]

    Scores matrix shows which tokens attend to which:
                        Token1  Token2  Token3  Token4  Token5
         Token1 attends:  0.9    0.05    0.02    0.02    0.01
         Token2 attends:  0.1    0.7     0.1     0.05    0.05
         Token3 attends:  0.05   0.15    0.6     0.15    0.05
         Token4 attends:  0.02   0.08    0.2     0.6     0.1
         Token5 attends:  0.01   0.04    0.1     0.25    0.6

Step 3: Apply attention to values

    Scores [5 × 5]  ×  Value [5 × 384]  =  Output [5 × 384]

    Each output row is a weighted combination of all Value rows,
    weighted by the attention scores.
```

### Matrix Operations in Feed-Forward

Two-layer network that processes each position:

```
    Input [5 × 384]  ×  W1 [384 × 1536]  =  Hidden [5 × 1536]
                                                    │
                                          Apply ReLU/SiLU activation
                                                    │
                                                    ▼
    Hidden [5 × 1536]  ×  W2 [1536 × 384]  =  Output [5 × 384]

The expansion to 1536 (4× the 384 dims) gives the network
more "room" to learn complex patterns.
```

---

## 6. Fine-Tuning Data Flow

Fine-tuning adds a training loop that adjusts model weights based on example data. Here's how data flows during training.

### Training Data Input

```
Training example (JSONL format):

{
  "instruction": "How do I configure a pool?",
  "output": "To configure a pool on F5 BIG-IP: 1. Navigate..."
}

Formatted for training:

### Instruction:
How do I configure a pool?

### Response:
To configure a pool on F5 BIG-IP: 1. Navigate...
```

### Forward Pass (Same as Inference)

```
Training text ──► Tokenize ──► Embed ──► 22 Layers ──► Logits
                                                         │
                                                         ▼
                                              Predicted token probs
```

### Loss Calculation

The loss measures how wrong the model's predictions were:

```
Example for one position:

  Actual next token: "pool" (ID: 4521)

  Model predicted probabilities:
    "the":    0.15
    "pool":   0.08  ◄── Should be higher!
    "server": 0.12
    "a":      0.10
    ... (32000 words)

  Loss = -log(0.08) = 2.53  ◄── High loss = bad prediction

  If model had predicted "pool" with 0.90 probability:
  Loss = -log(0.90) = 0.11  ◄── Low loss = good prediction
```

### Backward Pass (Gradient Computation)

Gradients tell us how to adjust each weight to reduce the loss:

```
Loss ──► Compute gradients for all weights ──► Store gradients

For each weight W in the network:
  gradient = ∂Loss/∂W
  (how much would loss change if we slightly changed W?)

Gradient computation flows backward through the network:

  Layer 22 gradients ──► Layer 21 gradients ──► ... ──► Layer 1

This is called "backpropagation" - computing gradients by
propagating error information backward through the network.
```

### LoRA Weight Update

With LoRA, only the small adapter matrices are updated:

```
Original weights W: [2048 × 2048] = 4.2M params ◄── FROZEN

LoRA adapter A: [2048 × 16] = 32,768 params  ◄── TRAINABLE
LoRA adapter B: [16 × 2048] = 32,768 params  ◄── TRAINABLE

Update rule (simplified):

  A_new = A_old - learning_rate × gradient_A
  B_new = B_old - learning_rate × gradient_B

Example:
  learning_rate = 0.0002
  gradient for A[0][0] = 0.05

  A[0][0]_new = A[0][0]_old - 0.0002 × 0.05
             = 0.0341 - 0.00001
             = 0.03409

This tiny adjustment, repeated millions of times across many
examples, gradually teaches the model the new task.
```

---

## 7. Hardware Mapping: CPU vs GPU Operations

Different parts of the pipeline run on different hardware. Understanding this helps with debugging and optimization.

### Complete Operation Mapping

| Operation | Hardware | Why |
|-----------|----------|-----|
| File loading | CPU + Disk | I/O operation, no parallelism needed |
| Text chunking | CPU | String manipulation, sequential |
| Tokenization | CPU | Dictionary lookups, not parallelizable |
| Embedding lookup | GPU | Large table lookup, benefits from memory bandwidth |
| Attention computation | GPU | Massive matrix multiplications |
| Feed-forward layers | GPU | Matrix multiplications |
| Softmax | GPU | Element-wise operations on large tensors |
| Vector similarity search | CPU | ChromaDB uses CPU-based HNSW |
| Prompt formatting | CPU | String concatenation |
| Token sampling | CPU | Random sampling from distribution |
| Gradient computation | GPU | Backprop through matrix operations |
| Weight updates | GPU | Element-wise operations on weight tensors |

### Memory Locations

| Data | Location | Size (approx) |
|------|----------|---------------|
| TinyLlama weights (4-bit) | GPU VRAM | ~2 GB |
| KV Cache (inference) | GPU VRAM | ~0.5-2 GB depending on context |
| Embedding model weights | GPU VRAM (or RAM) | ~80 MB |
| ChromaDB index | System RAM + Disk | ~10-50 MB for lab data |
| Document chunks (text) | System RAM | ~1-5 MB |
| Activations (training) | GPU VRAM | ~2-4 GB |
| Gradients (training) | GPU VRAM | ~0.5 GB (LoRA only) |
| Optimizer states | GPU VRAM | ~0.5 GB (LoRA only) |

### Data Transfer Bottlenecks

Data moving between CPU and GPU is a common bottleneck:

```
CPU RAM ◄────────────────────────────────► GPU VRAM
         PCIe bus: ~32 GB/s max

Slow transfers:
  • Loading model weights at startup (~2GB, takes ~0.5 seconds)
  • Copying token IDs to GPU (small, but frequent)
  • Copying generated tokens back to CPU

Optimizations used:
  • Batch multiple tokens together
  • Keep model weights on GPU permanently
  • Use pinned memory for faster CPU-GPU transfer
```

---

## 8. Complete Data Flow Diagram

This diagram shows the entire system with all data transformations:

```
══════════════════════════════════════════════════════════════════════════════
                         COMPLETE RAG SYSTEM DATA FLOW
══════════════════════════════════════════════════════════════════════════════

PHASE 1: DOCUMENT INGESTION (runs once at setup)
─────────────────────────────────────────────────────────────────────────────

  F5 Docs (6 files, ~50KB total)
       │
       ▼ [CPU: TextLoader]
  Document objects in RAM
       │
       ▼ [CPU: RecursiveCharacterTextSplitter]
  ~100 chunks (500 chars each)
       │
       ▼ [GPU: all-MiniLM-L6-v2]
       │   ├── Tokenize (CPU): text → token IDs
       │   ├── Embed lookup (GPU): IDs → vectors [seq × 384]
       │   ├── 6 Transformer layers (GPU): ~50M ops/chunk
       │   ├── Mean pooling (GPU): [seq × 384] → [1 × 384]
       │   └── Normalize (GPU): scale to unit length
       │
  100 embeddings [100 × 384] floats
       │
       ▼ [CPU: ChromaDB]
  Stored in ./chroma_db/ with HNSW index


PHASE 2: QUERY PROCESSING (runs for each question)
─────────────────────────────────────────────────────────────────────────────

  User: "How do I configure a health monitor?"
       │
       ▼ [GPU: all-MiniLM-L6-v2] (same as above)
  Query embedding [1 × 384]
       │
       ▼ [CPU: ChromaDB HNSW search]
       │   └── Cosine similarity: query · each_doc_embedding
       │
  Top 4 similar chunks retrieved
       │
       ▼ [CPU: LangChain PromptTemplate]
  Assembled prompt (~500 tokens):
  "Use the following context... [chunks] ...Question: [query]"
       │
       ▼ [CPU: Tokenizer]
  Token IDs [1 × ~500]
       │
       ▼ [GPU: TinyLlama-1.1B]
       │   ├── Embed lookup: IDs → [500 × 2048]
       │   ├── Layer 1-22 (each layer):
       │   │     ├── Multi-head attention: ~7B ops
       │   │     └── Feed-forward: ~12B ops
       │   ├── Total: ~156B ops for prompt
       │   │
       │   ├── Generate token 1:
       │   │     ├── Output projection: [2048] → [32000] logits
       │   │     ├── Softmax: logits → probabilities
       │   │     └── Sample: select token ID
       │   │
       │   ├── Generate tokens 2-100 (with KV cache):
       │   │     └── ~1.5B ops per token
       │   │
       │   └── Stop at </s> or max_tokens
       │
  Generated token IDs [1 × ~100]
       │
       ▼ [CPU: Tokenizer.decode]
  Response text: "To configure a health monitor: 1. Navigate..."
       │
       ▼
  Displayed to user

══════════════════════════════════════════════════════════════════════════════
```

---

## 9. Glossary

| Term | Definition |
|------|------------|
| Activation Function | Non-linear function (like ReLU, SiLU) applied between layers to enable learning complex patterns |
| Attention | Mechanism that computes which parts of input to focus on when producing each output |
| Autoregressive | Generating output one element at a time, each depending on previous elements |
| Backpropagation | Algorithm for computing gradients by propagating error backward through network |
| Cosine Similarity | Measure of similarity based on angle between vectors (1 = identical, 0 = unrelated) |
| Dot Product | Sum of element-wise products: [a,b,c]·[x,y,z] = ax + by + cz |
| Embedding | Dense vector representation of discrete items (words, tokens) in continuous space |
| Gradient | Derivative indicating direction and magnitude for adjusting weights to reduce loss |
| HNSW | Hierarchical Navigable Small World - graph-based index for fast similarity search |
| KV Cache | Stored Key and Value tensors from previous tokens to avoid recomputation |
| Logits | Raw output scores before softmax; higher logit = higher probability after softmax |
| Loss | Number measuring how wrong predictions are; training tries to minimize this |
| Matrix Multiplication | Core operation combining two matrices: [m×n] × [n×p] → [m×p] |
| Mean Pooling | Averaging multiple vectors into one; used to get single sentence embedding |
| Normalization | Scaling vectors to unit length or activations to standard distribution |
| Residual Connection | Adding layer input to layer output; helps gradients flow in deep networks |
| Softmax | Function converting logits to probabilities that sum to 1 |
| Token | Basic unit of text processing; roughly a word or word piece |
| Transformer | Neural network architecture using self-attention; basis of modern LLMs |
| Vector | Ordered list of numbers; in ML, often represents meaning in high-dimensional space |
| VRAM | Video RAM; GPU memory where model weights and computations reside |

---

*End of Document*
