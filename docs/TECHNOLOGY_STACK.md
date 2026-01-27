# LLM Fine-Tuning & RAG Lab
## Comprehensive Technology Stack Documentation

**F5 AI Technical Assistant Training Lab**

Repository: [github.com/therealnoof/llm-finetuning-rag-lab](https://github.com/therealnoof/llm-finetuning-rag-lab)

Version 1.1 | January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Technologies](#3-core-technologies)
   - 3.1 [TinyLlama-1.1B-Chat](#31-tinyllama-11b-chat)
   - 3.2 [Fine-Tuning Framework: Unsloth + QLoRA + PEFT](#32-fine-tuning-framework-unsloth--qlora--peft)
   - 3.3 [LangChain - RAG Orchestration Framework](#33-langchain---rag-orchestration-framework)
   - 3.4 [ChromaDB - Vector Database](#34-chromadb---vector-database)
   - 3.5 [Embedding Model: all-MiniLM-L6-v2](#35-embedding-model-all-minilm-l6-v2)
4. [Compute Environments](#4-compute-environments)
   - 4.1 [Google Colab (Public Option)](#41-google-colab-public-option)
   - 4.2 [AWS EC2 G4dn Instance (Private Option)](#42-aws-ec2-g4dn-instance-private-option)
   - 4.3 [Environment Comparison](#43-environment-comparison)
   - 4.4 [PyTorch & CUDA](#44-pytorch--cuda)
   - 4.5 [Hugging Face Transformers](#45-hugging-face-transformers)
5. [Data Architecture](#5-data-architecture)
6. [Lab Module Breakdown](#6-lab-module-breakdown)
7. [Glossary of Terms](#7-glossary-of-terms)

---

## 1. Executive Summary

This document provides a comprehensive overview of all technologies used in the LLM Fine-Tuning and RAG (Retrieval-Augmented Generation) Student Lab. The lab transforms a general-purpose language model (TinyLlama-1.1B) into an F5 domain expert through a combination of RAG and fine-tuning techniques.

The lab supports two deployment options: Google Colab's free tier for public access, or a private AWS EC2 instance with T4 GPU for secure, controlled environments. Both options provide sufficient compute resources to complete the entire lab in approximately 2 hours.

### Learning Objectives

- Load and run quantized LLMs on limited hardware
- Build a RAG system with LangChain and ChromaDB
- Fine-tune models using QLoRA for domain specialization
- Evaluate and compare different LLM enhancement approaches

### Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Base Model | TinyLlama-1.1B-Chat | Small LLM for text generation (2GB quantized) |
| Fine-tuning | Unsloth + QLoRA + PEFT | Memory-efficient model adaptation |
| RAG Framework | LangChain | Orchestrates retrieval and generation pipeline |
| Vector Database | ChromaDB | Stores and searches document embeddings |
| Embeddings | all-MiniLM-L6-v2 | Converts text to semantic vectors |
| Environment (Public) | Google Colab (T4 GPU) | Free cloud computing platform |
| Environment (Private) | AWS EC2 G4dn (T4 GPU) | Secure private cloud instance |

---

## 2. Architecture Overview

The lab architecture consists of interconnected layers that work together to create an intelligent F5 technical assistant.

### System Layers

| Layer | Components | Data Flow |
|-------|------------|-----------|
| User Interface | Jupyter Notebooks | Receives user queries and displays responses |
| Orchestration | LangChain | Routes queries, manages context, coordinates components |
| Retrieval | ChromaDB + Embeddings | Finds relevant documents for context |
| Generation | TinyLlama (base or fine-tuned) | Produces natural language responses |
| Training | Unsloth + QLoRA | Adapts model to F5 domain (offline) |

### Data Flow Explanation

When a user asks a question like "How do I configure load balancing on F5 BIG-IP?", the system processes it through several stages:

1. **Step 1 - Query Embedding**: The user's question is converted into a numerical vector using the all-MiniLM-L6-v2 model.
2. **Step 2 - Document Retrieval**: ChromaDB searches its index to find the most semantically similar text chunks.
3. **Step 3 - Prompt Construction**: LangChain combines the original question with the retrieved context.
4. **Step 4 - Response Generation**: TinyLlama generates a response using the provided context.
5. **Step 5 - Output**: The generated response is displayed to the user.

---

## 3. Core Technologies

### 3.1 TinyLlama-1.1B-Chat

TinyLlama is a compact open-source large language model with 1.1 billion parameters. Despite its small size compared to GPT-4, it demonstrates remarkable language understanding capabilities.

#### What is a Language Model?

A language model is AI that has learned patterns in human language by analyzing massive amounts of text. It can predict what words come next in a sentence and generate coherent, contextually appropriate text.

#### Technical Specifications

| Specification | Value | What This Means |
|---------------|-------|-----------------|
| Parameters | 1.1 billion | Adjustable values learned during training. More = more capability but more memory. |
| Architecture | Llama 2 Transformer | Same design as Meta's Llama models with attention mechanisms. |
| Context Length | 2,048 tokens | Can process about 1,500 words at once. |
| Quantized Size | ~2GB (4-bit) | Compressed to use less memory while maintaining capability. |
| Training Data | 3 trillion tokens | Trained on diverse internet text including code and books. |
| License | Apache 2.0 | Free for commercial and educational use. |

#### How Transformers Work

TinyLlama uses a Transformer architecture, which is the foundation of all modern language models including ChatGPT and Claude. Here's a simplified explanation:

- **Tokenization**: Text is split into "tokens" (word pieces). Each token gets a number.
- **Embeddings**: Each token number is converted to a vector capturing its meaning.
- **Attention**: The model calculates how much each token should "pay attention" to every other token for context.
- **Feed-Forward Networks**: Each position passes through neural network layers that transform the information.
- **Layer Stacking**: TinyLlama has 22 layers. Early layers learn basic patterns, later layers learn abstract concepts.
- **Output**: The final layer predicts probabilities for the next token.

#### Why TinyLlama for This Lab?

- **Size**: Small enough for T4 GPU (16GB VRAM) while leaving room for fine-tuning
- **Speed**: Generates responses quickly enough for interactive learning
- **Quality**: Produces coherent responses despite its small size
- **Trainability**: Can be fine-tuned with QLoRA in under 30 minutes
- **Educational**: Same architecture as industry models, concepts transfer to larger systems

---

### 3.2 Fine-Tuning Framework: Unsloth + QLoRA + PEFT

Fine-tuning is taking a pre-trained model and training it further on specific data. This lab uses three technologies together to make fine-tuning possible on limited hardware.

#### What is Fine-Tuning?

Imagine TinyLlama as someone with a general education. Fine-tuning is like sending them to specialized training for F5 technologies. After fine-tuning, the model uses F5 terminology naturally and provides more accurate responses.

#### QLoRA (Quantized Low-Rank Adaptation)

QLoRA is a breakthrough technique that makes fine-tuning accessible:

- **Quantization**: The base model is compressed to 4-bit precision, reducing memory by 8x.
- **Low-Rank Adaptation (LoRA)**: Instead of updating all 1.1B parameters, LoRA adds small adapter matrices (4-8M parameters) that learn the new task. Original weights stay frozen.
- **Memory Savings**: QLoRA allows training with under 8GB VRAM instead of 50+ GB.

#### How LoRA Works

Traditional fine-tuning updates a 4096x4096 weight matrix (16.7M values). LoRA approximates changes with two smaller matrices:

- **Matrix A**: 4096 x 8 = 32,768 values
- **Matrix B**: 8 x 4096 = 32,768 values
- **Total**: 65,536 values instead of 16.7 million (0.4% of original!)

During inference, the LoRA adapter adds its contribution: `Output = W*x + (A*B)*x`. The original weights W never change, so you can easily swap different LoRA adapters for different tasks.

#### PEFT (Parameter-Efficient Fine-Tuning)

PEFT is Hugging Face's library implementing LoRA and other efficient fine-tuning methods, providing easy integration and adapter management.

#### Unsloth

Unsloth makes fine-tuning 2x faster and uses 60% less memory through custom CUDA kernels and efficient memory management.

#### Fine-Tuning Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| LoRA Rank (r) | 16 | Size of adapter matrices. Higher = more capacity but more memory. |
| LoRA Alpha | 32 | Scaling factor. Usually 2x the rank. |
| Target Modules | q_proj, k_proj, v_proj, o_proj | Which attention layers get LoRA adapters. |
| Learning Rate | 2e-4 | How fast the model learns. |
| Batch Size | 4 | Examples processed together. |
| Gradient Accumulation | 4 | Simulates batch size of 16 to stabilize training. |
| Epochs | 1 | Times through the training data. |
| Training Examples | 150+ | Question-answer pairs about F5 technologies. |

---

### 3.3 LangChain - RAG Orchestration Framework

LangChain is the industry-standard framework for building LLM-powered applications. It provides the "plumbing" connecting different AI components together.

#### What Problem Does LangChain Solve?

Building an AI application involves many steps: loading documents, splitting text, creating embeddings, storing vectors, retrieving context, constructing prompts, calling the LLM. LangChain provides pre-built, tested components for each step.

#### Key LangChain Components

| Component | Class/Module | What It Does |
|-----------|--------------|--------------|
| Document Loaders | TextLoader, DirectoryLoader | Reads files into Document objects. |
| Text Splitters | RecursiveCharacterTextSplitter | Breaks documents into smaller chunks. |
| Embeddings | HuggingFaceEmbeddings | Integrates sentence-transformers for vectors. |
| Vector Stores | Chroma | Interface to ChromaDB for storing/searching. |
| Retrievers | VectorStoreRetriever | Finds relevant documents for a query. |
| Chains | RetrievalQA | Combines retriever + LLM into a pipeline. |
| Prompts | PromptTemplate | Formats context and question for the LLM. |

#### The RAG Pipeline in Detail

1. **Document Loading**: TextLoader reads F5 documentation files from `data/f5_docs/`
2. **Text Splitting**: RecursiveCharacterTextSplitter divides documents into ~500 character chunks with overlap.
3. **Embedding Generation**: Each chunk passes through all-MiniLM-L6-v2 to create a 384-dim vector.
4. **Vector Storage**: ChromaDB stores each chunk's text alongside its vector.
5. **Query Processing**: User questions are embedded with the same model.
6. **Similarity Search**: ChromaDB finds the 4 most similar chunks using cosine similarity.
7. **Prompt Assembly**: LangChain formats a prompt with retrieved chunks as context.
8. **LLM Generation**: TinyLlama generates a response using the context.

---

### 3.4 ChromaDB - Vector Database

ChromaDB is an open-source embedding database designed for AI applications. It stores, indexes, and searches vector embeddings.

#### What is a Vector Database?

Traditional databases search by exact matches. Vector databases search by similarity: find documents semantically similar to a query, even if they share no words in common. This is crucial for RAG.

#### How Vector Search Works

- **Embedding**: Each document is converted to a vector. Similar texts have similar vectors.
- **Indexing**: ChromaDB builds an index (HNSW) for fast approximate nearest neighbor search.
- **Query**: The query is embedded and the index finds the K closest vectors.
- **Similarity Metric**: Distance is measured by cosine similarity - smaller angle = more similar.

#### ChromaDB Features

| Feature | Usage | Explanation |
|---------|-------|-------------|
| Local Storage | Persist to ./chroma_db | Data saved to disk, survives restarts |
| Collection | f5_docs | Named group of embeddings, like a table |
| Embedding Function | all-MiniLM-L6-v2 | Automatically embeds text when adding/querying |
| Metadata | source, chunk_id | Extra info stored with each embedding |
| Similarity Search | query(query_texts, n_results) | Find top N similar documents |

#### Why ChromaDB?

- **No API Keys**: Runs entirely locally, no signup required
- **Zero Configuration**: Works out of the box with sensible defaults
- **Python Native**: Designed for Python-first workflows like Jupyter notebooks
- **LangChain Integration**: First-class support in LangChain ecosystem

---

### 3.5 Embedding Model: all-MiniLM-L6-v2

This sentence-transformers model converts text into 384-dimensional vectors capturing semantic meaning. It translates human language into mathematical space where similarity can be computed.

#### What Are Embeddings?

An embedding is a list of numbers representing text where similar texts have similar numbers:

- "The cat sat on the mat" → [0.12, -0.45, 0.78, ...] (384 numbers)
- "A feline rested on a rug" → [0.11, -0.44, 0.77, ...] (similar vector!)
- "Load balancing distributes traffic" → [-0.56, 0.23, 0.91, ...] (very different)

#### Model Specifications

| Specification | Value | Significance |
|---------------|-------|--------------|
| Model Name | sentence-transformers/all-MiniLM-L6-v2 | Full Hugging Face identifier |
| Dimensions | 384 | Output vector length. Smaller than many models for efficiency. |
| Max Tokens | 256 | Maximum input length. |
| Model Size | ~80MB | Small enough to load quickly |
| Speed | ~2800 sentences/second on GPU | Fast for real-time applications |
| License | Apache 2.0 | Free for all uses |

---

## 4. Compute Environments

The lab supports two deployment options to accommodate different organizational requirements: Google Colab for public/educational use, and AWS EC2 for private/enterprise environments.

### 4.1 Google Colab (Public Option)

Google Colaboratory is a free cloud-based Jupyter notebook environment providing GPU access. Ideal for public training sessions and self-paced learning.

#### Colab Advantages

- **Free GPU Access**: T4 GPU with 16GB VRAM at no cost
- **Zero Setup**: No local installation required, works in any browser
- **Easy Sharing**: Distribute lab materials via simple Colab links
- **Pre-installed Libraries**: Most ML libraries available or easily added

#### Colab Specifications

| Resource | Free Tier | Notes |
|----------|-----------|-------|
| GPU | Tesla T4 (16GB VRAM) | Enable: Runtime > Change runtime type > T4 GPU |
| RAM | ~12.7 GB | Sufficient for this lab |
| Disk | ~78 GB | Ephemeral - cleared when disconnected |
| Session Length | ~12 hours max | Disconnects after idle timeout |
| Network | Public internet | Requires Google account |

#### Colab Limitations

- **Session Persistence**: Runtime disconnects after idle period; work may be lost
- **Data Privacy**: Code and data processed on Google infrastructure
- **Network Restrictions**: Cannot access private/internal resources
- **Resource Availability**: GPU availability not guaranteed during peak times

---

### 4.2 AWS EC2 G4dn Instance (Private Option)

For organizations requiring data privacy, network isolation, or consistent resource availability, the lab can be deployed on AWS EC2 G4dn instances featuring NVIDIA T4 GPUs.

#### Why AWS for Private Deployments?

- **Data Sovereignty**: All data remains within your AWS account and chosen region
- **Network Isolation**: Deploy in private VPC with no public internet exposure
- **Consistent Availability**: Dedicated GPU resources, no contention with other users
- **Integration**: Connect to internal data sources, Active Directory, and corporate systems
- **Compliance**: Meet regulatory requirements (FedRAMP, HIPAA, etc.) with appropriate configurations

#### Recommended Instance Type

| Specification | g4dn.xlarge | Notes |
|---------------|-------------|-------|
| GPU | 1x NVIDIA T4 (16GB VRAM) | Same GPU as Colab free tier |
| vCPUs | 4 | Intel Cascade Lake |
| RAM | 16 GB | More than Colab |
| Storage | 125 GB NVMe SSD | Persistent, fast local storage |
| Network | Up to 25 Gbps | High bandwidth for data transfer |
| On-Demand Price | ~$0.526/hour | US East region, subject to change |
| Spot Price | ~$0.16-0.20/hour | Up to 70% savings, may be interrupted |

#### AWS Environment Setup

The following components are required for the AWS deployment:

- **EC2 Instance**: g4dn.xlarge (or g4dn.2xlarge for larger classes)
- **AMI**: Deep Learning AMI (Ubuntu) - pre-installed NVIDIA drivers and CUDA
- **Storage**: 100+ GB EBS volume for model weights and datasets
- **Security Group**: Allow SSH (22) and Jupyter (8888) from authorized IPs only
- **IAM Role**: Optional - for S3 access to training data

#### AWS Deep Learning AMI

AWS provides pre-configured Deep Learning AMIs that include all necessary drivers and frameworks:

- **NVIDIA Driver**: Pre-installed and configured for T4 GPU
- **CUDA Toolkit**: Version 11.x or 12.x depending on AMI version
- **PyTorch**: Pre-installed with GPU support
- **Conda Environments**: Isolated Python environments for different frameworks
- **JupyterLab**: Pre-configured for remote notebook access

#### Connecting to AWS Instance

Students access the lab environment via SSH tunnel to JupyterLab:

```bash
# 1. Connect via SSH
ssh -i key.pem -L 8888:localhost:8888 ubuntu@<instance-ip>

# 2. Activate environment
conda activate pytorch

# 3. Start Jupyter
jupyter lab --no-browser --port=8888

# 4. Open browser: Navigate to http://localhost:8888
```

#### Cost Optimization Strategies

- **Spot Instances**: Use for non-critical training; 60-70% cost savings
- **Instance Scheduling**: Stop instances outside lab hours
- **Right-sizing**: g4dn.xlarge sufficient for individual use; g4dn.2xlarge for shared access
- **Reserved Instances**: Consider for recurring training programs (up to 40% savings)

---

### 4.3 Environment Comparison

Choose the appropriate environment based on your organization's requirements:

| Factor | Google Colab | AWS EC2 G4dn |
|--------|--------------|--------------|
| Cost | Free | $0.16-0.53/hour |
| Setup Time | Instant | 15-30 minutes |
| Data Privacy | Google infrastructure | Your AWS account |
| Network Access | Public internet only | Private VPC supported |
| Session Persistence | Limited (disconnects) | Persistent until stopped |
| GPU Availability | Not guaranteed | Guaranteed (dedicated) |
| Compliance | Limited | FedRAMP, HIPAA capable |
| Best For | Public training, self-study | Enterprise, sensitive data |

---

### 4.4 PyTorch & CUDA

PyTorch is the deep learning framework underlying all model operations. CUDA enables GPU acceleration on both Colab and AWS environments.

- **Tensor Operations**: Multi-dimensional array computations with GPU support
- **Automatic Differentiation**: Computes gradients for training
- **Neural Network Modules**: Pre-built layers, loss functions, optimizers
- **CUDA**: NVIDIA's platform allowing PyTorch to run on T4 GPU (2,560 CUDA cores)

---

### 4.5 Hugging Face Transformers

Library providing pre-trained models and NLP tools.

| Component | Class | Purpose |
|-----------|-------|---------|
| Model Loading | AutoModelForCausalLM | Loads TinyLlama |
| Tokenization | AutoTokenizer | Converts text to/from tokens |
| Training | Trainer, TrainingArguments | High-level training loop |
| PEFT Integration | PeftModel, LoraConfig | Adds LoRA adapters |
| Quantization | BitsAndBytesConfig | Configures 4-bit quantization |

---

## 5. Data Architecture

The lab uses two data types: documentation for RAG retrieval and Q&A pairs for fine-tuning.

### RAG Knowledge Base (data/f5_docs/)

| File | Content | Purpose |
|------|---------|---------|
| bigip_basics.txt | BIG-IP architecture and concepts | Foundation knowledge |
| irules_guide.txt | iRules scripting syntax | Traffic manipulation |
| load_balancing.txt | Load balancing algorithms | Traffic distribution |
| ssl_offloading.txt | SSL/TLS termination | Encryption handling |
| health_monitors.txt | Service health checking | Availability monitoring |
| troubleshooting.txt | Common issues and solutions | Problem resolution |

### Fine-Tuning Data (data/training/)

- **f5_qa_train.jsonl**: 150+ training examples in instruction/response format
- **f5_qa_eval.jsonl**: 30+ held-out examples for measuring performance

---

## 6. Lab Module Breakdown

| Module | Duration | Technologies | Outcome |
|--------|----------|--------------|---------|
| 01 - Setup & Base Model | 20 min | Colab/AWS, PyTorch, Transformers, Unsloth | Load model, run baseline |
| 02 - RAG System | 45 min | LangChain, ChromaDB, sentence-transformers | Build retrieval pipeline |
| 03 - Fine-tuning with QLoRA | 30 min | Unsloth, PEFT, LoRA, Trainer | Train domain adapter |
| 04 - Comparison & Evaluation | 25 min | All above + matplotlib | Compare approaches |

### Module 1: Setup & Base Model

Configure the compute environment (Colab or AWS), install dependencies, load TinyLlama-1.1B in 4-bit quantized form, and run baseline queries to understand the model's capabilities.

### Module 2: RAG System

Build a complete RAG pipeline: load F5 documentation, split into chunks, create embeddings, store in ChromaDB, and wire everything together with LangChain.

### Module 3: Fine-tuning with QLoRA

Configure and run QLoRA fine-tuning using Unsloth. Prepare training data, set hyperparameters, monitor training loss, and save the LoRA adapter (~5 min on T4).

### Module 4: Comparison & Evaluation

Test the same questions across three configurations: base model, base + RAG, and fine-tuned + RAG. Compare response quality.

---

## 7. Glossary of Terms

| Term | Definition |
|------|------------|
| AMI | Amazon Machine Image - pre-configured virtual machine template for EC2 |
| Attention | Mechanism allowing models to focus on relevant parts of input |
| CUDA | NVIDIA's parallel computing platform for GPU acceleration |
| EC2 | Amazon Elastic Compute Cloud - virtual servers in AWS |
| Embedding | Numerical vector representation of text capturing semantic meaning |
| Epoch | One complete pass through the entire training dataset |
| Fine-tuning | Further training a pre-trained model on task-specific data |
| G4dn | AWS instance family with NVIDIA T4 GPUs |
| Gradient | Mathematical derivative indicating how to adjust weights |
| Inference | Using a trained model to generate outputs |
| LoRA | Low-Rank Adaptation - efficient fine-tuning with small adapter matrices |
| LLM | Large Language Model - AI trained to understand and generate language |
| Parameter | Learnable value in a neural network |
| Quantization | Reducing numerical precision to save memory |
| RAG | Retrieval-Augmented Generation - enhancing LLM with retrieved context |
| T4 GPU | NVIDIA Tesla T4 - inference-optimized GPU with 16GB VRAM |
| Token | Basic unit of text processing, roughly a word or subword |
| Transformer | Neural network architecture using self-attention |
| Vector Database | Database optimized for storing and searching embedding vectors |
| VPC | Virtual Private Cloud - isolated network in AWS |
| VRAM | Video RAM - GPU memory for model parameters |

---

*End of Document*
