# F5 AI Technical Assistant Training Lab

A hands-on 2-3 hour training lab where students create an LLM-powered F5 technical assistant using RAG (Retrieval-Augmented Generation) and fine-tuning techniques. Transform a general-purpose model (TinyLlama-1.1B) into an F5 domain expert.

## Learning Objectives

By the end of this lab, students will be able to:
- Load and run quantized LLMs on limited hardware
- Build a RAG system with LangChain and ChromaDB
- Fine-tune models using QLoRA for domain specialization
- Evaluate and compare different LLM enhancement approaches

## Prerequisites

- Basic Python programming knowledge
- Familiarity with machine learning concepts
- Google account (for Colab access)
- No local GPU required - everything runs on Colab free tier

## Tech Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| Base Model | TinyLlama-1.1B-Chat | Small (2GB quantized), fits Colab free tier |
| Fine-tuning | Unsloth + QLoRA + PEFT | 2x faster, memory efficient |
| RAG Framework | LangChain | Industry standard |
| Vector DB | ChromaDB | Local, no API keys |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, fast |
| Environment | Google Colab (T4 GPU) | Free tier sufficient |

## Lab Structure

| Module | Duration | Description |
|--------|----------|-------------|
| 01 - Setup & Base Model | 20 min | Environment setup, load model, baseline evaluation |
| 02 - RAG System | 45 min | Build retrieval-augmented generation pipeline |
| 03 - Fine-tuning with QLoRA | 60 min | Train model on F5-specific data |
| 04 - Comparison & Evaluation | 25 min | Compare approaches, visualize results |

## Quick Start

### Option 1: Google Colab (Recommended)

Go to Google Colab and create a Free account. https://colab.research.google.com/
- We are recommending using Google Colab for two reasons, 1. It offers a free GPU and 2. the serverless functions are also free
- After creating a free account Git clone this repo in VSCode or have the notebooks on your desktop for easy upload.

1. Open the notebooks in Google Colab:
   - [Module 1: Setup & Base Model](notebooks/01_Setup_and_Base_Model.ipynb)
   - [Module 2: RAG System](notebooks/02_RAG_System.ipynb)
   - [Module 3: Fine-tuning with QLoRA](notebooks/03_FineTuning_QLoRA.ipynb)
   - [Module 4: Comparison & Evaluation](notebooks/04_Comparison_Evaluation.ipynb)

2. Ensure GPU is enabled:
   - Runtime > Change runtime type > T4 GPU

3. Follow the notebooks sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/therealnoof/llm-finetuning-rag-lab.git
cd llm-finetuning-rag-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: Local setup requires a CUDA-capable GPU with at least 6GB VRAM.

## Project Structure

```
LLM-FineTuning-RAG-Lab/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── notebooks/
│   ├── 01_Setup_and_Base_Model.ipynb     # Environment + baseline
│   ├── 02_RAG_System.ipynb               # RAG implementation
│   ├── 03_FineTuning_QLoRA.ipynb         # QLoRA training
│   └── 04_Comparison_Evaluation.ipynb    # Evaluation framework
├── data/
│   ├── f5_docs/             # RAG knowledge base
│   │   ├── bigip_basics.txt
│   │   ├── irules_guide.txt
│   │   ├── load_balancing.txt
│   │   ├── ssl_offloading.txt
│   │   ├── health_monitors.txt
│   │   └── troubleshooting.txt
│   └── training/
│       ├── f5_qa_train.jsonl    # 150+ training Q&A pairs
│       └── f5_qa_eval.jsonl     # 30+ evaluation pairs
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── data_loader.py       # Data loading utilities
│   ├── rag_utils.py         # RAG helper functions
│   ├── training_utils.py    # Training utilities
│   └── evaluation.py        # Evaluation metrics
└── docs/
    ├── INSTRUCTOR_GUIDE.md  # Teaching notes
    └── TROUBLESHOOTING.md   # Common issues and solutions
```

## Data Files

### F5 Documentation (RAG Knowledge Base)
The `data/f5_docs/` directory contains curated F5 technical documentation covering:
- BIG-IP architecture and components
- iRules scripting guide
- Load balancing algorithms and configuration
- SSL/TLS offloading
- Health monitors
- Troubleshooting procedures

### Training Data
- `f5_qa_train.jsonl`: 150+ question-answer pairs for fine-tuning
- `f5_qa_eval.jsonl`: 30+ pairs for evaluation

## Verification Checklist

### Pre-Lab
- [ ] Colab has T4 GPU enabled
- [ ] All dependencies install without errors
- [ ] `torch.cuda.is_available()` returns `True`

### Per Module
- [ ] **Module 1**: Model loads, baseline responses generated
- [ ] **Module 2**: Vector store created, RAG retrieves relevant chunks
- [ ] **Module 3**: Training loss decreases, model saves successfully
- [ ] **Module 4**: All results load, visualizations render

### End-to-End
- [ ] Same questions tested across all three approaches
- [ ] Fine-tuned + RAG shows clear improvement over baseline

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

## For Instructors

See [INSTRUCTOR_GUIDE.md](docs/INSTRUCTOR_GUIDE.md) for teaching notes, timing suggestions, and discussion points.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TinyLlama team for the base model
- Unsloth for the optimized training framework
- LangChain and ChromaDB communities
- F5 Networks for technical documentation inspiration
