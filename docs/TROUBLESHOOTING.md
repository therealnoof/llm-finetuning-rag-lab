# Troubleshooting Guide

This guide covers common issues encountered during the F5 AI Technical Assistant Training Lab.

## Table of Contents
- [Environment Issues](#environment-issues)
- [GPU and Memory Issues](#gpu-and-memory-issues)
- [Model Loading Issues](#model-loading-issues)
- [RAG System Issues](#rag-system-issues)
- [Training Issues](#training-issues)
- [Evaluation Issues](#evaluation-issues)

---

## Environment Issues

### Issue: "No module named 'X'"

**Symptoms:** Import errors when running cells.

**Solution:**
```python
# Restart runtime and run installation cells again
# In Colab: Runtime > Restart runtime

# Then run the installation cell:
!pip install -q transformers accelerate peft bitsandbytes trl
```

### Issue: Package version conflicts

**Symptoms:** Warnings about incompatible versions.

**Solution:**
```python
# Install specific versions
!pip install transformers==4.52.3 peft==0.18.0 trl==0.24.0

# Restart runtime after installation
```

### Issue: torchao C++ extensions warning (Local Setup)

**Symptoms:**
```
Skipping import of cpp extensions due to incompatible torch version 2.10.0+cu128 for torchao version 0.15.0
Please see https://github.com/pytorch/ao/issues/2919 for more info
```

**What this means:** This is a **warning, not an error**. The `torchao` package (a dependency of Unsloth) has C++ extensions that aren't compatible with your PyTorch version. Python fallbacks will be used instead.

**Can I ignore it?** **Yes.** The lab will run successfully. You may see slightly slower performance without the C++ optimizations, but functionality is not affected.

**Verify it's working:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

**Optional fix (if you want to eliminate the warning):**
```bash
# Option 1: Downgrade to a more compatible PyTorch + CUDA combination
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Option 2: Update torchao to latest version
pip install --upgrade torchao
```

**Note:** This commonly occurs with newer PyTorch versions (2.10+) or CUDA 12.8. The PyTorch 2.x + CUDA 12.1 combination has the best compatibility with current Unsloth/torchao releases.

### Issue: Colab session disconnected

**Symptoms:** Runtime disconnected, variables lost.

**Solution:**
1. Reconnect and restart runtime
2. Run all cells from the beginning
3. Consider saving checkpoints more frequently
4. Keep the browser tab active to prevent timeout

---

## GPU and Memory Issues

### Issue: CUDA out of memory (OOM)

**Symptoms:**
```
CUDA out of memory. Tried to allocate X GiB
RuntimeError: CUDA error: out of memory
```

**Solutions:**

1. **Clear GPU memory:**
```python
import torch
import gc

gc.collect()
torch.cuda.empty_cache()
```

2. **Reduce batch size:**
```python
# In training config
per_device_train_batch_size = 1  # Reduce from 2
gradient_accumulation_steps = 8  # Increase to compensate
```

3. **Use smaller max sequence length:**
```python
max_seq_length = 1024  # Reduce from 2048
```

4. **Restart runtime completely:**
   - Runtime > Disconnect and delete runtime
   - Reconnect with fresh GPU

### Issue: torch.cuda.is_available() returns False

**Symptoms:** No GPU detected.

**Solutions:**

1. **Check runtime type:**
   - Runtime > Change runtime type > Hardware accelerator: T4 GPU

2. **Verify GPU allocation:**
```python
!nvidia-smi
```

3. **GPU unavailable (Colab limit reached):**
   - Wait a few hours and try again
   - Use Colab Pro for guaranteed access
   - Continue with CPU (slower but works for small experiments)

### Issue: "RuntimeError: Expected all tensors to be on the same device"

**Solution:**
```python
# Ensure model and inputs are on same device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = tokenizer(text, return_tensors="pt").to(device)
```

---

## Model Loading Issues

### Issue: Model download fails or times out

**Symptoms:** Timeout or connection errors during model download.

**Solutions:**

1. **Retry the download:**
```python
# Sometimes HuggingFace Hub has temporary issues
# Simply re-run the cell
```

2. **Check internet connection:**
```python
!ping -c 3 huggingface.co
```

3. **Use alternative mirror (if available):**
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### Issue: "ValueError: Tokenizer class X does not exist"

**Solution:**
```python
# Use AutoTokenizer instead of specific class
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
```

### Issue: Unsloth installation fails

**Symptoms:** Errors installing unsloth package.

**Solution:**
```python
# Use the Colab-specific installation
!pip uninstall unsloth -y
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Restart runtime after installation
```

---

## RAG System Issues

### Issue: ChromaDB "sqlite3" version error

**Symptoms:**
```
RuntimeError: Your system has an unsupported version of sqlite3
```

**Solution:**
```python
# Install pysqlite3-binary and patch
!pip install pysqlite3-binary

# Add at the top of your notebook
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### Issue: "No documents found" when loading

**Symptoms:** Empty document list.

**Solutions:**

1. **Check file paths:**
```python
import os
print(os.listdir("data/f5_docs/"))  # Verify files exist
```

2. **Check file encoding:**
```python
# Ensure files are UTF-8 encoded
loader = TextLoader(file_path, encoding="utf-8")
```

### Issue: Poor retrieval results

**Symptoms:** Retrieved documents don't match query well.

**Solutions:**

1. **Adjust chunk size:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Try smaller chunks
    chunk_overlap=50
)
```

2. **Increase k (number of retrieved docs):**
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

3. **Try different embedding model:**
```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"  # Alternative model
)
```

### Issue: Embedding model download slow

**Solution:**
```python
# Pre-download the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

---

## Training Issues

### Issue: Training loss not decreasing

**Symptoms:** Loss stays flat or increases.

**Solutions:**

1. **Check data format:**
```python
# Verify training data is correctly formatted
import jsonlines
with jsonlines.open("data/training/f5_qa_train.jsonl") as reader:
    sample = next(reader)
    print(sample)  # Should have 'question' and 'answer' keys
```

2. **Adjust learning rate:**
```python
learning_rate = 1e-4  # Try lower rate
# or
learning_rate = 5e-4  # Try higher rate
```

3. **Increase training epochs:**
```python
num_train_epochs = 5  # More epochs may help
```

### Issue: Training extremely slow

**Solutions:**

1. **Verify GPU is being used:**
```python
print(f"Using device: {model.device}")
!nvidia-smi  # Check GPU utilization
```

2. **Enable gradient checkpointing:**
```python
# Already enabled by default in Unsloth
# If using standard PEFT:
model.gradient_checkpointing_enable()
```

3. **Reduce logging frequency:**
```python
logging_steps = 50  # Less frequent logging
```

### Issue: "ValueError: Target modules not found"

**Symptoms:** LoRA can't find layers to adapt.

**Solution:**
```python
# Print available modules
print(model)

# Or for Unsloth, use default target modules
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
```

### Issue: Checkpoint saving fails

**Symptoms:** Error when saving model.

**Solutions:**

1. **Check disk space:**
```python
!df -h
```

2. **Save to different location:**
```python
trainer.save_model("/content/drive/MyDrive/checkpoints")
```

3. **Save only LoRA adapter:**
```python
model.save_pretrained("./lora_adapter")  # Smaller than full model
```

---

## Evaluation Issues

### Issue: Results don't load

**Symptoms:** FileNotFoundError when loading results.

**Solution:**
```python
# Verify file exists
import os
print(os.listdir("results/"))

# Check for correct path
results_path = "results/evaluation_results.json"
```

### Issue: Visualization doesn't render

**Symptoms:** Empty or broken charts.

**Solutions:**

1. **Enable inline plotting:**
```python
%matplotlib inline
```

2. **Check data exists:**
```python
print(evaluator.get_summary_stats())  # Verify data
```

3. **Save to file instead:**
```python
fig.savefig("comparison_chart.png")
```

### Issue: JSON parsing errors

**Solution:**
```python
import json

# Ensure proper JSON format
with open("results/results.json", "r") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON error at position {e.pos}: {e.msg}")
```

---

## General Tips

### Restart Runtime Properly
```
Runtime > Restart runtime
# Then re-run all cells from the beginning
```

### Check Package Versions
```python
import transformers, peft, trl, torch
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"trl: {trl.__version__}")
print(f"torch: {torch.__version__}")
```

### Monitor Resources
```python
# GPU memory
!nvidia-smi

# System memory
!free -h

# Disk space
!df -h
```

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Getting Help

If issues persist:

1. **Check the error message carefully** - Often contains the solution
2. **Search the error online** - Stack Overflow, GitHub Issues
3. **Consult library documentation:**
   - [Transformers](https://huggingface.co/docs/transformers)
   - [PEFT](https://huggingface.co/docs/peft)
   - [Unsloth](https://github.com/unslothai/unsloth)
   - [LangChain](https://python.langchain.com/)
4. **File an issue** on the lab repository with:
   - Error message
   - Code that caused the error
   - Environment details (Colab, package versions)
