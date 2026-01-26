# AWS Environment Setup Guide

This guide covers setting up the F5 AI Technical Assistant Training Lab on an AWS g4dn.4xlarge instance.

---

## Instance Specifications

| Resource | Specification |
|----------|---------------|
| Instance Type | g4dn.4xlarge |
| GPU | 1x NVIDIA T4 (16GB VRAM) |
| vCPUs | 16 |
| RAM | 64GB |
| Storage | Recommended 100GB+ EBS |
| OS | Ubuntu 22.04 LTS |

---

## Prerequisites

Before running the lab, ensure your instance has:

- [x] Ubuntu Desktop installed
- [x] XRDP configured for remote access
- [x] NVIDIA drivers installed
- [x] CUDA toolkit installed
- [x] Python 3.10+ installed
- [x] Jupyter Notebook/Lab installed

---

## Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output should show:
# - NVIDIA T4 GPU
# - Driver Version: 535.x or higher
# - CUDA Version: 12.x

# Check CUDA
nvcc --version

# Check Python
python3 --version
```

---

## Installation Steps

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/therealnoof/llm-finetuning-rag-lab.git
cd llm-finetuning-rag-lab
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install PyTorch with CUDA Support

First, install PyTorch matching your CUDA version:

```bash
# For CUDA 12.1+ (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Unsloth (CUDA-specific)

```bash
# For CUDA 12.1+
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

### 6. Verify Installation

```python
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 12.1
GPU: Tesla T4
```

---

## Running Jupyter Notebooks

### Option 1: JupyterLab (Recommended)

```bash
# Activate virtual environment
source ~/llm-finetuning-rag-lab/venv/bin/activate

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Access via browser at: http://<instance-ip>:8888
```

### Option 2: Jupyter Notebook Classic

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### Option 3: Via Ubuntu Desktop (XRDP)

1. Connect via RDP to your instance
2. Open terminal
3. Navigate to project and launch Jupyter:
   ```bash
   cd ~/llm-finetuning-rag-lab
   source venv/bin/activate
   jupyter lab
   ```
4. Browser opens automatically

---

## Running the Lab Modules

### Module 1: Setup and Base Model
```bash
# Open notebooks/01_Setup_and_Base_Model.ipynb
# Skip the Colab-specific installation cells (they start with !pip install)
# The dependencies are already installed via requirements.txt
```

### Module 2: RAG System
```bash
# Open notebooks/02_RAG_System.ipynb
# Skip installation and git clone cells
# Start from the imports
```

### Module 3: Fine-Tuning with QLoRA
```bash
# Open notebooks/03_FineTuning_QLoRA.ipynb
# Skip installation cells
# Training will be faster and more stable than Colab
```

### Module 4: Comparison and Evaluation
```bash
# Open notebooks/04_Comparison_Evaluation.ipynb
# All visualizations will render in JupyterLab
```

---

## Key Differences from Google Colab

| Aspect | Google Colab | AWS g4dn.4xlarge |
|--------|--------------|------------------|
| Session time | ~12 hours max, may disconnect | Unlimited (you control) |
| GPU | T4 (shared) | T4 (dedicated) |
| RAM | ~12GB | 64GB |
| Storage | Temporary | Persistent EBS |
| Dependencies | Install each session | Install once |
| Cost | Free tier available | ~$1.20/hour |

### Notebook Modifications for AWS

The notebooks contain Colab-specific cells that you can skip:

1. **Skip cells starting with:**
   - `!pip install`
   - `!git clone`
   - `%cd llm-finetuning-rag-lab`

2. **Path adjustments:**
   - Colab paths: `/content/llm-finetuning-rag-lab/`
   - AWS paths: `~/llm-finetuning-rag-lab/` or relative paths

3. **No runtime restart needed:**
   - Colab sometimes requires restart after installing packages
   - AWS environment is stable after initial setup

---

## Troubleshooting

### CUDA Out of Memory

```python
# Clear GPU memory between runs
import torch
torch.cuda.empty_cache()

# Or restart the kernel
```

### bitsandbytes Issues

```bash
# Reinstall with CUDA support
pip uninstall bitsandbytes -y
pip install bitsandbytes --no-cache-dir
```

### Unsloth Import Errors

```bash
# Ensure correct CUDA version
pip uninstall unsloth -y
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

### Jupyter Kernel Not Found

```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=llm-lab --display-name="LLM Training Lab"
```

Then select "LLM Training Lab" kernel in Jupyter.

---

## Recommended Workflow

1. **Start instance** via AWS Console
2. **Connect via RDP** to Ubuntu Desktop
3. **Open terminal** and activate environment:
   ```bash
   cd ~/llm-finetuning-rag-lab
   source venv/bin/activate
   jupyter lab
   ```
4. **Run notebooks** in order (Module 1 → 2 → 3 → 4)
5. **Stop instance** when done to save costs

---

## Security Notes

- Keep security group restricted (don't expose Jupyter to 0.0.0.0/0)
- Use SSH tunneling or XRDP for secure access
- Don't commit AWS credentials to the repository
- Consider using IAM roles instead of access keys

---

## Cost Optimization

| Action | Savings |
|--------|---------|
| Stop instance when not in use | ~$1.20/hour |
| Use spot instances for training | Up to 70% off |
| Reduce EBS volume size | ~$0.10/GB/month |
| Use smaller instance for non-GPU work | Switch to t3.medium |

---

## Additional Resources

- [AWS g4dn Instance Documentation](https://aws.amazon.com/ec2/instance-types/g4/)
- [NVIDIA T4 Specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
