# AWS Environment Setup Guide

Complete installation guide for running the F5 AI Technical Assistant Training Lab on an AWS g4dn.4xlarge instance with Ubuntu Desktop and XRDP.

---

## Table of Contents
- [Instance Specifications](#instance-specifications)
- [Step 1: Launch EC2 Instance](#step-1-launch-ec2-instance)
- [Step 2: Initial System Setup](#step-2-initial-system-setup)
- [Step 3: Install NVIDIA Drivers](#step-3-install-nvidia-drivers)
- [Step 4: Install Ubuntu Desktop](#step-4-install-ubuntu-desktop)
- [Step 5: Install XRDP](#step-5-install-xrdp)
- [Step 6: Install Python and Dependencies](#step-6-install-python-and-dependencies)
- [Step 7: Clone and Setup the Lab](#step-7-clone-and-setup-the-lab)
- [Step 8: Connect via Remote Desktop](#step-8-connect-via-remote-desktop)
- [Step 9: Running the Lab](#step-9-running-the-lab)
- [Troubleshooting](#troubleshooting)

---

## Instance Specifications

| Resource | Specification |
|----------|---------------|
| Instance Type | g4dn.4xlarge |
| GPU | 1x NVIDIA T4 (16GB VRAM) |
| vCPUs | 16 |
| RAM | 64GB |
| Storage | 100GB+ EBS (gp3 recommended) |
| OS | Ubuntu 22.04 LTS |

**Estimated Cost:** ~$1.20/hour (on-demand)

---

## Step 1: Launch EC2 Instance

### 1.1 Choose AMI
- Go to EC2 Console → Launch Instance
- Select: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
- Architecture: **64-bit (x86)**

### 1.2 Choose Instance Type
- Select: **g4dn.4xlarge**

### 1.3 Configure Storage
- Root volume: **100 GB** (minimum)
- Volume type: **gp3** (better performance)

### 1.4 Configure Security Group
Create or select a security group with these inbound rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| RDP | 3389 | Your IP | Remote Desktop |
| Custom TCP | 8888 | Your IP | Jupyter (optional) |

### 1.5 Key Pair
- Create or select an existing key pair
- Download the .pem file (you'll need this to connect)

### 1.6 Launch
- Click **Launch Instance**
- Wait for instance to reach **Running** state

---

## Step 2: Initial System Setup

### 2.1 Connect via SSH

```bash
# Make key file secure
chmod 400 your-key.pem

# Connect to instance
ssh -i your-key.pem ubuntu@<your-instance-public-ip>
```

### 2.2 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.3 Set Timezone (Optional)

```bash
sudo timedatectl set-timezone America/New_York
```

---

## Step 3: Install NVIDIA Drivers

### 3.1 Install NVIDIA Driver

```bash
# Install required packages
sudo apt install -y linux-headers-$(uname -r) build-essential

# Add NVIDIA driver repository
sudo apt install -y nvidia-driver-535

# Reboot to load the driver
sudo reboot
```

### 3.2 Verify NVIDIA Driver (after reboot)

```bash
# Reconnect via SSH after reboot
ssh -i your-key.pem ubuntu@<your-instance-public-ip>

# Verify driver installation
nvidia-smi
```

Expected output should show:
- NVIDIA T4 GPU
- Driver Version: 535.x
- CUDA Version: 12.x

### 3.3 Install CUDA Toolkit

```bash
# Install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA
nvcc --version
```

---

## Step 4: Install Ubuntu Desktop

### 4.1 Install Desktop Environment

```bash
# Install Ubuntu Desktop (minimal version for faster install)
sudo apt install -y ubuntu-desktop-minimal

# Or full desktop (more applications, larger download)
# sudo apt install -y ubuntu-desktop
```

This takes 10-15 minutes to download and install.

### 4.2 Set Default Target to Graphical

```bash
sudo systemctl set-default graphical.target
```

---

## Step 5: Install XRDP

### 5.1 Install XRDP Server

```bash
# Install XRDP
sudo apt install -y xrdp

# Add xrdp user to ssl-cert group
sudo adduser xrdp ssl-cert

# Enable and start XRDP service
sudo systemctl enable xrdp
sudo systemctl start xrdp
```

### 5.2 Configure XRDP for Ubuntu Desktop

```bash
# Configure XRDP to use the desktop session
echo "gnome-session" > ~/.xsession
chmod +x ~/.xsession

# Fix potential black screen issues
sudo sed -i 's/^#\?allowed_users=.*/allowed_users=anybody/' /etc/X11/Xwrapper.config 2>/dev/null || true
```

### 5.3 Set Password for Ubuntu User

```bash
# Set a password (required for RDP login)
sudo passwd ubuntu
```
Enter a strong password when prompted.

### 5.4 Configure Firewall (if enabled)

```bash
# Allow RDP through firewall
sudo ufw allow 3389/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

### 5.5 Reboot

```bash
sudo reboot
```

---

## Step 6: Install Python and Dependencies

### 6.1 Reconnect via SSH

```bash
ssh -i your-key.pem ubuntu@<your-instance-public-ip>
```

### 6.2 Install Python 3.10+ and pip

```bash
# Python should already be installed, verify:
python3 --version

# Install pip and venv
sudo apt install -y python3-pip python3-venv python3-dev
```

### 6.3 Install System Dependencies

```bash
# Install git and other utilities
sudo apt install -y git curl wget

# Install dependencies for some Python packages
sudo apt install -y libsqlite3-dev libffi-dev
```

---

## Step 7: Clone and Setup the Lab

### 7.1 Clone the Repository

```bash
cd ~
git clone https://github.com/therealnoof/llm-finetuning-rag-lab.git
cd llm-finetuning-rag-lab
```

### 7.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 7.3 Install PyTorch with CUDA Support

```bash
# For CUDA 12.1+ (check with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 7.4 Install All Dependencies

```bash
pip install -r requirements.txt
```

### 7.5 Install Unsloth

```bash
# First, ensure setuptools is up to date (prevents build errors)
pip install --upgrade setuptools wheel

# For CUDA 12.1+
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# If the above fails, try:
# pip install unsloth
```

### 7.6 Install Jupyter

```bash
pip install jupyterlab notebook
```

### 7.7 Verify Installation

```bash
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

### 7.8 Register Jupyter Kernel

```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=llm-lab --display-name="LLM Training Lab"
```

---

## Step 8: Connect via Remote Desktop

### 8.1 Get Instance Public IP
- Go to EC2 Console
- Select your instance
- Copy the **Public IPv4 address**

### 8.2 Connect from Windows
1. Open **Remote Desktop Connection** (mstsc.exe)
2. Enter the instance public IP
3. Click **Connect**
4. Login with:
   - Username: `ubuntu`
   - Password: (the password you set in Step 5.3)

### 8.3 Connect from macOS
1. Install **Microsoft Remote Desktop** from App Store
2. Add a new PC with the instance public IP
3. Connect and login with ubuntu credentials

### 8.4 Connect from Linux
```bash
# Install Remmina or another RDP client
sudo apt install remmina remmina-plugin-rdp

# Connect using Remmina GUI or:
remmina -c rdp://ubuntu@<your-instance-ip>
```

---

## Step 9: Running the Lab

### 9.1 Open Terminal in Ubuntu Desktop
- Click Activities (top left)
- Search for "Terminal"
- Open Terminal application

### 9.2 Navigate and Activate Environment

```bash
cd ~/llm-finetuning-rag-lab
source venv/bin/activate
```

### 9.3 Launch Jupyter Lab

```bash
jupyter lab
```

This will open Jupyter Lab in Firefox within the desktop session.

### 9.4 Open the AWS Notebooks
1. In Jupyter Lab, navigate to `notebooks/aws/`
2. Open notebooks in order:
   - `01_Setup_and_Base_Model.ipynb`
   - `02_RAG_System.ipynb`
   - `03_FineTuning_QLoRA.ipynb`
   - `04_Comparison_Evaluation.ipynb`

### 9.5 Select the Correct Kernel
- When opening a notebook, select kernel: **LLM Training Lab**
- Or go to Kernel → Change Kernel → LLM Training Lab

---

## Quick Reference Commands

```bash
# Activate environment
cd ~/llm-finetuning-rag-lab && source venv/bin/activate

# Start Jupyter Lab
jupyter lab

# Check GPU status
nvidia-smi

# Check GPU memory
watch -n 1 nvidia-smi

# Clear GPU memory (if needed)
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Troubleshooting

### XRDP Black Screen

```bash
# Reconnect via SSH and run:
echo "gnome-session" > ~/.xsession
sudo systemctl restart xrdp
```

### XRDP Connection Refused

```bash
# Check XRDP status
sudo systemctl status xrdp

# Restart XRDP
sudo systemctl restart xrdp

# Check if port 3389 is listening
sudo netstat -tlnp | grep 3389
```

### NVIDIA Driver Not Found

```bash
# Reinstall NVIDIA driver
sudo apt install --reinstall nvidia-driver-535
sudo reboot
```

### CUDA Out of Memory

```python
# In Python/Jupyter:
import torch
torch.cuda.empty_cache()

# Or restart the kernel
```

### Jupyter Kernel Not Found

```bash
# Re-register the kernel
source ~/llm-finetuning-rag-lab/venv/bin/activate
python -m ipykernel install --user --name=llm-lab --display-name="LLM Training Lab"
```

### Unsloth Import Errors

```bash
# Reinstall Unsloth
pip uninstall unsloth -y
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

### bitsandbytes Errors

```bash
# Reinstall bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes --no-cache-dir
```

---

## Stopping the Instance

When you're done with the lab, **stop the instance** to avoid charges:

1. Go to EC2 Console
2. Select your instance
3. Instance State → Stop Instance

**Note:** Stopped instances still incur EBS storage charges (~$0.10/GB/month).

To completely remove all charges, terminate the instance (this deletes all data).

---

## Cost Optimization Tips

| Action | Savings |
|--------|---------|
| Stop instance when not in use | ~$1.20/hour saved |
| Use Spot instances | Up to 70% off (but may be interrupted) |
| Reduce EBS volume size | ~$0.10/GB/month |
| Use gp3 instead of gp2 | Better performance at same cost |

---

## Complete Installation Script

For convenience, here's a script that automates most of the installation:

```bash
#!/bin/bash
# Save as setup.sh and run with: sudo bash setup.sh

set -e

echo "=== Updating system ==="
apt update && apt upgrade -y

echo "=== Installing NVIDIA drivers ==="
apt install -y linux-headers-$(uname -r) build-essential
apt install -y nvidia-driver-535

echo "=== Installing Ubuntu Desktop ==="
apt install -y ubuntu-desktop-minimal
systemctl set-default graphical.target

echo "=== Installing XRDP ==="
apt install -y xrdp
adduser xrdp ssl-cert
systemctl enable xrdp
systemctl start xrdp

echo "=== Installing Python dependencies ==="
apt install -y python3-pip python3-venv python3-dev git curl wget

echo "=== Setup complete! ==="
echo "Please run: sudo passwd ubuntu"
echo "Then reboot: sudo reboot"
```

After reboot, run as the ubuntu user:

```bash
#!/bin/bash
# Save as setup_lab.sh and run with: bash setup_lab.sh

set -e

cd ~
git clone https://github.com/therealnoof/llm-finetuning-rag-lab.git
cd llm-finetuning-rag-lab

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" || pip install unsloth
pip install jupyterlab notebook
python -m ipykernel install --user --name=llm-lab --display-name="LLM Training Lab"

echo "=== Lab setup complete! ==="
echo "Run: cd ~/llm-finetuning-rag-lab && source venv/bin/activate && jupyter lab"
```
