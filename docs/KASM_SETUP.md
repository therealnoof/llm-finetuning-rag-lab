# KASM Workspaces Setup Guide

Browser-based desktop access for the F5 AI Technical Assistant Training Lab. Use this as an alternative to XRDP for students who cannot install an RDP client.

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Install KASM Workspaces](#step-1-install-kasm-workspaces)
- [Step 2: Open Port 3410](#step-2-open-port-3410)
- [Step 3: Verify Access](#step-3-verify-access)
- [Step 4: Add a Desktop Workspace](#step-4-add-a-desktop-workspace)
- [Step 5: Create Student Accounts](#step-5-create-student-accounts)
- [Step 6: Student Connection Instructions](#step-6-student-connection-instructions)
- [Step 7: Running the Lab via KASM](#step-7-running-the-lab-via-kasm)
- [Administration Reference](#administration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

KASM Workspaces provides a full Linux desktop accessible from any modern web browser over HTTPS. Students do not need to install any software — they simply navigate to a URL and log in.

| Feature | Detail |
|---------|--------|
| Access Method | HTTPS via browser (Chrome, Firefox, Edge, Safari) |
| Port | **3410** |
| Client Software Required | None |
| Protocol | HTTPS (TLS encrypted) |

---

## Prerequisites

- Ubuntu 22.04 LTS server (private hypervisor environment)
- SSH access to the server
- Port **3410** exposed from the hypervisor to the student network
- Minimum 2 CPU cores and 4GB RAM available for KASM services (on top of lab workload)

---

## Step 1: Install KASM Workspaces

### 1.1 SSH into the Server

```bash
ssh <username>@<server-ip>
```

### 1.2 Install Dependencies

```bash
sudo apt update && sudo apt install -y curl tar
```

### 1.3 Download and Install KASM

```bash
cd /tmp
curl -O https://kasm-static-content.s3.amazonaws.com/kasm_release_1.16.1.98d6fa.tar.gz
tar -xf kasm_release_1.16.1.98d6fa.tar.gz
sudo bash kasm_release/install.sh -L 3410
```

The `-L 3410` flag configures KASM to listen on port **3410** instead of the default 443.

### 1.4 Save Your Credentials

The installer will print credentials to the terminal upon completion:

```
Admin Account
  username: admin@kasm.local
  password: <SAVE THIS>

User Account
  username: user@kasm.local
  password: <SAVE THIS>
```

**Save both passwords immediately.** You will need the admin password to configure workspaces and create student accounts.

> The installer handles Docker, the internal database, and all KASM services automatically. Installation takes approximately 5-10 minutes.

---

## Step 2: Open Port 3410

### 2.1 OS Firewall (UFW)

If UFW is enabled on the server:

```bash
sudo ufw allow 3410/tcp
sudo ufw reload
```

### 2.2 OS Firewall (iptables)

If using iptables directly:

```bash
sudo iptables -A INPUT -p tcp --dport 3410 -j ACCEPT
```

### 2.3 Hypervisor / Network

Ensure port **3410** is reachable from the student network. This depends on your hypervisor configuration:

| Network Mode | Action Required |
|--------------|-----------------|
| **Bridged** | VM gets a LAN IP directly — no port forwarding needed |
| **NAT** | Add a port forward rule on the hypervisor: `host:3410 → guest:3410` |

### 2.4 Verify the Port is Listening

```bash
sudo ss -tlnp | grep 3410
```

You should see KASM's proxy process listening on `0.0.0.0:3410`.

---

## Step 3: Verify Access

From any machine on the student network, open a browser and navigate to:

```
https://<server-ip>:3410
```

- You will see a **certificate warning** (self-signed cert) — click through to proceed
- Log in as **admin@kasm.local** with the admin password from Step 1.4

---

## Step 4: Add a Desktop Workspace

1. Log into KASM as **admin** at `https://<server-ip>:3410`
2. Navigate to **Workspaces** → **Workspaces**
3. Click **Add Workspace**
4. From the registry, install **Ubuntu Jammy Desktop** (matches the host OS)
5. Adjust resource limits per session as needed:
   - **CPU Cores** — e.g., 4 per student
   - **Memory** — e.g., 8GB per student

### Optional: GPU Passthrough

If students need direct GPU access from within their KASM desktop session, edit the workspace and set **Docker Run Config Override**:

```json
{
  "devices": [
    "/dev/nvidia0:/dev/nvidia0",
    "/dev/nvidiactl:/dev/nvidiactl",
    "/dev/nvidia-uvm:/dev/nvidia-uvm"
  ],
  "runtime": "nvidia"
}
```

> **Note:** GPU passthrough shares the single T4 GPU across sessions. For this lab, it is recommended to run Jupyter on the host and have students access it through the KASM browser instead of passing the GPU into each container.

---

## Step 5: Create Student Accounts

1. In the admin panel, go to **Access Management** → **Users**
2. Click **Add User** for each student
3. Set a username (e.g., `student01@kasm.local`) and password
4. Assign them to the **default** group (or create a custom group)

---

## Step 6: Student Connection Instructions

Provide the following to your students:

### Connecting to the Lab

1. Open any modern browser (Chrome, Firefox, Edge, or Safari)
2. Navigate to: `https://<server-ip>:3410`
3. You will see a certificate warning — click **Advanced** → **Proceed** (this is expected)
4. Log in with the credentials provided by your instructor
5. Click the **Ubuntu Desktop** workspace to launch a desktop session
6. Your desktop will load in the browser — no software installation required

---

## Step 7: Running the Lab via KASM

Once inside the KASM desktop session:

### 7.1 Open Terminal

- Click **Activities** (top left) → search for **Terminal**

### 7.2 Navigate and Activate Environment

```bash
cd ~/llm-finetuning-rag-lab
source venv/bin/activate
```

### 7.3 Launch Jupyter Lab

```bash
jupyter lab
```

This will open Jupyter Lab in the browser within the KASM desktop session.

### 7.4 Open the Notebooks

1. In Jupyter Lab, navigate to `notebooks/aws/`
2. Open notebooks in order:
   - `01_Setup_and_Base_Model.ipynb`
   - `02_RAG_System.ipynb`
   - `03_FineTuning_QLoRA.ipynb`
   - `04_Comparison_Evaluation.ipynb`

### 7.5 Select the Correct Kernel

- Select kernel: **LLM Training Lab**
- Or go to Kernel → Change Kernel → LLM Training Lab

---

## Administration Reference

| Item | Detail |
|------|--------|
| Web UI | `https://<server-ip>:3410` |
| Admin login | `admin@kasm.local` |
| Config path | `/opt/kasm/` |
| Logs | `/opt/kasm/current/log/` |
| Start services | `sudo /opt/kasm/bin/start` |
| Stop services | `sudo /opt/kasm/bin/stop` |
| Restart services | `sudo /opt/kasm/bin/stop && sudo /opt/kasm/bin/start` |
| Check containers | `sudo docker ps` |
| Reset admin password | `sudo /opt/kasm/bin/utils/db_init -q -s` |

---

## Troubleshooting

### Cannot Reach KASM in Browser

```bash
# Verify KASM containers are running
sudo docker ps

# Verify port 3410 is listening
sudo ss -tlnp | grep 3410

# Check if firewall is blocking the port
sudo ufw status

# Restart KASM services
sudo /opt/kasm/bin/stop
sudo /opt/kasm/bin/start
```

### Certificate Warning in Browser

This is expected with the default self-signed certificate. Students should click through the warning. To use a trusted certificate:

1. Place your cert and key in `/opt/kasm/current/certs/`
2. Restart KASM services

### Desktop Session Stuck or Frozen

1. Log into the admin panel at `https://<server-ip>:3410`
2. Go to **Sessions** → find the stuck session
3. Click **Delete** to terminate it
4. The student can log in again and start a new session

### KASM Services Won't Start

```bash
# Check Docker is running
sudo systemctl status docker

# If Docker is not running, start it
sudo systemctl start docker

# Then start KASM
sudo /opt/kasm/bin/start
```

### Out of Disk Space

KASM container images can be large. Check available space:

```bash
df -h /

# Clean up unused Docker images
sudo docker image prune -a
```

### View KASM Logs

```bash
# All logs
ls /opt/kasm/current/log/

# Proxy/web access logs
sudo tail -f /opt/kasm/current/log/nginx/access.log

# KASM manager logs
sudo tail -f /opt/kasm/current/log/manager.log
```
