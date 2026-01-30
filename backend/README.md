# MCC AI Concrete Structure Backend

## Setup & Installation

To ensure reproducibility and handle binary dependencies correctly, we use a hybrid installation approach (Conda + Pip).

### 1. Create Conda Environment
First, create a fresh Conda environment with Python 3.10:

```bash
conda create -n mcc-backend python=3.10.19
conda activate mcc-backend
```

### 2. Install Conda Dependencies
Some libraries (like `ifcopenshell`) rely on heavy system dependencies (C++, geometry kernels) and are best installed via Conda to avoid compilation issues.

```bash
# Install IfcOpenShell (v0.8.4+)
conda install -c conda-forge ifcopenshell
```

### 3. Install Pip Dependencies
The rest of the Python ecosystem libraries are managed via `pip`.

```bash
pip install -r backend/requirements.txt

conda env create -f environment.yml
```

> **Note:** `requirements.txt` contains specific versions for `fastapi`, `ultralytics`, `paddlepaddle`, etc. Ensure you install `ifcopenshell` via Conda *before* running pip install to prevent dependency conflicts.

## Running the Application

```bash
# Start the FastAPI server
python main.py
```

##### For Agentic Programming (Vision-Language Model)
```bash
# Install Git LFS first if you haven't
git lfs install

# Create a 'models' directory at the project root
mkdir -p ../models
cd ../models

# Clone Qwen2.5-VL (Latest Available Version)
# Choose the size that fits your hardware:

# Option 1: 3B Model (Recommended for Consumer GPUs - 6GB+ VRAM)
git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

# Option 2: 7B Model (Recommended for Better Reasoning - 16GB+ VRAM)
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

# The backend code is configured to look for the 3B model by default at:
# ../models/Qwen2.5-VL-3B-Instruct
# You can override this by setting the QWEN_MODEL_PATH environment variable.
```

##### Advanced Reconstruction (GNN: Graphormer)
```bash
# Clone Graphormer (Microsoft) for GNN-based structural connectivity
cd ../models
git clone https://github.com/microsoft/Graphormer.git

# The backend checks ../models/Graphormer automatically in advanced mode.
# If not present, SystemManager will attempt to clone it at runtime.
```

##### Environment Variables (Optional)
```bash
# Override Qwen model location if you used a custom path
export QWEN_MODEL_PATH="$(pwd)/models/Qwen2.5-VL-3B-Instruct"
# On Linux/macOS:
# export QWEN_MODEL_PATH=../models/Qwen2.5-VL-3B-Instruct

# Override Graphormer model path (Advanced mode)
export GNN_MODEL_PATH="$(pwd)/models/Graphormer"
# On Linux/macOS:
# export GNN_MODEL_PATH=../models/Graphormer
```

Persist across sessions
```bash
# ensure git lfs is installed
sudo apt update && sudo apt install -y git-lfs && git lfs install

echo 'export QWEN_MODEL_PATH="$HOME/Documents/mcc-ai-concrete-structure-of-construction/models/Qwen2.5-VL-3B-Instruct"' >> ~/.bashrc
echo 'export GNN_MODEL_PATH="$HOME/Documents/mcc-ai-concrete-structure-of-construction/models/Graphormer"' >> ~/.bashrc
source ~/.bashrc

# quick check
echo "$QWEN_MODEL_PATH"
echo "$GNN_MODEL_PATH"
```
