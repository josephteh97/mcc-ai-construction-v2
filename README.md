# MCC AI Concrete Structure System

An AI-powered system to transform 2D engineering PDF drawings into 3D IFC models for the construction industry.

## Features

- **AI Processing Unit**:
  - **YOLOv11**: Custom trained model to detect Columns, Beams, Slabs, and Grid Lines.
  - **PaddleOCR**: Extract text dimensions and labels.
  - **Vision Model**: Integration for Qwen3-VL / Claude (Conceptual).
- **3D Generation Unit**:
  - Generates industry-standard **IFC** files (International Foundation Class).
  - Uses `IfcOpenShell` for precise BIM data creation.
- **Frontend**:
  - React + Vite + Tailwind CSS.
  - Interactive 3D Viewer using `Three.js` and `web-ifc`.

## Prerequisites

- **OS**: Windows (Development) or Ubuntu Linux (Production).
- **Python**: 3.10+
- **Node.js**: 18+
- **Conda** (Recommended for Python environment management).

## Installation

### 1. Backend Setup

```bash
cd backend
# Using Conda (Recommended)
conda env create -f environment.yml
conda activate mcc-ai-construction

# OR Using Pip
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

## Pre-training (YOLOv11)

Before running the system, you must train the YOLO model to recognize structural elements.

1. Prepare your dataset (images and labels) in YOLO format.
2. Configure `dataset.yaml`.
3. download dataset

```bash
python backend/training/download_data.py [YOUR_roboflow_API_KEY]
```
4. Run the training script:

```bash
cd backend/training
python train_yolo.py [Path_to_dataset]
# python train_yolo.py /columns-and-ducts-detection-1
```

4. The best model will be saved. Update `backend/processing_unit/object_detection.py` to point to your new `.pt` file.

## Usage

### Windows
Run the start script:
```cmd
scripts\run.bat
```

### Linux
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs

## Architecture

- `backend/`: FastAPI application.
  - `processing_unit/`: AI models (YOLO, OCR).
  - `generating_unit/`: IFC generation logic.
  - `training/`: Model training scripts.
- `frontend/`: React application.
- `infrastructure/`: Nginx and systemd configs.

#### Environment Setup and run
To set up on Linux
```
# Option A: Using Conda (Recommended)
conda env create -f backend/environment.yml
conda activate mcc-ai-construction

# Option B: Using Pip (if you prefer standard venv)
pip install -r backend/requirements.txt
```

#### Roboflow Dataset & Training Pipeline
step 1 download the Roboflow dataset
```
cd backend/training
python download_data.py [YOUR_API_KEY]
```
step 2 Train, Validate and Test 
Use the updated train_yolo.py, which now includes a full pipeline
- Train : Fine-tunes yolo11n.pt on your data.
- Validate : Calculates mAP (Mean Average Precision) on the validation set.
- Test : Runs inference on the test set and saves visualized results.
- Export : Saves the model as ONNX for potential deployment.
```
python train_yolo.py
```
step 3 Test the frontend (user interface)
vite installation
```bash
npm install
```
start development server
```bash
npm run dev
```
real deployment (for actual deployment only)
```bash
npm run build
```

## License
MCC Engineering
