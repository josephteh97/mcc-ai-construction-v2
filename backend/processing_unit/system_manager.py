import uuid
import os
import shutil
import subprocess
from enum import Enum
from typing import Dict, Any, List, Optional
import time
import fitz

# Import actual processing classes
from processing_unit.object_detection import ObjectDetector
from processing_unit.ocr_extraction import OCRExtractor
from generating_unit.ifc_generator import IfcGenerator

class SystemStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"
    PAUSED = "paused" # For intervention

class SystemManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.status = SystemStatus.IDLE
        self.logs = []
        self.current_job = {}
        self.last_job_context = {} # Store context for retry/resume
        self.config = {
            "scale": 0.05,
            "height": 3.0,
            "floor_count": 1,
            "conf_threshold": 0.25, # Added confidence threshold
            "generation_mode": "simple" # "simple" or "advanced"
        }
        
        model_path_env = os.environ.get("YOLO_MODEL_PATH")
        # Check root directory (one level up from backend) and current directory
        yolo26_paths = ["yolo26n.pt", "../yolo26n.pt", "backend/yolo26n.pt"]
        default_yolo = "yolo11n.pt"
        
        selected_path = None
        for p in yolo26_paths:
            if os.path.exists(p):
                selected_path = p
                break
        
        model_path = model_path_env if model_path_env else (selected_path if selected_path else default_yolo)
        self.detector = ObjectDetector(model_path=model_path)
        self.ocr = OCRExtractor()
        
        self.upload_dir = "uploads"
        self.output_dir = "outputs"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log("System Initialized. Models Loaded.")

    def log(self, message: str, level: str = "INFO"):
        entry = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
        self.logs.append(entry)
        print(entry)

    def update_config(self, key: str, value: Any):
        self.config[key] = value
        self.log(f"Config updated: {key} = {value}")

    async def resume_workflow(self):
        """
        Resumes the workflow using the last job context.
        """
        if not self.last_job_context or "file_path" not in self.last_job_context:
            self.log("No job context to resume.", "ERROR")
            return {"status": "error", "message": "No job context to resume."}
        
        self.log("Resuming workflow with updated configuration...")
        # Re-run detection and generation
        return await self._execute_processing(
            self.last_job_context["file_path"],
            self.last_job_context["job_id"]
        )

    async def process_workflow(self, file, scale=None, height=None, floor_count=None, generation_mode=None):
        self.status = SystemStatus.PROCESSING
        
        # Update config with provided parameters
        if scale is not None: self.update_config("scale", scale)
        if height is not None: self.update_config("height", height)
        if floor_count is not None: self.update_config("floor_count", floor_count)
        if generation_mode is not None: self.update_config("generation_mode", generation_mode)

        job_id = str(uuid.uuid4())
        self.current_job = {"id": job_id, "step": "init"}
        
        # Save File First
        self.current_job["step"] = "saving_file"
        file_path = os.path.join(self.upload_dir, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        self.log(f"File saved: {file_path}")

        # Store context for potential retry
        self.last_job_context = {
            "file_path": file_path,
            "job_id": job_id,
            "original_filename": file.filename
        }
        
        return await self._execute_processing(file_path, job_id)

    async def _execute_processing(self, file_path: str, job_id: str):
        """
        Internal method to execute the core processing logic.
        """
        try:
            input_paths = [file_path]
            if file_path.lower().endswith(".pdf"):
                input_paths = self._convert_pdf_to_images(file_path, job_id)
            total = 0
            merged = []
            self.current_job["step"] = "detection"
            self.log(f"Starting Object Detection with conf={self.config['conf_threshold']}...")
            for p in input_paths:
                r = self.detector.predict(p, conf_threshold=self.config['conf_threshold'])
                total += r["count"]
                merged.extend(r["detections"])
            det_results = {"count": total, "detections": merged}
            
            # Monitoring / Intervention Point
            if det_results['count'] == 0:
                self.status = SystemStatus.PAUSED
                self.log("Warning: No objects detected. Pausing for agent intervention.", "WARN")
                return {
                    "status": "paused", 
                    "reason": "no_objects_detected", 
                    "job_id": job_id,
                    "message": "No objects were detected. Please check the image or adjust parameters (e.g. threshold)."
                }

            self.log(f"Detected {det_results['count']} objects.")

            # Step 3: IFC Gen
            self.current_job["step"] = "ifc_generation"
            self.log(f"Generating IFC Model (Mode: {self.config.get('generation_mode', 'simple')})...")
            ifc_gen = IfcGenerator(project_name=f"Project_{job_id}")
            
            # Use current config
            scale = self.config["scale"]
            height = self.config["height"]
            floor_count = self.config["floor_count"]
            mode = self.config.get("generation_mode", "simple")

            if mode == "advanced":
                # Advanced Mode: GNN-based
                # 1. Clone/Source Model if needed
                self._ensure_gnn_model()
                # 2. Run GNN Inference (Mocked for now as we don't have the repo)
                graph_data = self._run_gnn_inference(file_path, det_results)
                # 3. Generate Structure
                ifc_gen.generate_advanced_structure(graph_data, scale, height, floor_count)
            else:
                # Simple Mode: Rule-based
                ifc_gen.generate_simple_extrusion(det_results, scale, height, floor_count)
            
            output_filename = f"{job_id}.ifc"
            output_path = os.path.join(self.output_dir, output_filename)
            ifc_gen.save(output_path)
            
            self.status = SystemStatus.COMPLETED
            self.log(f"Job completed. Output: {output_filename}")
            
            return {
                "status": "success",
                "file_id": job_id,
                "detections": det_results['count'],
                "ifc_url": f"/download/{output_filename}"
            }

        except Exception as e:
            self.status = SystemStatus.ERROR
            self.log(f"Error: {str(e)}", "ERROR")
            return {"status": "error", "message": str(e)}
        finally:
            if self.status != SystemStatus.PAUSED:
                self.status = SystemStatus.IDLE

    def _ensure_gnn_model(self):
        """
        Clones the GNN model if not present.
        Repo: Graphormer (Microsoft) - Adapted for Structural Connectivity
        """
        model_dir = os.environ.get("GNN_MODEL_PATH", "../models/Graphormer")
        if not os.path.exists(model_dir):
            self.log("Cloning Graphormer model...", "INFO")
            try:
                subprocess.run(["git", "clone", "https://github.com/microsoft/Graphormer.git", model_dir], check=True)
                self.log("Graphormer model cloned successfully.")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to clone Graphormer: {e}", "ERROR")
        else:
            self.log("Graphormer repository already exists. Skipping clone.", "INFO")
            
        if os.path.exists(model_dir):
             self.log("Graphormer model found/cloned.")
        else:
             self.log("Graphormer model NOT found. Please check git availability.", "WARN")

    def _convert_pdf_to_images(self, pdf_path: str, job_id: str) -> List[str]:
        paths = []
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img_path = os.path.join(self.upload_dir, f"{job_id}_page_{i+1}.png")
            pix.save(img_path)
            paths.append(img_path)
        doc.close()
        return paths
    def _run_gnn_inference(self, image_path, detections):
        """
        Runs the GNN model to predict connectivity.
        Returns a graph dict: {'nodes': [], 'edges': []}
        """
        self.log("Running GNN Inference for Structural Connectivity...")
        # Mocking the output of a GNN that connects columns with beams
        nodes = []
        for i, det in enumerate(detections['detections']):
             bbox = det['bbox']
             width = bbox[2] - bbox[0]
             depth = bbox[3] - bbox[1]
             cx = bbox[0] + width / 2
             cy = bbox[1] + depth / 2
             nodes.append({'id': i, 'x': cx, 'y': cy, 'width': width, 'depth': depth})
        
        # Mock Edges: Connect adjacent nodes
        edges = []
        for i in range(len(nodes) - 1):
            edges.append({'source': i, 'target': i+1})
            
        return {'nodes': nodes, 'edges': edges}
