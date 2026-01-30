# This is a placeholder for the Vision Model integration (Qwen3-VL / Claude Sonnet 4.5)
# Actual implementation requires model weights or API keys.

import requests
import base64
import os
import torch
import json
import re

try:
    # Try importing the specific class first (newer transformers)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_QWEN_CLASS = True
except ImportError:
    # Fallback to generic AutoModel (older transformers)
    from transformers import AutoModelForCausalLM, AutoProcessor
    HAS_QWEN_CLASS = False

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False

HAS_QWEN = (HAS_QWEN_CLASS or True) and HAS_QWEN_UTILS # AutoModel is always available in modern transformers

from processing_unit.system_manager import SystemManager, SystemStatus

class VisionReasoner:
    def __init__(self, system=None, model_type: str = "qwen-vl"):
        self.model_type = model_type
        # Allow injecting a system instance (useful for testing or shared state)
        # If none provided, use the singleton SystemManager
        self.system = system if system else SystemManager() 
        
        self.model = None
        self.processor = None
        
        # Configuration for Local Model Path
        # Attempt to load model if configured
        # Users can update this path to their local download of Qwen-VL
        # Defaulting to Qwen2.5-VL-3B-Instruct for efficiency
        self.local_model_path = os.environ.get("QWEN_MODEL_PATH", "../models/Qwen2.5-VL-3B-Instruct")
        
        self.is_model_loaded = False
        self._load_local_qwen()

    def _load_local_qwen(self):
        """
        Loads the Qwen2.5-VL model from the local path if available.
        """
        if not HAS_QWEN:
            print("Warning: Qwen dependencies not installed. Please run `pip install -r requirements.txt`.")
            return

        if os.path.exists(self.local_model_path):
            print(f"Loading Qwen-VL from {self.local_model_path}...")
            try:
                # Use Qwen2_5_VLForConditionalGeneration or fall back to AutoModel if using a different variant
                # Added trust_remote_code=True for newer/custom models like Qwen3-Thinking
                if HAS_QWEN_CLASS:
                    model_cls = Qwen2_5_VLForConditionalGeneration
                else:
                    # Fallback for older transformers versions
                    model_cls = AutoModelForCausalLM

                self.model = model_cls.from_pretrained(
                    self.local_model_path, 
                    torch_dtype="auto", 
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(self.local_model_path, trust_remote_code=True)
                self.is_model_loaded = True
                print("Qwen-VL loaded successfully.")
            except Exception as e:
                print(f"Error loading Qwen-VL: {e}")
                self.is_model_loaded = False
        else:
            print(f"Qwen-VL model path not found at: {self.local_model_path}")

    def _generate_agent_response(self, prompt: str):
        try:
            messages = [
                {"role": "system", "content": "You are the System Manager for the MCC AI Concrete Structure Construction system. Your goal is to assist the user in managing the workflow, updating configurations, and resolving errors. Be concise and helpful."},
                {"role": "user", "content": prompt}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            # Decode only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return output_text
        except Exception as e:
            print(f"Error in agent generation: {e}")
            return None

    async def chat_with_user(self, message: str) -> dict:
        """
        Process a text message from the user, simulating a VL/LLM agent (RPA Manager).
        It can inspect system state and trigger actions.
        """
        if not self.is_model_loaded:
            return await self._chat_fallback_regex(message)

        # 1. Build Agent Context
        status = self.system.status.value
        config_str = json.dumps(self.system.config)
        recent_logs = "\n".join(self.system.logs[-3:]) if self.system.logs else "No logs yet."
        
        prompt = f"""
Current System State:
- Status: {status}
- Configuration: {config_str}
- Recent Logs:
{recent_logs}

User Message: "{message}"

Instructions:
1. Analyze the user's intent.
2. If the user wants to update a parameter (floor_count, conf_threshold, etc.), output a JSON action block.
3. If the user's request is ambiguous (e.g., "change floors" without a number), ASK for clarification.
4. If the system is in an ERROR or PAUSED state, guide the user on how to resolve it.

Essense:
The user is an Construction Engineer who wants to send 2D input in PDF format to the system and receive a 3D model in IFC format.

Output Format:
Provide a natural language response first.
If an action is required, append a JSON block at the end like this:
```json
{{
  "action": "update_config",
  "key": "floor_count",
  "value": 5
}}
```
OR
```json
{{
  "action": "command",
  "command": "retry"
}}
```
"""
        # Generate response
        response_text = self._generate_agent_response(prompt)
        
        if not response_text:
             return self._chat_fallback_regex(message)

        # Parse Response
        response = {
            "reply": response_text,
            "updated_params": {}
        }
        
        # Extract JSON if present
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                action_data = json.loads(json_match.group(1))
                # Remove the JSON block from the reply to keep it clean for the user
                response["reply"] = response_text.replace(json_match.group(0), "").strip()
                
                if action_data.get("action") == "update_config":
                    key = action_data.get("key")
                    val = action_data.get("value")
                    if key and val is not None:
                        self.system.update_config(key, val)
                        response["updated_params"][key] = val
                        
                elif action_data.get("action") == "command":
                    cmd = action_data.get("command")
                    if cmd == "retry" and self.system.status == SystemStatus.PAUSED:
                        result = await self.system.resume_workflow()
                        if result.get("status") == "success":
                            response["reply"] += f"\n[System] Success! Download: {result.get('ifc_url')}"
                        else:
                            response["reply"] += f"\n[System] Failed: {result.get('message')}"
                            
            except Exception as e:
                print(f"Failed to parse agent action: {e}")

        return response

    async def _chat_fallback_regex(self, message: str) -> dict:
        """
        Original regex-based implementation for fallback.
        Now async to support workflow control.
        """
        message_lower = message.lower()
        response = {
            "reply": "I received your message.",
            "updated_params": {}
        }
        
        # 1. System Status Check
        if "status" in message_lower or "what is happening" in message_lower:
            status = self.system.status.value
            last_log = self.system.logs[-1] if self.system.logs else "No logs yet."
            response["reply"] = f"Manager: Current System Status is [{status.upper()}].\nLast Activity: {last_log}"
            return response

        # 2. Configuration Updates
        
        # Floor Count
        floor_match = re.search(r"(\d+)\s*floor", message_lower)
        if floor_match:
            count = int(floor_match.group(1))
            self.system.update_config("floor_count", count)
            response["updated_params"]["floor_count"] = count
            response["reply"] = f"Manager: I've updated the plan to {count} floors."

        # Generation Mode (Simple vs Advanced)
        if "advanced" in message_lower or "gnn" in message_lower or "graph" in message_lower:
            self.system.update_config("generation_mode", "advanced")
            response["updated_params"]["generation_mode"] = "advanced"
            response["reply"] = "Manager: Switched to Advanced Mode (GNN-based Structural Reconstruction)."
        elif "simple" in message_lower or "rule" in message_lower or "basic" in message_lower:
            self.system.update_config("generation_mode", "simple")
            response["updated_params"]["generation_mode"] = "simple"
            response["reply"] = "Manager: Switched to Simple Mode (Rule-based Extrusion)."

        # Confidence Threshold
        conf_match = re.search(r"(?:threshold|conf|confidence)\s*(?:to|is|=)?\s*([0-9]*\.?[0-9]+)", message_lower)
        if conf_match:
            try:
                val = float(conf_match.group(1))
                if 0 < val < 1.0:
                    self.system.update_config("conf_threshold", val)
                    response["updated_params"]["conf_threshold"] = val
                    response["reply"] = f"Manager: Confidence threshold set to {val}."
                else:
                    response["reply"] = "Manager: Confidence threshold must be between 0 and 1."
            except ValueError:
                pass

        # 3. Intervention / Workflow Control 
        if "retry" in message_lower or "resume" in message_lower or "try again" in message_lower: 
            if self.system.status == SystemStatus.PAUSED: 
                response["reply"] = "Manager: Resuming workflow with the current settings..." 
                result = await self.system.resume_workflow() 
                
                if result.get("status") == "success": 
                    response["reply"] += f"\nSuccess! Process completed. Download: {result.get('ifc_url')}" 
                elif result.get("status") == "paused": 
                    response["reply"] += f"\nStill no luck. Reason: {result.get('message')}" 
                else: 
                    response["reply"] += f"\nError encountered: {result.get('message')}" 
                return response 
            else:
                response["reply"] = "Manager: The system is not currently paused, so there is nothing to resume."
                return response

        # 4. Proactive Clarification & Safety Checks
        elif "floor" in message_lower and not response["updated_params"] and ("set" in message_lower or "change" in message_lower):
            response["reply"] = "Manager: You mentioned setting the floor count, but I missed the number. How many floors should I assume?"
            
        elif ("threshold" in message_lower or "confidence" in message_lower) and not response["updated_params"] and ("set" in message_lower or "change" in message_lower):
            response["reply"] = "Manager: What confidence threshold should I use? (0.0 to 1.0)"

        # Fallback
        elif response["reply"] == "I received your message.": 
             response["reply"] = "Manager: (Fallback Mode) I am monitoring the workflow. If you need to change settings, just tell me (e.g., 'set floors to 5')." 

        return response

    def analyze_structure(self, image_path: str, prompt: str) -> str:
        """
        Analyze the image using the VLM to understand spatial relationships.
        
        Args:
            image_path (str): Path to the engineering drawing.
            prompt (str): Question or instruction for the model.
            
        Returns:
            str: The model's textual analysis.
        """
        if self.model and self.processor and HAS_QWEN:
            return self._analyze_with_qwen(image_path, prompt)

        # Mock response for development without heavy model weights
        return f"Mock Analysis: The image contains a structural layout with columns arranged in a grid. Detected beam connections between columns."

    def _analyze_with_claude(self, image_path: str, prompt: str, api_key: str):
        # Implementation for Claude 3/4.5 Vision API
        pass
    
    def _analyze_with_qwen(self, image_path: str, prompt: str):
        # Implementation for Qwen-VL local inference
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            # Move inputs to same device as model
            inputs = inputs.to(self.model.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # The output usually contains the prompt too, we might want to strip it
            # But for now returning the raw list/string is fine
            return output_text[0]
            
        except Exception as e:
            return f"Error running Qwen-VL: {str(e)}"
