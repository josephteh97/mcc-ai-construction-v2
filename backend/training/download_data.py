from roboflow import Roboflow
import os
import sys

import time
from urllib.error import URLError

def download_dataset(api_key: str):
    """
    Download the 'columns-fncne' dataset from Roboflow.
    
    Args:
        api_key (str): Your private Roboflow API Key.
    """
    rf = Roboflow(api_key=api_key)
    # Workspace: columns-and-ducts
    # Project: columns-and-ducts-detection (Updated based on user link)
    
    print("Accessing Roboflow Workspace...")
    project = rf.workspace("columns-and-ducts").project("columns-and-ducts-detection")
    
    print("Downloading dataset...")
    
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            
            # Try YOLOv11 first
            try:
                print("Requesting 'yolov11' format...")
                dataset = project.version(1).download("yolov11")
                
                # Check if download was successful (folder not empty)
                if not os.path.exists(dataset.location) or not os.listdir(dataset.location):
                    print(f"Warning: Downloaded folder '{dataset.location}' is empty or does not exist.")
                else:
                    print(f"Dataset downloaded to: {dataset.location}")
                    file_count = sum([len(files) for r, d, files in os.walk(dataset.location)])
                    print(f"Total files found: {file_count}")
                    
                return dataset.location
            except Exception as e:
                # If it's a connection error, re-raise to outer loop
                error_msg = str(e).lower()
                if "connection" in error_msg or "reset by peer" in error_msg or "handshake" in error_msg:
                    print(f"Connection error during yolov11 download: {e}")
                    raise e
                
                # Otherwise, assume format issue and try yolov8
                print(f"Format 'yolov11' issue ({e}), falling back to 'yolov8'...")
                dataset = project.version(1).download("yolov8")
                
                # Check if download was successful (folder not empty)
                if not os.listdir(dataset.location):
                    print(f"Warning: Downloaded folder '{dataset.location}' is empty.")
                else:
                    print(f"Dataset downloaded to: {dataset.location}")
                    file_count = sum([len(files) for r, d, files in os.walk(dataset.location)])
                    print(f"Total files found: {file_count}")
                
                return dataset.location

        except (ConnectionError, URLError, OSError, Exception) as e:
            # Check if it's a network-related error that we should retry
            error_msg = str(e).lower()
            is_network_error = (
                "connection" in error_msg or 
                "reset by peer" in error_msg or 
                "handshake" in error_msg or
                "timed out" in error_msg
            )
            
            if is_network_error:
                print(f"Network error detected: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            
            # If not a network error or we ran out of retries
            if attempt == max_retries - 1:
                print("Failed to download dataset after multiple attempts.")
                raise e
            else:
                 # If it's not a network error, we probably shouldn't retry, but let's be safe and print
                 print(f"Encountered error: {e}")
                 raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_data.py <ROBOFLOW_API_KEY>")
        print("Or set ROBOFLOW_API_KEY environment variable.")
        api_key = os.getenv("ROBOFLOW_API_KEY")
    else:
        api_key = sys.argv[1]
        
    if not api_key:
        print("Error: No API Key provided.")
        sys.exit(1)
        
    download_dataset(api_key)
