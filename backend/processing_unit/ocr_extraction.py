from paddleocr import PaddleOCR
import os

class OCRExtractor:
    def __init__(self, lang: str = 'en'):
        """
        Initialize PaddleOCR.
        
        Args:
            lang (str): Language code (e.g., 'en', 'ch').
        """
        # use_angle_cls=True enables orientation classification
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract_text(self, image_path: str) -> list:
        """
        Extract text from an image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            list: List of detected text blocks with coordinates and confidence.
                  Format: [[[x1,y1], [x2,y2], ...], "text", confidence]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        result = self.ocr.ocr(image_path, cls=True)
        
        # PaddleOCR returns a list of lists (one per page/image)
        # We flatten it for simpler usage if it's a single image
        if not result or result[0] is None:
            return []
            
        return result[0]

    def extract_dimensions(self, image_path: str) -> list:
        """
        Specific method to look for numeric dimensions.
        """
        raw_results = self.extract_text(image_path)
        dimensions = []
        for line in raw_results:
            coords = line[0]
            text = line[1][0]
            conf = line[1][1]
            
            # Simple heuristic: if text contains digits, treat as potential dimension
            if any(char.isdigit() for char in text):
                dimensions.append({
                    "text": text,
                    "coords": coords,
                    "confidence": conf
                })
        return dimensions
