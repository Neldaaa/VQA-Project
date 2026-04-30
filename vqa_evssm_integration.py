"""
VQA + EVSSM Integration Module
Combines EVSSM image deblurring with Gemini VQA
"""

import logging
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT COMPONENTS
# ============================================================================

try:
    from utils.image_processor import EVSSMImageProcessor, get_processor_status
    HAS_EVSSM = True
except ImportError as e:
    logger.warning(f"Could not import EVSSM processor: {e}")
    HAS_EVSSM = False


# ============================================================================
# VQA PIPELINE
# ============================================================================

class VQAPipeline:
    """
    Integrated pipeline: Image deblurring + VQA
    """
    
    def __init__(self, enable_deblur: bool = True):
        """
        Initialize VQA Pipeline
        
        Args:
            enable_deblur: Enable EVSSM deblurring preprocessing
        """
        self.enable_deblur = enable_deblur
        self.processor = None
        
        if enable_deblur and HAS_EVSSM:
            try:
                self.processor = EVSSMImageProcessor(enable_deblur=True)
                logger.info("✅ EVSSM Image Processor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize EVSSM: {e}")
                self.processor = None
        
        logger.info(f"✅ VQA Pipeline initialized (deblur={enable_deblur})")
    
    def preprocess_image(
        self, 
        image: Image.Image,
        detect_blur: bool = True
    ) -> Image.Image:
        """
        Preprocess image with EVSSM deblurring
        
        Args:
            image: Input PIL Image
            detect_blur: Only deblur if detected as blurry
            
        Returns:
            Processed Image
        """
        if not self.processor:
            logger.debug("EVSSM processor not available, returning original image")
            return image
        
        try:
            logger.debug("🔧 Starting EVSSM preprocessing...")
            processed = self.processor.process(image, detect_blur=detect_blur)
            logger.debug("✅ EVSSM preprocessing completed")
            return processed
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return image
    
    def get_status(self) -> dict:
        """Get pipeline status"""
        status = {
            "deblur_enabled": self.enable_deblur,
            "processor_available": self.processor is not None,
        }
        
        if self.processor:
            status["processor_status"] = self.processor.get_status()
        
        return status


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global pipeline instance
_pipeline = None

def create_vqa_pipeline(enable_deblur: bool = True) -> VQAPipeline:
    """
    Create or get VQA Pipeline instance
    
    Args:
        enable_deblur: Enable EVSSM preprocessing
        
    Returns:
        VQAPipeline instance
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = VQAPipeline(enable_deblur=enable_deblur)
    return _pipeline


def preprocess_for_vqa(image: Image.Image, detect_blur: bool = True) -> Image.Image:
    """
    Preprocess image for VQA using EVSSM
    
    Convenience function for app.py integration
    
    Args:
        image: Input PIL Image
        detect_blur: Only deblur if detected as blurry
        
    Returns:
        Preprocessed Image
    """
    pipeline = create_vqa_pipeline()
    return pipeline.preprocess_image(image, detect_blur=detect_blur)


def get_vqa_status() -> dict:
    """Get VQA pipeline status"""
    pipeline = create_vqa_pipeline()
    return pipeline.get_status()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_evssm_available() -> bool:
    """Check if EVSSM is available"""
    return HAS_EVSSM


def check_model_file_exists() -> bool:
    """Check if EVSSM model weights file exists"""
    try:
        from config.evssm_config import MODEL_WEIGHTS_PATH
        return MODEL_WEIGHTS_PATH.exists()
    except Exception:
        return False


def print_integration_info():
    """Print integration information"""
    print("\n" + "="*60)
    print("VQA + EVSSM Integration Status")
    print("="*60)
    print(f"EVSSM Available: {check_evssm_available()}")
    print(f"Model File Exists: {check_model_file_exists()}")
    
    if check_evssm_available():
        status = get_vqa_status()
        print(f"Pipeline Status: {status}")
    
    print("="*60 + "\n")


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print info
    print_integration_info()
    
    # Create and test pipeline
    try:
        pipeline = create_vqa_pipeline()
        print("✅ VQA Pipeline created successfully")
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
