"""
Configuration variables for AI Personal Stylist.
Centralizes all configurable settings for easy tuning.
"""

# Paths
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Image settings
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

# Vision / detection settings
DEFAULT_IMAGE_SIZE = (512, 512)

# Recommender settings
DEFAULT_TOP_K_RECOMMENDATIONS = 5

# Generator settings
DEFAULT_NUM_IMAGES = 1
# Stable Diffusion img2img: lower strength = more face preservation
GENERATOR_STRENGTH = 0.45
GENERATOR_GUIDANCE_SCALE = 7.5
GENERATOR_STEPS = 30
GENERATOR_MODEL_ID = "runwayml/stable-diffusion-v1-5"
