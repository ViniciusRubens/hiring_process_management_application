import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """
    Application configuration class.
    Handles environment variables and validates system requirements.
    """

    APP_NAME = os.getenv("APP_NAME", "ML_Service")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))

    # ML Artifacts
    MODEL_PATH = os.getenv("MODEL_PATH")
    SCALER_PATH = os.getenv("SCALER_PATH")

    @classmethod
    def check_artifacts(cls):
        """
        Validates if the pickle files exist at the specified paths
        """

        required_paths = [cls.MODEL_PATH, cls.SCALER_PATH]

        for path in required_paths:
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"Critical Error: Artifact not found at {path}")
            
        print(f"--- [INFO] {cls.APP_NAME}: All ML artifacts are ready. ---")