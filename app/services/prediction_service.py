import pickle
import pandas as pd
from config.config import AppConfig

class PredictorService:
    """
    Service layer for Machine Learning inference.
    """
    
    def __init__(self):
        AppConfig.check_artifacts()
        self._model = self._load_artifact(AppConfig.MODEL_PATH)
        self._scaler = self._load_artifact(AppConfig.SCALER_PATH)
        
        # Define the exact column order used during model training
        self._training_columns = [
            'english_proficiency', 
            'cognitive_ability', 
            'aptitude_test_result'
        ]

    def _load_artifact(self, path: str):
        """Safely loads a pickle file."""
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            raise RuntimeError(f"Error loading ML artifact: {e}")

    def predict_status(self, candidate_data: dict) -> dict:
        """
        Processes inference using original dataset column names.
        """

        # Convert dictionary to DataFrame and ensure correct column order
        input_df = pd.DataFrame([candidate_data])[self._training_columns]

        # Apply scaling
        scaled_features = self._scaler.transform(input_df)

        # Prediction (0 = Denied, 1 = Allowed)
        prediction = self._model.predict(scaled_features)
        status_code = int(prediction[0])

        return {
            "status": status_code,
            "is_eligible": bool(status_code == 1),
            "display_message": "Candidate is eligible for enrollment." if status_code == 1 
                               else "Candidate does not meet the eligibility criteria."
        }

# Singleton instance
ml_service = PredictorService()