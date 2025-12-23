from flask import render_template, request
from app.services.prediction_service import ml_service
from app.schemas.schema import CandidateData
from pydantic import ValidationError

class PredictionController:
    """
    Controller responsible for handling web requests and returning responses.
    """

    @staticmethod
    def render_index():
        """Renders the main dashboard page."""
        return render_template('index.html')

    @staticmethod
    def handle_prediction():
        """
        Receives input, validates it via Pydantic, and triggers inference.
        """

        try:
            # Extract data from the form
            form_payload = request.form.to_dict()

            # Validate input using Pydantic
            # This ensures types and ranges (0-10, 0-200, etc.) are respected
            validated_data = CandidateData(**form_payload)

            # Perform prediction via Service Layer
            # .model_dump() converts the Pydantic object back to a clean dictionary
            result = ml_service.predict_student_status(validated_data.model_dump())

            return render_template(
                'index.html', 
                result = result['display_message'],
                status = result['status']
            )

        except ValidationError as e:
            # Format Pydantic errors for the user
            error_msg = f"Data Validation Error: {e.errors()[0]['msg']}"
            return render_template('index.html', result = error_msg), 400

        except Exception as e:
            # General system error handling
            return render_template('index.html', result = f"Service Error: {str(e)}"), 500