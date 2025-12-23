from flask import Blueprint
from app.controllers.prediction_controller import PredictionController

# Defining a Blueprint for main application routes
main_bp = Blueprint('main', __name__)

# Route for the landing page (GET)
main_bp.route('/', methods = ['GET'])(PredictionController.render_index)

# Route for model inference (POST)
main_bp.route('/predict', methods = ['POST'])(PredictionController.handle_prediction)