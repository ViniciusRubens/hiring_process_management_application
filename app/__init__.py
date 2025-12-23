from flask import Flask
from app.routes.api_routes import main_bp
from config.config import AppConfig

def create_app():
    """
    Application Factory pattern to initialize the Flask App.
    Configures settings and registers blueprints.
    """
    
    app = Flask(__name__)
    
    # Load configurations
    app.config.from_object(AppConfig)

    # Registering blueprints to organize routes
    app.register_blueprint(main_bp)
    
    return app