from app import create_app
from config.config import AppConfig

app = create_app()

if __name__ == "__main__":
    # Start the Flask application using configuration from AppConfig
    app.run(
        host = AppConfig.HOST,
        port = AppConfig.PORT,
        debug = AppConfig.DEBUG
    )