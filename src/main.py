from flask import Flask, Blueprint
from src.appgoogle import blueprint as google_app
from src.servicedb import blueprint as db_app, init_oauth
from authlib.integrations.flask_client import OAuth

# Crear una nueva aplicación Flask principal
app = Flask(__name__)

# Inicializar OAuth
oauth = OAuth(app)
google = init_oauth(app)

# Registrar los blueprints de cada aplicación
app.register_blueprint(google_app)
app.register_blueprint(db_app)

if __name__ == "__main__":
    port = 5000
    app.run(host='0.0.0.0', port=port)