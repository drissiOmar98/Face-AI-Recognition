from flask import Flask
from flask_cors import CORS

from app.routes import api

app = Flask(__name__)
CORS(app)  # allow Angular frontend

# register blueprint
app.register_blueprint(api, url_prefix='/api')

# optional home route
@app.route('/')
def home():
    return "Welcome to the Face Recognition API!"

if __name__ == '__main__':
    app.run(debug=True)
