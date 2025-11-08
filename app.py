from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from phishing_Detection import get_prediction_from_url
import os

app = Flask(__name__, static_folder='.', static_url_path='')  # this line serves index.html
CORS(app)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing URL input'}), 400

    url_input = data['url'].strip()
    if not url_input:
        return jsonify({'error': 'Empty URL input'}), 400

    try:
        if '.' not in url_input or len(url_input) < 5:
            raise ValueError('Invalid URL format')
    except Exception:
        return jsonify({'error': 'Invalid URL format'}), 400

    try:
        prediction = get_prediction_from_url(url_input)
        return jsonify({'result': prediction})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

