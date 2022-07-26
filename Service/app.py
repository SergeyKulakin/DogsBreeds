from flask import Flask, request, render_template, redirect, url_for, abort
from werkzeug.utils import secure_filename
from argparse import ArgumentParser
import os
from PIL import Image
import model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2024 * 2024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.jpeg', '.JPEG']
app.config['UPLOAD_PATH'] = 'static/img'

def parse_args():
    """Parse input arguments"""
    parser = ArgumentParser('Test web app')
    parser.add_argument('--port', type=int, help='Port to use')
    parser.add_argument('--host', type=str, help='Host to use')
    return parser.parse_args()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        file_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(file_path)
        with Image.open(file_path) as im:
            label = model.predict_image(im)

        return render_template('predict.html', predict=label.capitalize(), user_image=file_path)
    else:
        return redirect(url_for('home'))


if __name__ == '__main__':
    args = parse_args()
    app.run(host=args.host, port=args.port)