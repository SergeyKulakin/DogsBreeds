from flask import Flask, request, render_template, redirect, url_for, abort, flash, jsonify, make_response
from werkzeug.utils import secure_filename
import os
from PIL import Image
import model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.jpeg', '.JPEG']
app.config['UPLOAD_PATH'] = 'static/img'


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

'''@login_required
def delete_item(item_id):
    new_id = item_id
    item = self.session.query(Item).get(item_id)
    os.remove(os.path.join(app.config['UPLOADED_ITEMS_DEST'], item.filename))
    self.session.delete(item)
    db.session.commit()
    return redirect(url_for('admin_items'))'''


if __name__ == '__main__':
    app.run()