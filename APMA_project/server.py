import os
from flask import Flask, render_template, request, send_from_directory
from face_detection import detect_faces
from prediction import predict_face
import glob

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')

    for upload in request.files.getlist("file"):
        filename = "img.jpg"
        destination = "/".join([target, filename])
        upload.save(destination)

    image = os.path.join(APP_ROOT, 'static/img.jpg')
    detect_faces(image)
    image_path = os.path.join(APP_ROOT, 'static/file.jpg')
    predict_face(image_path)

    return "done"


@app.route('/show')
def show():
    return send_from_directory(os.path.join(APP_ROOT, 'static/'), filename='img.jpg', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)