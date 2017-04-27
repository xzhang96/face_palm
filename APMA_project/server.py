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


#if __name__ == '__main__':
#    app.run(debug=True)

if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:

        python server.py

    Show the help text using:

        python server.py --help

    """

    HOST, PORT = host, port
    print( "running on %s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()

