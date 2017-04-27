import os
from flask import Flask, render_template, request, send_from_directory
from face_detection import detect_faces
from prediction import predict_face
import time
import glob

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    t = time.time()
    for upload in request.files.getlist("file"):
        filename = str(t) + "upload.jpg"
        destination = "/".join([target, filename])
        upload.save(destination)

    image = os.path.join(APP_ROOT, 'static/'+filename)
    detect_faces(image,filename)
    image_path = os.path.join(APP_ROOT, 'static/'+filename+'face.jpg')
    result = predict_face(image_path)

    return render_template('show.html', filename=filename, anger=result[0][0], contempt=result[1][0],
                           disgust=result[2][0], fear=result[3][0], happy=result[4][0], neutral=result[5][0],
                           sadness=result[6][0], surprise=result[7][0])


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

