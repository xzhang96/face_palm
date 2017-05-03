import os
from flask import Flask, render_template, request
from face_detection import detect_faces
from prediction import predict_face
import time


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')    # Route the Home page
def index():
    return render_template('upload.html')     # upload.html is rendered


@app.route('/upload', methods=['POST'])   # Route the /upload page, after the image is uploaded by user
def upload():
    target = os.path.join(APP_ROOT, 'static/')   # Define the path where the uploaded image would be saved
    t = time.time()      # Create a time stamp to distinguish each uploaded image
    for upload in request.files.getlist("file"):
        filename = str(t) + "upload.jpg"       # Obtain the filename of the uploaded image
        destination = "/".join([target, filename])     # Define the path of the image
        upload.save(destination)     # Save the image in the 'static' directory

    image = os.path.join(APP_ROOT, 'static/'+filename)    # Get the path of the original image
    detect_faces(image, filename)     # Detect the face on the uploaded image
    image_path = os.path.join(APP_ROOT, 'static/'+filename+'face.jpg')    # Get the path of the gray-scaled image

    # Check if there is any face detected
    if os.path.exists(image_path):
        result = predict_face(image_path)     # Use the trained model to predict the probabilities

        # 'show.html' is rendered to show the results, passing the filename and the results to html page
        return render_template('show.html', filename=filename, anger=result[0][0], contempt=result[1][0],
                               disgust=result[2][0], fear=result[3][0], happy=result[4][0], neutral=result[5][0],
                               sadness=result[6][0], surprise=result[7][0])
    else:
        # 'notshow.html' is rendered to show an error message
        return render_template('notshow.html')

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
        print("running on %s:%d" % (HOST, PORT))
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


    run()

