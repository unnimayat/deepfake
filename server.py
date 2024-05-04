from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
from audio import predict_audio

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

UPLOAD_FOLDER='Uploaded_Files'

app = Flask("__main__",template_folder="templates")
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/audio_detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('audio.html')
    if request.method == 'POST':
        video = request.files['audio']
        print(video.filename)
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        audio_path = "Uploaded_Files/" + video_filename
        predicted_class = predict_audio(audio_path)
        print("Predicted Class:", predicted_class)
        return render_template('audio.html', data=predicted_class)
        
@app.route('/')
def index():
    return render_template('index.html')
app.run(port=3000)