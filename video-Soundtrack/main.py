from flask import Flask, request, jsonify, render_template, send_file
from generator import main
from musicgen import generate_music
from musicprompt import generate_music_prompt
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'flv', 'wmv'}


def combine_video_audio():
  cmd = "ffmpeg -y -i ./static/video.mp4 -i ./static/audio.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac ./static/combined_video.mp4"
  os.system(cmd)


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_video():

  descriptions=None
  music_prompt=None

  if request.method == 'POST':
    if 'file' not in request.files:
      return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
      return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
      return jsonify({'error': 'Invalid file type'}), 400

    input_video = "./static/video.mp4"

    file.save(input_video)

    descriptions, music_prompt = main()

    combine_video_audio()

  return render_template('output.html',
                         descriptions=descriptions,
                         music_prompt=music_prompt)


@app.route('/regenerate/prompt', methods=['POST', 'GET'])
def regenerate_prompt():
  data = request.get_json()
  video_description = data['description']

  music_prompt = generate_music_prompt(video_description)

  return jsonify({'music_prompt': music_prompt})


@app.route('/regenerate/music', methods=['POST', 'GET'])
def regenerate_music():
  data = request.get_json()
  music_prompt = data['music_prompt']

  generate_music(music_prompt)
  combine_video_audio()

  return jsonify({'success': True})


app.run(host='0.0.0.0', port=5000)
