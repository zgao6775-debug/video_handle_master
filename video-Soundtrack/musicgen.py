import replicate
import requests
from moviepy.editor import *


def generate_music(prompt):

  video = VideoFileClip("./static/video.mp4")
  duration = video.duration
  video.close()

  output = replicate.run(
    "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906",
    input={
      "model_version": "large",
      "prompt": prompt,
      "duration": int(duration),
    })

  response = requests.get(output)
  with open("./static/audio.wav", 'wb') as f:
    f.write(response.content)
