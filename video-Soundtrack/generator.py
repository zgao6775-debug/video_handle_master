from extract import extract_images
from describe import describe_images
from musicprompt import generate_music_prompt
from musicgen import generate_music


def main():
  
  extract_images("./static/video.mp4")
  descriptions = describe_images()
  music_prompt = generate_music_prompt(descriptions)
  generate_music(music_prompt)

  return descriptions, music_prompt
