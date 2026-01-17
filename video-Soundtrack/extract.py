import os
from moviepy.editor import *
from PIL import Image


def extract_images(video_path):
  video_clip = VideoFileClip(video_path)

 
  folder = './frames'
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))


  for i in range(0, int(video_clip.duration), 2):
    frame = video_clip.get_frame(i)
    new_img = Image.fromarray(frame)
    new_img.save(f'./frames/frame{i//2}.png')

  video_clip.close()
