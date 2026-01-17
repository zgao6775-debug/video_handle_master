from replit.ai.modelfarm import CompletionModel

def generate_music_prompt(image_descriptions):

  model = CompletionModel("text-bison")
  response = model.complete(["You are a famous music composer. Describe the music (instruments, genre, mood, feel, time signature, bpm, kbps, khz) most suitable for this video, keep it concise:" + image_descriptions], temperature=0.2)
  
  result=response.responses[0].choices[0].content

  print("Music prompt: ", result)
  return result
  