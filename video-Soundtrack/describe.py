from replit.ai.modelfarm import CompletionModel
import replicate
import os


def describe_images():

  descriptions = []

  for file in os.listdir("./frames"):

    output = replicate.run(
      "gfodor/instructblip:ca869b56b2a3b1cdf591c353deb3fa1a94b9c35fde477ef6ca1d248af56f9c84",
      input={
        "image_path": open("frames/" + file, "rb"),
        "prompt": "Describe the scene of this frame from a video"
      },
    )

    descriptions.append(output)

  model = CompletionModel("text-bison")
  response = model.complete(
    ["Summarize the description of this video from the descriptions of its frames" + "".join(descriptions)],
    temperature=0.2)

  result = response.responses[0].choices[0].content

  print("Description: ", result)
  return result
