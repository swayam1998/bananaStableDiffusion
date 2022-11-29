import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image

model_inputs = {
	"prompt": "concept art of a cat, cute, cottagecore, bloom, trending on artstation, illustration by james jean, ilya kuvshinov, greg rutkowski, loish van baarle, fine details, dramatic light, magical light, very crisp image, concept art of a cat witch, cute, cottagecore, bloom, trending on artstation, illustration by james jean, ilya kuvshinov, greg rutkowski, loish van baarle, fine details, dramatic light, magical light",
	"num_inference_steps":50,
	"guidance_scale":9,
	"height":512,
	"width":512,
	"seed":3242
}

api_key = "ba7d6d62-66ce-4aa3-a0ef-d24412f53e70"
model_key = "504c071d-c6e6-4a47-83bd-7b33e2a45bd1"

# Run the model
out = banana.run(api_key, model_key, model_inputs)

# Extract the image and save to output.jpg
image_byte_string = out["modelOutputs"][0]["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")
