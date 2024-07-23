# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from io import BytesIO

import torch
from diffusers import StableDiffusion3Pipeline
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

# Initialize pipeline
pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3-medium-diffusers',
                                                torch_dtype=torch.float16)
pipe = pipe.to('cuda')

# Create a FastAPI application
app = FastAPI()


# Define the input data model
class CaptionRequest(BaseModel):
    caption: str


# Defining API endpoints
@app.post('/generate_image/')
async def generate_image(request: CaptionRequest):
    caption = request.caption
    negative_prompt = 'blurry, low resolution, artifacts, unnatural, poorly drawn, bad anatomy, out of focus'
    image = pipe(
        caption,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.0
    ).images[0]

    # Converts an image to a byte stream
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type='image/png')


# Run the Uvicorn server
if __name__ == '__main__':
    import argparse

    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=11005, type=int)
    args = parser.parse_args()

    uvicorn.run(app, host='0.0.0.0', port=args.port)
