from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
import gradio as gr
from PIL import Image
import torch
import os
import time
import math

model_name_and_path = "/app/stabilityai/sdxl-turbo"
max_64_bit_int = 2 ** 63 - 1

# check if NSFW checker is enabled
flagNSFW = False
envNSFW = os.environ.get("NSFW", "false").strip().lower()
if envNSFW != "off" and envNSFW != "0" and envNSFW != "false":
    flagNSFW = True

device = torch.device("cuda")
torch_dtype = torch.float16

pipelines = {
    "img2img": AutoPipelineForImage2Image.from_pretrained(
        model_name_and_path, torch_dtype=torch_dtype, variant="fp16"
    ),
    "txt2img": AutoPipelineForText2Image.from_pretrained(
        model_name_and_path, torch_dtype=torch_dtype, variant="fp16"
    ),
}

if flagNSFW != True:
    pipelines["txt2img"].safety_checker = None
    pipelines["img2img"].safety_checker = None

pipelines["txt2img"].to(device)
pipelines["txt2img"].set_progress_bar_config(disable=True)

pipelines["img2img"].to(device)
pipelines["img2img"].set_progress_bar_config(disable=True)


def resize_crop(image: Image, size: int = 512):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    w, h = image.size
    image = image.resize((size, int(size * (h / w))), Image.BICUBIC)
    return image


async def predict(
    image: Image,
    prompt: str,
    strength: float = 0.7,
    guidance: float = 0.0,
    steps: int = 2,
    seed: int = 42,
):

    if image is not None:
        image = resize_crop(image)
        generator = torch.manual_seed(seed)
        last_time = time.time()

        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        results = pipelines["img2img"](
            prompt=prompt,
            image=image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            width=512,
            height=512,
            output_type="pil",
        )
    else:
        generator = torch.manual_seed(seed)
        last_time = time.time()
        results = pipelines["txt2img"](
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=512,
            height=512,
            output_type="pil",
        )
    print(f"Pipe took {time.time() - last_time} seconds")
    nsfw_content_detected = (
        results.nsfw_content_detected[0]
        if "nsfw_content_detected" in results
        else False
    )
    if nsfw_content_detected:
        gr.Warning("NSFW content detected.")
        return Image.new("RGB", (512, 512))
    return results.images[0]


with gr.Blocks() as app:
    init_image_state = gr.State()
    with gr.Column():
        with gr.Row():
            prompt = gr.Textbox(placeholder="Prompt", scale=5, container=False)
            submit = gr.Button("Generate", scale=1)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=["upload", "webcam", "clipboard"],
                    label="Webcam",
                    type="pil",
                )
            with gr.Column():
                generated = gr.Image(type="filepath")
                with gr.Accordion("Advanced options", open=False):
                    strength = gr.Slider(
                        label="Strength",
                        value=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                    )
                    guidance = gr.Slider(
                        label="Guidance",
                        value=0.0,
                        minimum=0.0,
                        maximum=2.0,
                        step=0.001,
                    )
                    steps = gr.Slider(
                        label="Steps", value=2, minimum=1, maximum=40, step=1
                    )
                    seed = gr.Slider(
                        randomize=True,
                        minimum=0,
                        maximum=12013012031030,
                        label="Seed",
                        step=1,
                    )

        inputs = [image_input, prompt, strength, guidance, steps, seed]
        submit.click(fn=predict, inputs=inputs, outputs=generated, show_progress=False)
        prompt.change(fn=predict, inputs=inputs, outputs=generated, show_progress=False)
        strength.change(
            fn=predict, inputs=inputs, outputs=generated, show_progress=False
        )
        guidance.change(
            fn=predict, inputs=inputs, outputs=generated, show_progress=False
        )
        steps.change(fn=predict, inputs=inputs, outputs=generated, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=generated, show_progress=False)
        image_input.change(
            fn=lambda x: x,
            inputs=image_input,
            outputs=init_image_state,
            show_progress=False,
            queue=False,
        )

app.queue()
app.launch(share=False, server_name="0.0.0.0", ssl_verify=False)