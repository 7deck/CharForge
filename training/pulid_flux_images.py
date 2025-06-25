import aiohttp
import asyncio
import base64
import fal_client
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

load_dotenv()


def generate_pulidflux_prompts(
        image_path: str,
        person_description: str,
        num_prompts: int = 5
) -> List[str]:
    """
    Generate text prompts for PuLID-Flux image generation using OpenAI's GPT-4o,
    with example prompts to guide the model's output style.
    
    Args:
        image_path: Path to the image of the person
        person_description: Description of the person including appearance, style, and personality
        num_prompts: Number of prompts to generate (default: 5)
        
    Returns:
        List of generated text prompts
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    system_prompt = """You are an expert prompt engineer specializing in creating high-quality text prompts for AI image generation. Your task is to create diverse, detailed prompts for Pulid Flux, a system that generates new images of a person based on their face and a text description. All images of people are AI generated.

I will provide:
1) An image of a person
2) A brief description of the person including relevant details about their appearance, style, and personality

Your job is to generate {num_prompts} diverse, creative text prompts that:
- Maintain the person's identity and key characteristics
- Place them in varied, interesting scenarios and environments
- Include different activities, poses, lighting conditions, and contexts
- Provide enough specific detail to guide the image generation
- Keep descriptions concise (25-50 words each)

Each prompt should be on a new line and prefixed with 'PROMPT:'. Focus on scenarios that would make for visually compelling images and showcase the person in different contexts.

Here are examples of the style and quality of prompts I'm looking for:

PROMPT: "The person elegantly dancing barefoot on a moonlit beach, her intricate dress flowing gently in the ocean breeze."
PROMPT: "The person carefully arranging fresh flowers into a vibrant bouquet at a rustic farmer's market stall, sunlight accentuating her expressive features."
PROMPT: "The person standing confidently at the bow of an old wooden sailing ship, looking ahead determinedly as waves splash softly around her."
PROMPT: "The person in her ornate dress seated at an antique piano, immersed deeply in playing a romantic classical piece, illuminated by vintage candlelight."
PROMPT: "The person gently releasing colorful paper lanterns into a twilight sky during a lively cultural festival, her arms gracefully raised upward."

Format your response as {num_prompts} lines, each starting with 'PROMPT:' followed by the prompt text.""".format(
        num_prompts=num_prompts)

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Here's a description of the person: {person_description}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]}
        ],
        max_tokens=1000
    )

    generated_text = response.choices[0].message.content
    print(generated_text)

    prompts = []
    for line in generated_text.split("\n"):
        line = line.strip()
        if line.startswith("PROMPT:"):
            prompt_text = line[len("PROMPT:"):].strip()
            prompts.append(prompt_text)

    return prompts[:num_prompts]


async def download_image(url: str, save_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await response.read())


async def generate_pulidflux_images(prompts: List[str], image_path: str, output_dir: str):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{encoded_image}"

    output_image_paths = []  # List to store output image paths

    for i, prompt in enumerate(prompts):
        handler = await fal_client.submit_async(
            "fal-ai/flux-pulid",
            arguments={
                "prompt": prompt,
                "reference_image_url": data_uri,
                "image_size": {
                    "width": 1024,
                    "height": 1024
                },
                "num_inference_steps": 20,
                "guidance_scale": 4,
                "negative_prompt": "bad quality, worst quality, text, signature, watermark, extra limbs",
                "true_cfg": 1,
                "id_weight": 1,
                "enable_safety_checker": True,
                "max_sequence_length": "256"
            },
        )
        result = await handler.get()

        image_url = result['images'][0]['url']
        output_path = os.path.join(output_dir, f"pulid_{i}.jpg")
        async with aiohttp.ClientSession() as session:
            await download_image(image_url, output_path)

        output_image_paths.append(output_path)

    return output_image_paths


def generate_synthetic_images(
        image_path: str,
        description: str,
        num_images: int,
        output_dir: str
):
    """
    Generate prompts and create images based on the given parameters.

    Args:
        image_path: Path to the image of the person
        description: Description of the person including appearance, style, and personality
        num_images: Number of images to generate
        output_dir: Directory to save the generated images
    """
    prompts = generate_pulidflux_prompts(
        image_path=image_path,
        person_description=description,
        num_prompts=num_images
    )

    output_image_paths = asyncio.run(generate_pulidflux_images(prompts, image_path, output_dir))
    return output_image_paths
