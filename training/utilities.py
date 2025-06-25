import base64
import cv2
import fal_client
import io
import numpy as np
import os
import requests
import sys
from PIL import Image
from together import Together

# Get the absolute path to the ComfyUI_AutoCropFaces directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)  # Get parent directory of training
autocrop_path = os.path.join(repo_root, 'ComfyUI_AutoCropFaces')

sys.path.append(autocrop_path)

from Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace


def get_system_prompt():
    return """You are a concise image captioning assistant.

Your task is to provide a brief caption for the given image (around 3 sentences).
Focus on describing:
- The subject (person, character, etc.)
- Physical appearance and attributes
- Clothing and accessories
- Facial features and expression

IMPORTANT: Completely IGNORE the background - do not mention it at all.
Remain objective – do not reference known characters, franchises, or people, even if recognizable.
Avoid making assumptions about things that aren't visible in the image.
"""


def image_to_base64(image):
    """Convert a PIL image to base64 encoded string."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_together_client():
    """Initialize and return the Together API client."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables!")
    return Together(api_key=api_key)


def generate_caption(image):
    """Generate a detailed caption for the image"""
    img_str = image_to_base64(image)
    client = get_together_client()
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                {"type": "text", "text": "Please provide a detailed caption for this image."}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=messages
    )
    caption = response.choices[0].message.content.strip()
    return caption


def upscale_image(image_path, output_path=None, scale=1.5, model="RealESRGAN_x4plus", face_enhance=True,
                  output_format="png"):
    """
    Upscale an image using fal.ai's ESRGAN service (synchronous version)
    
    Args:
        image_path: Path to the image file
        output_path: Path where the upscaled image will be saved (default: None, will return PIL Image)
        scale: Scaling factor (default: 1.5)
        model: Model to use (default: "RealESRGAN_x4plus")
        face_enhance: Whether to enhance faces (default: True)
        output_format: Output image format (default: "png")
    
    Returns:
        If output_path is provided: Path to the saved image
        If output_path is None: PIL Image of the upscaled result
    """
    # Load image from path
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_str = base64.b64encode(img_data).decode("utf-8")

    img_url = f"data:image/png;base64,{img_str}"

    # Submit to fal.ai API synchronously
    result = fal_client.subscribe(
        "fal-ai/esrgan",
        arguments={
            "image_url": img_url,
            "scale": scale,
            "model": model,
            "output_format": output_format,
            "face": face_enhance
        },
        with_logs=False
    )

    # Parse the result
    if "image" in result:
        # According to fal.ai docs, the result has format: {"image": {"url": "...", "content_type": "...", etc.}}
        if isinstance(result["image"], dict) and "url" in result["image"]:
            # Download image from the provided URL
            response = requests.get(result["image"]["url"])
            image_data = response.content
        else:
            # Fallback for any other format
            raise ValueError(f"Unexpected image format in result: {result['image']}")

        upscaled_image = Image.open(io.BytesIO(image_data))

        # Save to output path if provided
        if output_path:
            upscaled_image.save(output_path)
            return output_path

        return upscaled_image
    else:
        raise ValueError("No image was returned from the upscale service")


def rectangle_to_square(image, background_color=(128, 128, 128)):
    """
    Convert a rectangular image to a square by adding padding with a background color.
    If the image is already square, returns the original image.
    
    Args:
        image: PIL Image or numpy array
        background_color: RGB tuple for the background color (default: gray)
    
    Returns:
        A square PIL Image with the original image centered
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Get the original dimensions
    width, height = image.size

    # If image is already square, return the original
    if width == height:
        return image

    # Calculate the target size (use the larger dimension)
    target_size = max(width, height)

    # Create a new square image with the background color
    result = Image.new('RGB', (target_size, target_size), background_color)

    # Calculate position to paste the original image (centered)
    x_offset = (target_size - width) // 2
    y_offset = (target_size - height) // 2

    # Paste the original image onto the square background
    result.paste(image, (x_offset, y_offset))

    return result


def crop_face(image_path, output_dir, output_name, scale_factor=4.0):
    image = Image.open(image_path).convert("RGB")

    img_raw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_raw = img_raw.astype(np.float32)

    rf = Pytorch_RetinaFace(
        cfg='mobile0.25',
        pretrained_path='./weights/mobilenet0.25_Final.pth',
        confidence_threshold=0.02,
        nms_threshold=0.4,
        vis_thres=0.6
    )

    dets = rf.detect_faces(img_raw)
    print("Dets: ", dets)

    # Instead of asserting, handle multiple faces gracefully
    if len(dets) == 0:
        print("No faces detected!")
        return False

    # If multiple faces detected, use the one with highest confidence
    if len(dets) > 1:
        print(f"Warning: {len(dets)} faces detected, using the one with highest confidence")
        # Assuming dets is a list of [bbox, landmark, score] and we want to sort by score
        dets = sorted(dets, key=lambda x: x[2], reverse=True)  # Sort by confidence score
        # Just keep the highest confidence detection
        dets = [dets[0]]

    # Pass the scale_factor to center_and_crop_rescale for adjustable crop size
    try:
        # Unpack the tuple correctly - the function returns (cropped_imgs, bbox_infos)
        cropped_imgs, bbox_infos = rf.center_and_crop_rescale(img_raw, dets, shift_factor=0.45,
                                                              scale_factor=scale_factor)

        # Assuming cropped_imgs is a list or tuple (cropped_imgs, bbox_infos)
        for i, (cropped_img, bbox_info) in enumerate(zip(cropped_imgs, bbox_infos)):
            # Convert BGR to RGB
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            cropped_pil = Image.fromarray(cropped_img_rgb.astype(np.uint8))

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Save to output path
            out_name = output_name if isinstance(output_name, str) else f"face_{i}.png"
            output_path = os.path.join(output_dir, out_name)
            cropped_pil.save(output_path)

            print(f"Face cropped and saved to {output_path}")
            return output_path
    except Exception as e:
        print(f"Error cropping face: {e}")
        return False


def get_upscale_factor(image: str = None, input_size=None, target_size=768):
    """Calculate the scale factor needed to upscale an image to approximately targetxtarget"""

    if input_size is None:
        image = Image.open(image).convert("RGB")
        width, height = image.size
    else:
        width, height = input_size

    width_scale = target_size / width
    height_scale = target_size / height

    scale_factor = max(width_scale, height_scale)
    return scale_factor


def resize_if_large(image, max_size=1536):
    """
    Resize a square image if its dimensions exceed the specified maximum size.
    
    Args:
        image: PIL Image object (assumed to be square)
        max_size: Maximum allowed dimension (width/height) in pixels
        
    Returns:
        PIL Image: Resized image if needed, or original image if already small enough
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    width, height = image.size

    # Verify image is square (should be after rectangle_to_square)
    if width != height:
        print(f"Warning: Expected square image but got {width}x{height}")

    # If image is already small enough, return the original
    if width <= max_size:
        return image

    # Resize the square image
    resized_image = image.resize((max_size, max_size), Image.LANCZOS)

    return resized_image
