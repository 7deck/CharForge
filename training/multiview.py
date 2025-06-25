import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from training.utilities import upscale_image, crop_face, get_upscale_factor

# Get the absolute path to the MV_Adapter directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)  # Get parent directory of training
mv_adapter_path = os.path.join(repo_root, 'MV_Adapter')

# Add MV_Adapter to sys.path
sys.path.append(mv_adapter_path)

# Import from MV_Adapter/scripts
from scripts.inference_i2mv_sdxl import prepare_pipeline, run_pipeline, remove_bg

from training.workflows.upscale_grid_image import apply_upscale_grid_image

from install import COMFYUI_PATH


def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
    image = Image.open(image_path).convert("RGB")
    processed_image = image.resize(target_size, Image.LANCZOS)
    return processed_image


def generate_multiview_images(
        image_path,
        output_dir,
        text,
        base_model="RunDiffusion/Juggernaut-XL-v9",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler="ddpm",
        num_views=6,
        device="cuda",
        dtype=torch.float16,
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=-1,
        lora_scale=1.0,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast, mutant, broken limbs",
        azimuth_deg=[0, 45, 90, 180, 270, 315],
        do_remove_bg=True
):
    # Load the image
    image = load_and_preprocess_image(image_path)

    # Optionally remove background
    remove_bg_fn = None
    if do_remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"]
        ).to(device)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, device)

    # Prepare the pipeline using the function from inference_i2mv_sdxl.py
    pipe = prepare_pipeline(
        base_model=base_model,
        vae_model=vae_model,
        unet_model=unet_model,
        lora_model=lora_model,
        adapter_path=adapter_path,
        scheduler=scheduler,
        num_views=num_views,
        device=device,
        dtype=dtype,
        cache_dir=os.environ["HF_HOME"]
    )

    # Run the pipeline using the function from inference_i2mv_sdxl.py
    images, reference_image = run_pipeline(
        pipe,
        num_views=num_views,
        text=text,
        image=image,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        lora_scale=lora_scale,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        device=device,
        remove_bg_fn=remove_bg_fn,
        azimuth_deg=azimuth_deg
    )

    os.makedirs(os.path.join(output_dir, "multiview"), exist_ok=True)
    for idx, img in enumerate(images):
        img.save(os.path.join(output_dir, "multiview", f"multiview_{idx}.png"))

    # Save the reference image with removed background
    reference_image.save(os.path.join(output_dir, "cleaned_reference.png"))


def create_multi_view_grid(image_paths, output_dir, output_filename="multiview_grid.png"):
    """
    Creates a grid from 6 individual multi-view images and saves it.
    All images are resized to the smallest square dimension to maintain consistency.
    
    Args:
        image_paths: List of 6 image paths to create the grid from
        output_dir: Directory to save the grid image
        output_filename: Name of the output grid image file
        
    Returns:
        Path to the saved grid image
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(image_paths) != 6:
        raise ValueError(f"Expected 6 image paths, got {len(image_paths)}")

    # Load images and find smallest dimension
    images = []
    min_dimension = float('inf')
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        min_dimension = min(min_dimension, min(img.size))
        images.append(img)

    # Resize all images to the smallest square dimension
    resized_images = []
    for img in images:
        resized_img = img.resize((min_dimension, min_dimension), Image.LANCZOS)
        resized_images.append(resized_img)

    # Create grid (2 rows x 3 columns)
    rows, cols = 2, 3
    grid_width = cols * min_dimension
    grid_height = rows * min_dimension
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Place images in the grid
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        grid_image.paste(img, (col * min_dimension, row * min_dimension))

    grid_path = os.path.join(output_dir, output_filename)
    grid_image.save(grid_path)
    print(f"Created and saved multi-view grid: {grid_path}")
    return grid_path


def split_grid_into_images(grid_image_path, output_paths):
    """
    Splits a grid image back into individual images and saves them.
    Args:
        grid_image_path: str - Path to the grid image to split
        output_paths: list[str] - List of 6 output paths for the individual images
    """
    if len(output_paths) != 6:
        raise ValueError(f"Expected 6 output paths, got {len(output_paths)}")

    grid_image = Image.open(grid_image_path).convert("RGB")

    rows, cols = 2, 3

    grid_width, grid_height = grid_image.size
    img_width = grid_width // cols
    img_height = grid_height // rows

    # Split and save individual images
    for row in range(rows):
        for col in range(cols):
            # Calculate image index
            img_idx = row * cols + col

            # Extract individual image from grid
            left = col * img_width
            upper = row * img_height
            right = left + img_width
            lower = upper + img_height

            individual_img = grid_image.crop((left, upper, right, lower))

            # Save to the specified output path
            individual_img.save(output_paths[img_idx])

    print(f"Split grid into individual images")


def create_multiview_images(image_path, output_dir, caption):
    generate_multiview_images(
        image_path=image_path,
        output_dir=output_dir,
        text=caption,
        base_model=f"{COMFYUI_PATH}/models/checkpoints/juggernaut-xl.safetensors",  # civitai
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler="ddpm",
        num_views=6,
        device="cuda",
        dtype=torch.float16,
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=-1,
        lora_scale=1.0,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, disfigured, noisy, blurry, low resolution, low contrast",
        azimuth_deg=[0, 45, 90, 180, 270, 315],
        do_remove_bg=True
    )

    # Get list of multiview image paths
    multiview_dir = os.path.join(output_dir, "multiview")
    multiview_paths = [
        os.path.join(multiview_dir, f"multiview_{i}.png")
        for i in range(6)
    ]
    create_multi_view_grid(multiview_paths, output_dir, "multiview_grid.png")

    enhanced_caption = caption + " a character sheet showing multiple views of a character in front of a grey background"

    # Upscale the reference face image (using synchronous function)
    cleaned_reference = os.path.join(output_dir, "cleaned_reference.png")
    face_reference_path = crop_face(cleaned_reference, output_dir, "face_reference.png")
    # Fallback: if crop_face fails, use the cleaned_reference image directly
    if not face_reference_path:
        print("Warning: crop_face failed, using cleaned_reference as fallback for face_reference_path")
        face_reference_path = cleaned_reference

    face_upscaled_factor = get_upscale_factor(face_reference_path, target_size=768)
    print(f"Upscaled factor: {face_upscaled_factor}")
    face_upscaled_path = os.path.join(output_dir, "face_upscaled.png")
    upscale_image(face_reference_path, output_path=face_upscaled_path, scale=face_upscaled_factor)

    # Verify the file exists
    if not os.path.exists(face_upscaled_path):
        raise FileNotFoundError(f"Upscaled face image not found at: {face_upscaled_path}")

    apply_upscale_grid_image(
        face_image=face_upscaled_path,
        input_image=os.path.join(output_dir, "multiview_grid.png"),
        output_image=os.path.join(output_dir, "upscaled_multiview_grid.png"),
        positive_prompt=enhanced_caption,
        upscale_factor=get_upscale_factor(input_size=(768, 768), target_size=1024)
    )

    # Create list of output paths for the split images
    upscaled_multiview_paths = [
        os.path.join(output_dir, f"upscaled_multiview_{i}.png")
        for i in range(6)
    ]
    split_grid_into_images(os.path.join(output_dir, "upscaled_multiview_grid.png"), upscaled_multiview_paths)
