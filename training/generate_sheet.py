import gc
import json
import os
import time
import torch
from PIL import Image
from contextlib import contextmanager

from training.emotion_lighting import generate_emotion_lighting
from training.multiview import create_multiview_images, load_and_preprocess_image
from training.pulid_flux_images import generate_synthetic_images
from training.utilities import generate_caption, rectangle_to_square, resize_if_large


@contextmanager
def timing(component_name, log_file):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        with open(log_file, 'a') as f:
            f.write(f"{component_name}: {elapsed_time:.2f} seconds\n")


def cleanup_generation_models():
    """Clean up all models used in the character sheet generation process."""
    # Import cleanup functions directly
    from training.workflows.emotion_lighting import cleanup_models as cleanup_emotion
    from training.workflows.upscale_grid_image import cleanup_models as cleanup_pulid

    # Call each cleanup function with simple error handling
    try:
        cleanup_emotion()
        cleanup_pulid()
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("All models cleaned up after character sheet generation")


def gather_sheet_images(work_dir, pulid_image_paths=None):
    selected_images = [
        "upscaled_multiview_1.png",
        "upscaled_multiview_2.png",
        "upscaled_multiview_3.png",
        "upscaled_multiview_4.png",
        "upscaled_multiview_5.png",
        "face_upscaled.png",
        "original.png",
        "upscaled_lighting_0.png",
        "upscaled_lighting_1.png",
        "upscaled_lighting_2.png",
        "upscaled_lighting_3.png",
        "upscaled_emotions_1.png",
        "upscaled_emotions_3.png",  # only include smiling and surprised for now
    ]

    # If pulid_image_paths is provided, add their basenames to selected_images
    if pulid_image_paths:
        selected_images += [os.path.basename(path) for path in pulid_image_paths]

    # Create trash directory for non-selected images
    trash_dir = os.path.join(work_dir, "trash")
    os.makedirs(trash_dir, exist_ok=True)

    # Get all image files in the work directory
    all_files = os.listdir(work_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']

    # Move non-selected images to trash directory
    for file in all_files:
        file_path = os.path.join(work_dir, file)
        # Skip directories and non-image files
        if os.path.isdir(file_path) or not any(file.lower().endswith(ext) for ext in image_extensions):
            continue

        # Move non-selected images to trash
        if file not in selected_images:
            os.rename(file_path, os.path.join(trash_dir, file))

    # Build the list of selected image paths (deduplicate)
    sheet_image_paths = []
    seen = set()
    for img in selected_images:
        path = os.path.join(work_dir, img)
        if img not in seen and os.path.exists(path):
            sheet_image_paths.append(path)
            seen.add(img)

    return sheet_image_paths


def create_image_info_json(work_dir):
    """Create a JSON file with descriptions for each image."""
    image_info = {
        "upscaled_multiview_1.png": {
            "description": "photorealistic, Three-quarters view from the left side of the character, showing facial features and profile with a slight angle."
        },
        "upscaled_multiview_2.png": {
            "description": "photorealistic, Direct left profile view, displaying the complete side silhouette of the character's face and head."
        },
        "upscaled_multiview_3.png": {
            "description": "photorealistic, Rear view of the character, showing the back of the head and hair details."
        },
        "upscaled_multiview_4.png": {
            "description": "photorealistic, Direct right profile view, displaying the complete side silhouette of the character's face and head."
        },
        "upscaled_multiview_5.png": {
            "description": "photorealistic, Three-quarters view from the right side of the character, showing facial features and profile with a slight angle."
        },
        "face_upscaled.png": {
            "description": "photorealistic, High-resolution frontal portrait of the character "
        },
        "original.png": {
            "description": "photorealistic, Original source image used as the foundation for character generation."
        },
        "upscaled_lighting_0.png": {
            "description": "photorealistic, Character in dramatic natural lighting resembling overcast weather, with soft diffused light typical of cloudy conditions."
        },
        "upscaled_lighting_1.png": {
            "description": "photorealistic, Character illuminated by warm sunset lighting with golden hues, creating soft shadows and warm highlights across facial features."
        },
        "upscaled_lighting_2.png": {
            "description": "photorealistic, Character in vibrant nightclub lighting with contrasting red and blue/purple color gels creating a dramatic atmospheric effect."
        },
        "upscaled_lighting_3.png": {
            "description": "photorealistic, Character in alternative lighting scenario showing different mood and atmosphere."
        },
        "upscaled_emotions_1.png": {
            "description": "photorealistic, Character with eyes closed, showing a peaceful or contemplative expression."
        },
        "upscaled_emotions_3.png": {
            "description": "photorealistic, Character laughing heartily, displaying joy with an open mouth smile and animated facial features."
        },
        "upscaled_emotions_2.png": {
            "description": "photorealistic, Character with a cringing expression while winking, showing playful discomfort or mild embarrassment."
        },
        "upscaled_emotions_4.png": {
            "description": "photorealistic, Character with a surprised expression, featuring widened eyes and raised eyebrows showing astonishment."
        }
    }

    # Save the JSON file to the work directory
    json_path = os.path.join(work_dir, "image_info.json")
    with open(json_path, 'w') as f:
        json.dump(image_info, f, indent=4)

    return json_path


def generate_char_sheet(name, input_image, work_dir=None, log_file=None, pulidflux_images=0):
    if work_dir is None:
        work_dir = f'./scratch/{name}/'
    os.makedirs(work_dir, exist_ok=True)

    # Use provided log file or create one in the parent directory
    if log_file is None:
        parent_dir = os.path.dirname(work_dir.rstrip('/'))
        log_file = os.path.join(parent_dir, "timing.log")
        # Clear previous log if exists
        if os.path.exists(log_file):
            open(log_file, 'w').close()

    with timing("Total process", log_file):
        # Copy the input image to the work directory as original.png
        with timing("preprocessing", log_file):
            original_path = os.path.join(work_dir, "original.png")
            original_image = Image.open(input_image)
            original_image.save(original_path, format="PNG")

            # Convert to square and save as input.png
            input_image = rectangle_to_square(original_image)

            # Resize if the square image is too large (>1536 pixels per side)
            input_image = resize_if_large(input_image, max_size=1536)

            input_path = os.path.join(work_dir, "input.png")
            input_image.save(input_path, format="PNG")

            # Load the squared image for processing
            image = load_and_preprocess_image(input_path)
            caption = generate_caption(image)
            print(f"Caption:\n{caption}")

        with timing("Multi-view", log_file):
            create_multiview_images(input_path, work_dir, caption)
            print("Multi-view images generated")

        face_path = os.path.join(work_dir, 'face_upscaled.png')

        with timing("Emotion and lighting", log_file):
            generate_emotion_lighting(face_path, work_dir, caption)
            print("Emotion and lighting generated")

        pulid_image_paths = None
        if pulidflux_images > 0:
            with timing("Pulid-Flux", log_file):
                pulid_image_paths = generate_synthetic_images(face_path, caption, pulidflux_images, work_dir)
                print("Pulid-Flux images generated")

        sheet_images = gather_sheet_images(work_dir, pulid_image_paths)

        # Create image_info.json with detailed descriptions
        with timing("Creating image info", log_file):
            json_path = create_image_info_json(work_dir)
            print(f"Image info created at: {json_path}")

        print("Sheet images: ", sheet_images)
        print("Finished!\n")
        print(f"Timing data saved to {log_file}")

    try:
        cleanup_generation_models()
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")
    return sheet_images
