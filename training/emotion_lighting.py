import os

from training.multiview import create_multi_view_grid, split_grid_into_images
from training.utilities import get_upscale_factor
from training.workflows.emotion_lighting import apply_emotion_lighting
from training.workflows.upscale_grid_image import apply_upscale_grid_image


def generate_emotion_lighting(input_image, output_dir, caption):
    apply_emotion_lighting(
        input_image=input_image,
        output_dir=output_dir,
    )

    image_names = [f"lighting_{i}.png" for i in range(4)]
    image_names.extend([f"emotions_{1}.png", f"emotions_{3}.png"])

    image_paths = [os.path.join(output_dir, name) for name in image_names]

    create_multi_view_grid(image_paths, output_dir, "emotions_grid.png")

    enhanced_caption = caption + " a character sheet showing various emotions and lighting conditions"

    face_upscaled_path = os.path.join(output_dir, "face_upscaled.png")

    # Verify the file exists
    if not os.path.exists(face_upscaled_path):
        raise FileNotFoundError(f"Upscaled face image not found at: {face_upscaled_path}")

    apply_upscale_grid_image(
        face_image=face_upscaled_path,
        input_image=os.path.join(output_dir, "emotions_grid.png"),
        output_image=os.path.join(output_dir, "upscaled_emotions_grid.png"),
        positive_prompt=enhanced_caption,
        upscale_factor=get_upscale_factor(input_size=(768, 768), target_size=1024)
    )

    upscaled_emotions_lighting_paths = [os.path.join(output_dir, f"upscaled_{name}") for name in image_names]

    split_grid_into_images(os.path.join(output_dir, "upscaled_emotions_grid.png"), upscaled_emotions_lighting_paths)
