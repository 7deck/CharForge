import numpy as np
import os
import torch
from PIL import Image, ImageOps, ImageSequence


class LoadImageFromPath:
    """Custom image loader that can load images from any path."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img = Image.open(image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)


def save_comfy_images(images, output_dirs):
    # images is a PyTorch tensor with shape [batch_size, height, width, channels]
    for idx, image in enumerate(images):
        numpy_image = 255. * image.cpu().numpy()
        numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(numpy_image)
        pil_image.save(output_dirs[idx])
    return output_dirs


def load_lora_from_path(model, clip, lora_path, strength_model=1.0, strength_clip=1.0):
    """
    Load a LoRA model from any path.
    
    Args:
        model: The diffusion model the LoRA will be applied to
        clip: The CLIP model the LoRA will be applied to
        lora_path: Full path to the LoRA file (.safetensors, .pt, etc.)
        strength_model: Strength of the LoRA model effect (float)
        strength_clip: Strength of the LoRA clip effect (float)
    
    Returns:
        tuple: (modified_model, modified_clip)
    """
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)

    # Import required modules from ComfyUI
    from comfy.utils import load_torch_file
    from comfy.sd import load_lora_for_models

    # Check if the LoRA file exists
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA file not found at path: {lora_path}")

    # Load the LoRA file
    lora = load_torch_file(lora_path, safe_load=True)

    # Apply the LoRA to the models
    model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    return (model_lora, clip_lora)
