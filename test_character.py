import argparse
import os
import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
from io import BytesIO

from helpers import find_character_lora, optimize_prompt
from inference.postprocess import FaceEnhancer
from inference.safety import SafetyChecker, create_blank_image

load_dotenv()


class LoRAImageGen:

    def __init__(self, face_enhance=False):
        self.is_prepared = False
        self.pipe = None
        self.safety_checker = SafetyChecker()
        self.face_enhance = face_enhance
        self.face_enhancer = None
        if self.face_enhance:
            self.face_enhancer = FaceEnhancer()

    def prepare(self):
        """Load the Flux model, move to GPU, and basic optimizations"""
        if self.is_prepared:
            return
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir=os.environ["HF_HOME"]
        ).to("cuda")

        # Basic optimizations without compilation
        self.pipe.transformer.fuse_qkv_projections()
        self.pipe.vae.fuse_qkv_projections()
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)
        print("✅ Flux loaded successfully")
        self.safety_checker.prepare()
        if self.face_enhance and self.face_enhancer:
            self.face_enhancer.prepare()
        self.is_prepared = True

    def do_inference(
            self,
            prompt: str,
            lora_path: str,
            lora_strength: float,
            width: int,
            height: int,
            num_inference_steps: int,
            batch_size: int,
    ) -> list:
        """
        Returns:
            list[bytes]: List of generated images as JPEG bytes
        """
        print(f"🎨 Generating image with prompt: {prompt}")

        if lora_path is None:
            raise ValueError("LoRA path was not provided. Please provide a LoRA path when generating the image.")

        self.pipe.load_lora_weights(lora_path)
        print(f"✅ LoRA loaded from: {lora_path}")

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                joint_attention_kwargs={"scale": lora_strength} if lora_path else None,
                num_images_per_prompt=batch_size
            )

        images = result.images

        # Convert each image to bytes
        image_bytes_list = []
        for image in images:
            byte_stream = BytesIO()
            image.save(byte_stream, format="JPEG", quality=95)
            image_bytes_list.append(byte_stream.getvalue())

        print("✅ Image generation complete")
        return image_bytes_list

    def generate(
            self,
            character_name,
            prompt,
            work_dir=None,
            lora_weight=0.73,
            test_dim=1024,
            do_optimize_prompt=True,
            output_filenames=None,
            batch_size=1,
            num_inference_steps=30,
            fix_outfit=True,
            face_enhance=False,
    ):
        """
        Main interface for generating images. Handles all setup and logic from test_character.py except the actual image generation.
        Args:
            character_name (str): Name of the character (used to find LoRA and work_dir)
            prompt (str): The prompt to use for generation
            work_dir (str, optional): Working directory (defaults to ./scratch/{character_name}/)
            lora_weight (float): LoRA strength
            test_dim (int): Image width/height
            optimize_prompt (bool): Whether to optimize the prompt
            output_filenames (list[str], optional): Filenames for output images
            batch_size (int): Number of images to generate
            num_inference_steps (int): Steps for generation
            fix_outfit (bool): Whether to use the --reference_image flag in prompt optimization
            face_enhance (bool): Whether to apply face enhancement
        Returns:
            list[str]: List of paths to the generated images
        """
        # Set default work_dir if not provided
        if work_dir is None:
            app_path = os.environ.get('APP_PATH', os.getcwd())
            work_dir = os.path.join(app_path, 'scratch', character_name)

        # Optimize the prompt if requested
        if do_optimize_prompt and prompt:
            prompt = optimize_prompt(prompt, character_name, fix_outfit, work_dir)

        # Find the LoRA model
        lora_path = find_character_lora(character_name, work_dir)
        print(f"Found LoRA model: {lora_path}")

        # Create output directory
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Set default output filenames if not provided
        if output_filenames is None:
            import time
            timestamp = int(time.time())
            output_filenames = [f"{character_name}_{timestamp}_{i}.jpg" for i in range(batch_size)]

        # Prepare the model if not already done
        self.prepare()

        # Generate the images (JPEG bytes)
        image_bytes_list = self.do_inference(
            prompt=prompt,
            lora_path=lora_path,
            lora_strength=lora_weight,
            width=test_dim,
            height=test_dim,
            num_inference_steps=num_inference_steps,
            batch_size=batch_size
        )

        # Save images to files
        generated_files = save_images_to_files(image_bytes_list, output_dir, output_filenames)

        if face_enhance and self.face_enhancer:
            print("🔍 Applying face enhancement...")
            # Create enhanced filenames by appending '_enhanced' before the file extension
            enhanced_filenames = []
            for fname in output_filenames:
                base, ext = os.path.splitext(fname)
                enhanced_filenames.append(f"{base}_enhanced{ext}")
            enhanced_filepaths = [os.path.join(output_dir, fname) for fname in enhanced_filenames]
            face_image = os.path.join(work_dir, "sheet/face_upscaled.png")
            generated_files = self.face_enhancer.process(
                face_image=face_image,
                input_image=generated_files,
                output_image=enhanced_filepaths,
                batch_size=batch_size,
                output_filenames=enhanced_filepaths
            )

        return generated_files

    def check_safety(self, generated_files, prompt, test_dim):
        """Check if the generated images are safe."""
        print("🛡️ Performing safety check...")
        violations = self.safety_checker.check_multiple(generated_files, prompt)

        # Replace unsafe images with blank placeholders
        for i, is_violation in enumerate(violations):
            if is_violation:
                create_blank_image(generated_files[i], width=test_dim, height=test_dim)
                print(f"🔒 Replaced unsafe image with blank placeholder: {generated_files[i]}")

        violation_count = sum(violations)
        if violation_count > 0:
            print(f"⚠️ Safety check replaced {violation_count} unsafe images with blank placeholders")
        return violations


def save_images_to_files(image_bytes_list, output_dir, output_filenames=None):
    """
    Save a list of image bytes to files in the specified directory.
    Returns:
        list[str]: List of file paths to saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for i, image_bytes in enumerate(image_bytes_list):
        if output_filenames and i < len(output_filenames):
            filename = output_filenames[i]
        else:
            filename = f"generated_image_{i + 1:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        saved_files.append(output_path)
        print(f"✅ Saved image to: {output_path}")
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Generate images using LoRAImageGen")
    parser.add_argument("--character_name", type=str,
                        help="Name of the character (used to find LoRA and work_dir)")
    parser.add_argument("--prompt", type=str,
                        help="The prompt to use for generation")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Working directory (defaults to ./scratch/{character_name}/)")
    parser.add_argument("--lora_weight", type=float, default=0.73, help="LoRA strength")
    parser.add_argument("--test_dim", type=int, default=1024, help="Image width/height")
    parser.add_argument("--do_optimize_prompt", action="store_true", default=True,
                        help="Whether to optimize the prompt")
    parser.add_argument("--no_optimize_prompt", dest="do_optimize_prompt", action="store_false",
                        help="Disable prompt optimization")
    parser.add_argument("--output_filenames", type=str, nargs="*", default=None,
                        help="Filenames for output images (space separated list)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Steps for generation")
    parser.add_argument("--fix_outfit", action="store_true", default=False,
                        help="Use the --reference_image flag in prompt optimization")
    parser.add_argument("--no_fix_outfit", dest="fix_outfit", action="store_false",
                        help="Disable outfit fixing in prompt optimization")
    parser.add_argument("--safety_check", action="store_true", default=True,
                        help="Run safety checks on generated images")
    parser.add_argument("--no_safety_check", dest="safety_check", action="store_false",
                        help="Disable safety checks on generated images")
    parser.add_argument("--face_enhance", action="store_true", default=False,
                        help="Enable face enhancement")
    parser.add_argument("--no_face_enhance", dest="face_enhance", action="store_false",
                        help="Disable face enhancement")

    parser.set_defaults(do_optimize_prompt=True, fix_outfit=False, safety_check=True, face_enhance=False)
    args = parser.parse_args()

    generator = LoRAImageGen(face_enhance=args.face_enhance)
    generated_files = generator.generate(
        character_name=args.character_name,
        prompt=args.prompt,
        work_dir=args.work_dir,
        lora_weight=args.lora_weight,
        test_dim=args.test_dim,
        do_optimize_prompt=args.do_optimize_prompt,
        output_filenames=args.output_filenames,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        fix_outfit=args.fix_outfit,
        face_enhance=args.face_enhance,
    )

    if args.safety_check:
        generator.check_safety(generated_files, args.prompt, args.test_dim)

    print(f"Generated {len(generated_files)} images:")
    for file_path in generated_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
