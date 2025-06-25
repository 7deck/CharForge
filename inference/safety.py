import os
import torch
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from transformers import AutoModelForImageClassification, ViTImageProcessor


class SafetyChecker:
    """
    Safety checker for generated images using NSFW detection model.
    """

    def __init__(self):
        self.is_prepared = False
        self.model = None
        self.processor = None

    def prepare(self):
        """
        Load the NSFW detection model and processor.
        """
        if self.is_prepared:
            return

        print("🛡️ Loading safety checker model...")

        try:
            # Load the NSFW detection model and processor
            self.model = AutoModelForImageClassification.from_pretrained(
                "Falconsai/nsfw_image_detection",
                torch_dtype=torch.float32,
                cache_dir=os.environ.get("HF_HOME")
            )
            self.processor = ViTImageProcessor.from_pretrained(
                'Falconsai/nsfw_image_detection',
                cache_dir=os.environ.get("HF_HOME")
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

            self.is_prepared = True
            print("✅ Safety checker model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load safety checker model: {e}")
            raise

    def check(self, image_input, prompt=None):
        """
        Check if an image contains NSFW content.
        
        Args:
            image_input: Can be either:
                - PIL Image object
                - bytes (JPEG image data)
                - str (path to image file)
            prompt (str, optional): The prompt used to generate the image (for logging)
        
        Returns:
            bool: True if there is a violation (NSFW content detected), False if safe
        """
        if not self.is_prepared:
            self.prepare()

        try:
            if isinstance(image_input, bytes):
                image = Image.open(BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits

                predicted_label_idx = logits.argmax(-1).item()
                predicted_label = self.model.config.id2label[predicted_label_idx]

                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities.max().item()

                is_violation = predicted_label.lower() == "nsfw"

                status = "🚨 VIOLATION" if is_violation else "✅ SAFE"
                print(f"{status} - Classification: {predicted_label} (confidence: {confidence:.3f})")
                return is_violation

        except Exception as e:
            print(f"❌ Error during safety check: {e}, assuming safe")
            return False

    def check_multiple(self, image_inputs, prompt=None):
        """
        Returns:
            list[bool]: List of violation flags for each image
        """
        results = []
        for i, image_input in enumerate(image_inputs):
            print(f"🔍 Checking image {i + 1}/{len(image_inputs)}")
            violation = self.check(image_input, prompt)
            results.append(violation)

        violation_count = sum(results)
        print(f"📊 Safety check complete: {violation_count}/{len(image_inputs)} images flagged")

        return results


def create_blank_image(file_path, width=1024, height=1024):
    """Create a blank gray image as a placeholder for unsafe content."""
    blank_image = Image.new('RGB', (width, height), color='lightgray')
    draw = ImageDraw.Draw(blank_image)

    text = "Content Filtered\nfor Safety"
    try:
        font_size = min(width, height) // 10
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill='darkgray', font=font, align='center')

    blank_image.save(file_path, format="JPEG", quality=95)
    print(f"📝 Created blank placeholder image: {file_path}")
