import os
from typing import Union, List

from inference.workflows.face_enhance import initialize_models, face_enhance


class FaceEnhancer:
    """
    Wrapper class for face enhancement using the face_enhance pipeline.
    Use prepare() to load models and process() to run enhancement.
    """

    def __init__(self):
        self.prepared = False

    def prepare(self):
        """
        Loads and initializes all required models and resources for face enhancement.
        Call this before process().
        """
        initialize_models()
        self.prepared = True

    def process(
            self,
            face_image: str,
            input_image: Union[str, List[str]],
            output_image: Union[str, List[str]],
            positive_prompt: str = "",
            id_weight: float = 0.75,
            batch_size: int = 1,
            output_filenames: List[str] = None,
    ):
        """
        Runs the face enhancement process. Call prepare() before this.
        Args:
            face_image: Path to face image
            input_image: Path or list of paths to input images
            output_image: Path or list of output image paths
            positive_prompt: one prompt (no list)
            id_weight: Weight for identity
            batch_size: Number of images in batch
            output_filenames: Optional list of output filenames
        Returns:
            list[str]: List of filenames of the enhanced images
        """
        if not self.prepared:
            raise RuntimeError("Call prepare() before process().")
        return face_enhance(
            face_image=face_image,
            input_image=input_image,
            output_image=output_image,
            positive_prompt=positive_prompt,
            id_weight=id_weight,
            batch_size=batch_size,
            output_filenames=output_filenames,
        )
