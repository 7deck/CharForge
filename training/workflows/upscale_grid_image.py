import os
import random
import sys
import torch
from PIL import Image
from typing import Sequence, Mapping, Any, Union

from install import COMFYUI_PATH
from .tools import LoadImageFromPath, save_comfy_images

"""
To avoid loading the models each time, we store them in a global variable.
"""
COMFYUI_UPSCALE_MODELS = None


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    if obj is None:
        raise ValueError(f"Cannot get value at index {index} from None object. Check if model loading succeeded.")
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    sys.path.append(COMFYUI_PATH)


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    ControlNetApplyAdvanced,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    ControlNetLoader,
    VAEEncode,
    CheckpointLoaderSimple,
    VAEDecodeTiled,
)


@torch.inference_mode()
def load_models(upscale_model: str = "4x-ClearRealityV1.pth"):
    """Load all the models needed for the Upscale pipeline and return them in a dictionary."""

    checkpointloadersimple = CheckpointLoaderSimple()
    checkpointloadersimple_48 = checkpointloadersimple.load_checkpoint(
        ckpt_name="flux1-dev-fp8.safetensors"
    )

    pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
    pulidfluxmodelloader_44 = pulidfluxmodelloader.load_model(
        pulid_file="pulid_flux_v0.9.1.safetensors"
    )

    pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
    pulidfluxevacliploader_45 = pulidfluxevacliploader.load_eva_clip()

    pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
    pulidfluxinsightfaceloader_46 = pulidfluxinsightfaceloader.load_insightface(
        provider="CUDA"
    )

    controlnetloader = ControlNetLoader()
    controlnetloader_49 = controlnetloader.load_controlnet(
        control_net_name="Flux_Dev_ControlNet_Union_Pro_ShakkerLabs.safetensors"
    )

    coremldetailerhookprovider = NODE_CLASS_MAPPINGS["CoreMLDetailerHookProvider"]()
    coremldetailerhookprovider_57 = coremldetailerhookprovider.doit(mode="768x768")

    ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
    ultralyticsdetectorprovider_66 = ultralyticsdetectorprovider.doit(
        model_name="bbox/face_yolov8m.pt"
    )

    upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
    upscalemodelloader_69 = upscalemodelloader.load_model(
        model_name=upscale_model
    )

    return {
        "checkpointloadersimple_48": checkpointloadersimple_48,
        "pulidfluxmodelloader_44": pulidfluxmodelloader_44,
        "pulidfluxevacliploader_45": pulidfluxevacliploader_45,
        "pulidfluxinsightfaceloader_46": pulidfluxinsightfaceloader_46,
        "controlnetloader_49": controlnetloader_49,
        "coremldetailerhookprovider_57": coremldetailerhookprovider_57,
        "ultralyticsdetectorprovider_66": ultralyticsdetectorprovider_66,
        "upscalemodelloader_69": upscalemodelloader_69
    }


def initialize_upscale_models(upscale_model: str = "4x-ClearRealityV1.pth"):
    """Initialize the models if they haven't been loaded yet."""
    global COMFYUI_UPSCALE_MODELS
    
    # Always check the actual state of the global variable, not just if it was previously initialized
    if COMFYUI_UPSCALE_MODELS is None:
        print(f"Initializing upscale models with model: {upscale_model}")
        import_custom_nodes()  # Ensure NODE_CLASS_MAPPINGS is initialized
        COMFYUI_UPSCALE_MODELS = load_models(upscale_model)


def cleanup_models():
    """Clean up models explicitly to avoid garbage collection issues."""
    global COMFYUI_UPSCALE_MODELS
    if COMFYUI_UPSCALE_MODELS is not None:
        for key in list(COMFYUI_UPSCALE_MODELS.keys()):
            COMFYUI_UPSCALE_MODELS[key] = None
        COMFYUI_UPSCALE_MODELS = None
        print("Upscale models cleaned up")


def apply_upscale_grid_image(
        face_image: str,
        input_image: str,
        output_image: str,
        positive_prompt: str,
        negative_prompt: str = "(text, logo, watermark, font, name, digits:1.2), (ugly, amateur, bad art:1.3), (worst quality, lowres, low detail, low contrast:1.4), (over/underexposed, over/undersaturated), (blurry, grainy), (cropped, out of frame, cut off, jpeg artifacts), (duplicate, glitch, merging, mutant, uncanny), (error, disfigured), (nsfw, naked)",
        upscale_model: str = "4x-ClearRealityV1.pth",
        output_prefix: str = "upscaled_detailed_upscaled_faces",
        upscale_factor: float = 1.0,
):
    # Initialize models if they haven't been loaded yet
    global COMFYUI_UPSCALE_MODELS
    initialize_upscale_models(upscale_model)

    if COMFYUI_UPSCALE_MODELS is None:
        raise ValueError("Models failed to initialize properly.")

    loadimage = LoadImageFromPath()
    loadimage_24 = loadimage.load_image(image_path=face_image)
    loadimage_40 = loadimage.load_image(image_path=input_image)

    with torch.inference_mode():
        # Get the loaded models from the global dictionary
        checkpointloadersimple_48 = COMFYUI_UPSCALE_MODELS["checkpointloadersimple_48"]
        pulidfluxmodelloader_44 = COMFYUI_UPSCALE_MODELS["pulidfluxmodelloader_44"]
        pulidfluxevacliploader_45 = COMFYUI_UPSCALE_MODELS["pulidfluxevacliploader_45"]
        pulidfluxinsightfaceloader_46 = COMFYUI_UPSCALE_MODELS["pulidfluxinsightfaceloader_46"]
        controlnetloader_49 = COMFYUI_UPSCALE_MODELS["controlnetloader_49"]
        coremldetailerhookprovider_57 = COMFYUI_UPSCALE_MODELS["coremldetailerhookprovider_57"]
        ultralyticsdetectorprovider_66 = COMFYUI_UPSCALE_MODELS["ultralyticsdetectorprovider_66"]
        upscalemodelloader_69 = COMFYUI_UPSCALE_MODELS["upscalemodelloader_69"]

        cliptextencode = CLIPTextEncode()
        cliptextencode_23 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(checkpointloadersimple_48, 1),
        )

        vaeencode = VAEEncode()
        vaeencode_35 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_40, 0),
            vae=get_value_at_index(checkpointloadersimple_48, 2),
        )

        int_literal = NODE_CLASS_MAPPINGS["Int Literal"]()
        int_literal_36 = int_literal.get_int(int=25)

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_39 = randomnoise.get_noise(noise_seed=random.randint(1, 2 ** 64))

        cliptextencode_42 = cliptextencode.encode(
            text=positive_prompt,
            clip=get_value_at_index(checkpointloadersimple_48, 1),
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_50 = ksamplerselect.get_sampler(sampler_name="deis")

        cfg_literal = NODE_CLASS_MAPPINGS["Cfg Literal"]()
        cfg_literal_67 = cfg_literal.get_float(float=1)

        int_literal_70 = int_literal.get_int(int=30)

        string_literal = NODE_CLASS_MAPPINGS["String Literal"]()
        string_literal_75 = string_literal.get_string(
            string=output_prefix
        )

        applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecodetiled = VAEDecodeTiled()
        colormatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
        tobasicpipe = NODE_CLASS_MAPPINGS["ToBasicPipe"]()
        frombasicpipe_v2 = NODE_CLASS_MAPPINGS["FromBasicPipe_v2"]()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()

        applypulidflux_31 = applypulidflux.apply_pulid_flux(
            weight=0.9,
            start_at=0.25,
            end_at=1,
            fusion="mean",
            fusion_weight_max=1,
            fusion_weight_min=0,
            train_step=1000,
            use_gray=True,
            model=get_value_at_index(checkpointloadersimple_48, 0),
            pulid_flux=get_value_at_index(pulidfluxmodelloader_44, 0),
            eva_clip=get_value_at_index(pulidfluxevacliploader_45, 0),
            face_analysis=get_value_at_index(pulidfluxinsightfaceloader_46, 0),
            image=get_value_at_index(loadimage_24, 0),
            unique_id=3692283962600669055,
        )

        setunioncontrolnettype_41 = setunioncontrolnettype.set_controlnet_type(
            type="tile", control_net=get_value_at_index(controlnetloader_49, 0)
        )

        controlnetapplyadvanced_37 = controlnetapplyadvanced.apply_controlnet(
            strength=0.8,
            start_percent=0,
            end_percent=0.8,
            positive=get_value_at_index(cliptextencode_42, 0),
            negative=get_value_at_index(cliptextencode_23, 0),
            control_net=get_value_at_index(setunioncontrolnettype_41, 0),
            image=get_value_at_index(loadimage_40, 0),
            vae=get_value_at_index(checkpointloadersimple_48, 2),
        )

        cfgguider_47 = cfgguider.get_guider(
            cfg=1,
            model=get_value_at_index(applypulidflux_31, 0),
            positive=get_value_at_index(controlnetapplyadvanced_37, 0),
            negative=get_value_at_index(controlnetapplyadvanced_37, 1),
        )

        basicscheduler_38 = basicscheduler.get_sigmas(
            scheduler="beta",
            steps=get_value_at_index(int_literal_36, 0),
            denoise=0.7000000000000001,
            model=get_value_at_index(applypulidflux_31, 0),
        )

        samplercustomadvanced_1 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_39, 0),
            guider=get_value_at_index(cfgguider_47, 0),
            sampler=get_value_at_index(ksamplerselect_50, 0),
            sigmas=get_value_at_index(basicscheduler_38, 0),
            latent_image=get_value_at_index(vaeencode_35, 0),
        )

        vaedecodetiled_28 = vaedecodetiled.decode(
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
            samples=get_value_at_index(samplercustomadvanced_1, 0),
            vae=get_value_at_index(checkpointloadersimple_48, 2),
        )

        colormatch_34 = colormatch.colormatch(
            method="mkl",
            strength=1,
            image_ref=get_value_at_index(loadimage_40, 0),
            image_target=get_value_at_index(vaedecodetiled_28, 0),
        )

        tobasicpipe_68 = tobasicpipe.doit(
            model=get_value_at_index(applypulidflux_31, 0),
            clip=get_value_at_index(checkpointloadersimple_48, 1),
            vae=get_value_at_index(checkpointloadersimple_48, 2),
            positive=get_value_at_index(controlnetapplyadvanced_37, 0),
            negative=get_value_at_index(controlnetapplyadvanced_37, 1),
        )

        frombasicpipe_v2_59 = frombasicpipe_v2.doit(
            basic_pipe=get_value_at_index(tobasicpipe_68, 0)
        )

        # Get image dimensions to adapt upscaling parameters
        img = Image.open(input_image)
        width, height = img.size

        # Determine optimal tile size and upscale factor based on image dimensions
        if width <= 1024 and height <= 768:
            # For smaller images, use smaller tiles and higher upscale factor
            tile_size = 512
            adaptive_upscale = max(1.5, upscale_factor)
        else:
            # For larger images, use larger tiles and more conservative upscale
            tile_size = 768
            adaptive_upscale = max(1.0, upscale_factor)

        # If image is very large, further adjust parameters
        if width >= 2000 or height >= 1500:
            tile_size = 1024
            adaptive_upscale = max(1.0, upscale_factor)

        # Final upscale with adaptive parameters
        ultimatesdupscale_63 = ultimatesdupscale.upscale(
            upscale_by=adaptive_upscale,
            seed=random.randint(1, 2 ** 64),
            steps=get_value_at_index(int_literal_70, 0),
            cfg=get_value_at_index(cfg_literal_67, 0),
            sampler_name="deis",
            scheduler="beta",
            denoise=0.25,
            mode_type="Linear",
            tile_width=tile_size,
            tile_height=tile_size,
            mask_blur=8,
            tile_padding=32,
            seam_fix_mode="None",
            seam_fix_denoise=1,
            seam_fix_width=64,
            seam_fix_mask_blur=8,
            seam_fix_padding=16,
            force_uniform_tiles=False,
            tiled_decode=False,
            image=get_value_at_index(colormatch_34, 0),
            model=get_value_at_index(frombasicpipe_v2_59, 1),
            positive=get_value_at_index(frombasicpipe_v2_59, 4),
            negative=get_value_at_index(frombasicpipe_v2_59, 5),
            vae=get_value_at_index(frombasicpipe_v2_59, 3),
            upscale_model=get_value_at_index(upscalemodelloader_69, 0),
        )

        easy_cleangpuused_73 = easy_cleangpuused.empty_cache(
            anything=get_value_at_index(ultimatesdupscale_63, 0),
            unique_id=12861305271031241446,
        )

        facedetailer_87 = facedetailer.doit(
            guide_size=512,
            guide_size_for=True,
            max_size=1024,
            seed=random.randint(1, 2 ** 64),
            steps=30,
            cfg=1,
            sampler_name="deis",
            scheduler="beta",
            denoise=0.25,
            feather=5,
            noise_mask=True,
            force_inpaint=True,
            bbox_threshold=0.08,
            bbox_dilation=20,
            bbox_crop_factor=3,
            sam_detection_hint="center-1",
            sam_dilation=0,
            sam_threshold=0.5,
            sam_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            sam_mask_hint_use_negative="False",
            drop_size=10,
            wildcard="",
            cycle=1,
            inpaint_model=False,
            noise_mask_feather=20,
            tiled_encode=False,
            tiled_decode=False,
            image=get_value_at_index(easy_cleangpuused_73, 0),
            model=get_value_at_index(frombasicpipe_v2_59, 1),
            clip=get_value_at_index(frombasicpipe_v2_59, 2),
            vae=get_value_at_index(frombasicpipe_v2_59, 3),
            positive=get_value_at_index(frombasicpipe_v2_59, 4),
            negative=get_value_at_index(frombasicpipe_v2_59, 5),
            bbox_detector=get_value_at_index(ultralyticsdetectorprovider_66, 0),
            detailer_hook=get_value_at_index(coremldetailerhookprovider_57, 0),
        )
        # we'll crop the faces and upscale separately 
        output_images = get_value_at_index(facedetailer_87, 0)
        save_comfy_images(output_images, [output_image])
