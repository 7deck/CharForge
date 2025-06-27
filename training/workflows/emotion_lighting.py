import atexit
import os
import random
import sys
import torch
from typing import Sequence, Mapping, Any, Union

from install import COMFYUI_PATH
from .tools import LoadImageFromPath, save_comfy_images

"""
To avoid loading the models each time, we store them in a global variable.
"""
COMFY_EMOTION_LIGHTING_MODELS = None


# Function to clean up models
def cleanup_models():
    """Clean up models explicitly to avoid garbage collection issues."""
    global COMFY_EMOTION_LIGHTING_MODELS
    if COMFY_EMOTION_LIGHTING_MODELS is not None:
        for key in list(COMFY_EMOTION_LIGHTING_MODELS.keys()):
            COMFY_EMOTION_LIGHTING_MODELS[key] = None
        COMFY_EMOTION_LIGHTING_MODELS = None
        print("Models cleaned up")


# Register cleanup function to be called at exit
atexit.register(cleanup_models)


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
    VAEDecode,
    CheckpointLoaderSimple,
    KSampler,
    NODE_CLASS_MAPPINGS,
    VAEEncode,
    CLIPSetLastLayer,
    CLIPTextEncode,
    EmptyLatentImage,
)


@torch.inference_mode()
def load_models():
    """Load all the models needed for the emotion lighting pipeline and return them in a dictionary."""
    from . import upscale_grid_image
    
    # Ensure upscale models are loaded before accessing them
    upscale_grid_image.initialize_upscale_models()
    
    # Check if the upscale models were successfully initialized
    if upscale_grid_image.COMFYUI_UPSCALE_MODELS is None:
        raise ValueError("Failed to initialize upscale models. COMFYUI_UPSCALE_MODELS is None.")

    checkpointloadersimple = CheckpointLoaderSimple()
    flux_model = upscale_grid_image.COMFYUI_UPSCALE_MODELS["checkpointloadersimple_48"]

    checkpointloadersimple_726 = checkpointloadersimple.load_checkpoint(
        ckpt_name="photon.safetensors"
    )

    loadandapplyiclightunet = NODE_CLASS_MAPPINGS["LoadAndApplyICLightUnet"]()
    loadandapplyiclightunet_723 = loadandapplyiclightunet.load(
        model_path="iclight_sd15_fbc.safetensors",
        model=get_value_at_index(checkpointloadersimple_726, 0),
    )

    modelpassthrough = NODE_CLASS_MAPPINGS["ModelPassThrough"]()
    modelpassthrough_1038 = modelpassthrough.passthrough(
        model=get_value_at_index(flux_model, 0)
    )

    return {
        "flux_model": flux_model,
        "photon_model": checkpointloadersimple_726,
        "loadandapplyiclightunet_723": loadandapplyiclightunet_723,
        "modelpassthrough_1038": modelpassthrough_1038
    }


def initialize_emotion_lighting_models():
    """Initialize the models if they haven't been loaded yet."""
    global COMFY_EMOTION_LIGHTING_MODELS
    if COMFY_EMOTION_LIGHTING_MODELS is None:
        import_custom_nodes()  # Ensure NODE_CLASS_MAPPINGS is initialized
        COMFY_EMOTION_LIGHTING_MODELS = load_models()


def apply_emotion_lighting(input_image: str, output_dir: str, positive_prompt: str = "portrait photography"):
    # Initialize models if they haven't been loaded yet
    global COMFY_EMOTION_LIGHTING_MODELS
    initialize_emotion_lighting_models()

    if COMFY_EMOTION_LIGHTING_MODELS is None:
        raise ValueError("Models failed to initialize properly.")

    loadimage = LoadImageFromPath()
    loadimage_1395 = loadimage.load_image(image_path=input_image)

    with torch.inference_mode():
        # Get the loaded models from the global dictionary
        checkpointloadersimple_726 = COMFY_EMOTION_LIGHTING_MODELS["photon_model"]
        checkpointloadersimple_1401 = COMFY_EMOTION_LIGHTING_MODELS["flux_model"]
        loadandapplyiclightunet_723 = COMFY_EMOTION_LIGHTING_MODELS["loadandapplyiclightunet_723"]
        modelpassthrough_1038 = COMFY_EMOTION_LIGHTING_MODELS["modelpassthrough_1038"]

        string_literal = NODE_CLASS_MAPPINGS["String Literal"]()
        string_literal_426 = string_literal.get_string(
            string="Yellowstone National Park beneath looming thunderclouds, rugged rock formations, tall pines lining a gentle river, reflections shimmering in the overcast light."
        )

        string_literal_1408 = string_literal.get_string(
            string=positive_prompt
        )

        joinstrings = NODE_CLASS_MAPPINGS["JoinStrings"]()
        joinstrings_1001 = joinstrings.joinstring(
            string1=get_value_at_index(string_literal_1408, 0),
            string2=get_value_at_index(string_literal_426, 0),
            delimiter=",",
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_982 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2,
            clip=get_value_at_index(checkpointloadersimple_1401, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_431 = cliptextencode.encode(
            text=get_value_at_index(joinstrings_1001, 0),
            clip=get_value_at_index(clipsetlastlayer_982, 0),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_432 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        string_literal_436 = string_literal.get_string(
            string="A vivid sunset over New York City, sky ablaze in gold and amber as the sun sinks behind the skyline. Iconic skyscrapers in silhouette above dense rooftops and gridded streets. Warm light casts deep shadows."
        )

        joinstrings_1002 = joinstrings.joinstring(
            string1=get_value_at_index(string_literal_1408, 0),
            string2=get_value_at_index(string_literal_436, 0),
            delimiter=",",
        )

        cliptextencode_439 = cliptextencode.encode(
            text=get_value_at_index(joinstrings_1002, 0),
            clip=get_value_at_index(clipsetlastlayer_982, 0),
        )

        string_literal_443 = string_literal.get_string(
            string="A vibrant nightclub filled with energy — red gel lights wash over the left side, while blue and purple tones light the right. A tightly packed crowd in the foreground, silhouetted with raised hands, dances and cheers beneath the pulsing lights."
        )

        joinstrings_1003 = joinstrings.joinstring(
            string1=get_value_at_index(string_literal_1408, 0),
            string2=get_value_at_index(string_literal_443, 0),
            delimiter=",",
        )

        cliptextencode_446 = cliptextencode.encode(
            text=get_value_at_index(joinstrings_1003, 0),
            clip=get_value_at_index(clipsetlastlayer_982, 0),
        )

        string_literal_458 = string_literal.get_string(
            string="A sweeping desert scene under a clear blue sky. The sandy foreground is dotted with sparse shrubs, leading to rugged red rock formations stretching across the mid-ground. In the distance, jagged mesas and buttes rise, their textures emphasized by long, sun-cast shadows."
        )

        joinstrings_1004 = joinstrings.joinstring(
            string1=get_value_at_index(string_literal_1408, 0),
            string2=get_value_at_index(string_literal_458, 0),
            delimiter=",",
        )

        cliptextencode_461 = cliptextencode.encode(
            text=get_value_at_index(joinstrings_1004, 0),
            clip=get_value_at_index(clipsetlastlayer_982, 0),
        )

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_499 = randomnoise.get_noise(noise_seed=random.randint(1, 2 ** 64))

        cliptextencode_718 = cliptextencode.encode(
            text="poor quality, watermark, low resolution, lack of detail, extreme saturation, bad lighting blurry, grainy, cropped, off-center, NSFW",
            clip=get_value_at_index(checkpointloadersimple_726, 1),
        )

        cliptextencode_719 = cliptextencode.encode(
            text="high quality, detailed, portrait",
            clip=get_value_at_index(checkpointloadersimple_726, 1),
        )

        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        imageresizekj_725 = imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="lanczos",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(loadimage_1395, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_720 = vaeencode.encode(
            pixels=get_value_at_index(imageresizekj_725, 0),
            vae=get_value_at_index(checkpointloadersimple_726, 2),
        )

        cfg_literal = NODE_CLASS_MAPPINGS["Cfg Literal"]()
        cfg_literal_1397 = cfg_literal.get_float(float=1)

        cliptextencode_1403 = cliptextencode.encode(
            text="poor quality, watermark, low resolution, lack of detail, extreme saturation, bad lighting, blurry, grainy, cropped, off-center, NSFW",
            clip=get_value_at_index(checkpointloadersimple_1401, 1),
        )

        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        cfgguider_425 = cfgguider.get_guider(
            cfg=get_value_at_index(cfg_literal_1397, 0),
            model=get_value_at_index(modelpassthrough_1038, 0),
            positive=get_value_at_index(cliptextencode_431, 0),
            negative=get_value_at_index(cliptextencode_1403, 0),
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_1406 = ksamplerselect.get_sampler(sampler_name="deis")

        int_literal = NODE_CLASS_MAPPINGS["Int Literal"]()
        int_literal_1398 = int_literal.get_int(int=30)

        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        basicscheduler_747 = basicscheduler.get_sigmas(
            scheduler="beta",
            steps=get_value_at_index(int_literal_1398, 0),
            denoise=1,
            model=get_value_at_index(modelpassthrough_1038, 0),
        )

        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        samplercustomadvanced_424 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_499, 0),
            guider=get_value_at_index(cfgguider_425, 0),
            sampler=get_value_at_index(ksamplerselect_1406, 0),
            sigmas=get_value_at_index(basicscheduler_747, 0),
            latent_image=get_value_at_index(emptylatentimage_432, 0),
        )

        vaedecode = VAEDecode()
        vaedecode_429 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_424, 0),
            vae=get_value_at_index(checkpointloadersimple_1401, 2),
        )

        cfgguider_435 = cfgguider.get_guider(
            cfg=get_value_at_index(cfg_literal_1397, 0),
            model=get_value_at_index(modelpassthrough_1038, 0),
            positive=get_value_at_index(cliptextencode_439, 0),
            negative=get_value_at_index(cliptextencode_1403, 0),
        )

        samplercustomadvanced_434 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_499, 0),
            guider=get_value_at_index(cfgguider_435, 0),
            sampler=get_value_at_index(ksamplerselect_1406, 0),
            sigmas=get_value_at_index(basicscheduler_747, 0),
            latent_image=get_value_at_index(emptylatentimage_432, 0),
        )

        vaedecode_438 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_434, 0),
            vae=get_value_at_index(checkpointloadersimple_1401, 2),
        )

        cfgguider_442 = cfgguider.get_guider(
            cfg=get_value_at_index(cfg_literal_1397, 0),
            model=get_value_at_index(modelpassthrough_1038, 0),
            positive=get_value_at_index(cliptextencode_446, 0),
            negative=get_value_at_index(cliptextencode_1403, 0),
        )

        samplercustomadvanced_441 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_499, 0),
            guider=get_value_at_index(cfgguider_442, 0),
            sampler=get_value_at_index(ksamplerselect_1406, 0),
            sigmas=get_value_at_index(basicscheduler_747, 0),
            latent_image=get_value_at_index(emptylatentimage_432, 0),
        )

        vaedecode_445 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_441, 0),
            vae=get_value_at_index(checkpointloadersimple_1401, 2),
        )

        cfgguider_457 = cfgguider.get_guider(
            cfg=get_value_at_index(cfg_literal_1397, 0),
            model=get_value_at_index(modelpassthrough_1038, 0),
            positive=get_value_at_index(cliptextencode_461, 0),
            negative=get_value_at_index(cliptextencode_1403, 0),
        )

        samplercustomadvanced_456 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_499, 0),
            guider=get_value_at_index(cfgguider_457, 0),
            sampler=get_value_at_index(ksamplerselect_1406, 0),
            sigmas=get_value_at_index(basicscheduler_747, 0),
            latent_image=get_value_at_index(emptylatentimage_432, 0),
        )

        vaedecode_460 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_456, 1),
            vae=get_value_at_index(checkpointloadersimple_1401, 2),
        )

        impactmakeimagebatch = NODE_CLASS_MAPPINGS["ImpactMakeImageBatch"]()
        impactmakeimagebatch_741 = impactmakeimagebatch.doit(
            image1=get_value_at_index(vaedecode_429, 0),
            image2=get_value_at_index(vaedecode_438, 0),
            image3=get_value_at_index(vaedecode_445, 0),
            image4=get_value_at_index(vaedecode_460, 0),
        )

        blurimagefast = NODE_CLASS_MAPPINGS["BlurImageFast"]()
        blurimagefast_763 = blurimagefast.blur_image(
            radius_x=10,
            radius_y=10,
            images=get_value_at_index(impactmakeimagebatch_741, 0),
        )

        imageresizekj_728 = imageresizekj.resize(
            width=512,
            height=768,
            upscale_method="lanczos",
            keep_proportion=False,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(blurimagefast_763, 0),
            get_image_size=get_value_at_index(imageresizekj_725, 0),
        )

        vaeencode_722 = vaeencode.encode(
            pixels=get_value_at_index(imageresizekj_728, 0),
            vae=get_value_at_index(checkpointloadersimple_726, 2),
        )

        iclightconditioning = NODE_CLASS_MAPPINGS["ICLightConditioning"]()
        iclightconditioning_727 = iclightconditioning.encode(
            multiplier=0.18215,
            positive=get_value_at_index(cliptextencode_719, 0),
            negative=get_value_at_index(cliptextencode_718, 0),
            vae=get_value_at_index(checkpointloadersimple_726, 2),
            foreground=get_value_at_index(vaeencode_720, 0),
            opt_background=get_value_at_index(vaeencode_722, 0),
        )

        ksampler = KSampler()
        detailtransfer = NODE_CLASS_MAPPINGS["DetailTransfer"]()
        expressioneditor = NODE_CLASS_MAPPINGS["ExpressionEditor"]()
        imagebatchmulti = NODE_CLASS_MAPPINGS["ImageBatchMulti"]()

        ksampler_724 = ksampler.sample(
            seed=random.randint(1, 2 ** 64),
            steps=25,
            cfg=2.98,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(loadandapplyiclightunet_723, 0),
            positive=get_value_at_index(iclightconditioning_727, 0),
            negative=get_value_at_index(iclightconditioning_727, 1),
            latent_image=get_value_at_index(iclightconditioning_727, 2),
        )

        vaedecode_731 = vaedecode.decode(
            samples=get_value_at_index(ksampler_724, 0),
            vae=get_value_at_index(checkpointloadersimple_726, 2),
        )

        detailtransfer_732 = detailtransfer.process(
            mode="add",
            blur_sigma=1,
            blend_factor=0.8,
            target=get_value_at_index(vaedecode_731, 0),
            source=get_value_at_index(imageresizekj_725, 0),
        )

        expressioneditor_1342 = expressioneditor.run(
            rotate_pitch=0,
            rotate_yaw=11.5,
            rotate_roll=0,
            blink=-14,
            eyebrow=0,
            wink=25,
            pupil_x=0,
            pupil_y=0,
            aaa=0,
            eee=0,
            woo=0,
            smile=-0.24,
            src_ratio=1,
            sample_ratio=1,
            crop_factor=1.5,
            src_image=get_value_at_index(loadimage_1395, 0),
        )

        expressioneditor_1339 = expressioneditor.run(
            rotate_pitch=-8,
            rotate_yaw=-9,
            rotate_roll=0,
            blink=3,
            eyebrow=0,
            wink=11.5,
            pupil_x=0,
            pupil_y=0,
            aaa=39,
            eee=10.700000000000001,
            woo=0,
            smile=1.3,
            src_ratio=1,
            sample_ratio=1,
            crop_factor=1.5,
            src_image=get_value_at_index(loadimage_1395, 0),
        )

        expressioneditor_1340 = expressioneditor.run(
            rotate_pitch=15,
            rotate_yaw=-5.6000000000000005,
            rotate_roll=0,
            blink=5,
            eyebrow=15,
            wink=23,
            pupil_x=0,
            pupil_y=0,
            aaa=0,
            eee=10.700000000000001,
            woo=0,
            smile=-0.18,
            src_ratio=1,
            sample_ratio=1,
            crop_factor=1.5,
            src_image=get_value_at_index(loadimage_1395, 0),
        )

        expressioneditor_1341 = expressioneditor.run(
            rotate_pitch=0,
            rotate_yaw=12,
            rotate_roll=0,
            blink=0,
            eyebrow=0,
            wink=0,
            pupil_x=0,
            pupil_y=0,
            aaa=97,
            eee=-2.9000000000000004,
            woo=0,
            smile=-0.26,
            src_ratio=1,
            sample_ratio=1,
            crop_factor=1.5,
            src_image=get_value_at_index(loadimage_1395, 0),
        )

        imagebatchmulti_753 = imagebatchmulti.combine(
            inputcount=4,
            Update_inputs=None,
            image_1=get_value_at_index(expressioneditor_1342, 0),
            image_2=get_value_at_index(expressioneditor_1339, 0),
            image_3=get_value_at_index(expressioneditor_1340, 0),
            image_4=get_value_at_index(expressioneditor_1341, 0),
        )

        save_comfy_images(get_value_at_index(imagebatchmulti_753, 0),
                          [os.path.join(output_dir, f"emotions_{i}.png") for i in range(4)])
        save_comfy_images(get_value_at_index(detailtransfer_732, 0),
                          [os.path.join(output_dir, f"lighting_{i}.png") for i in range(4)])
