import argparse
import os
from dotenv import load_dotenv

load_dotenv()

# Platform constants
PLATFORM_HOSTED, PLATFORM_SERVERLESS = "hosted", "serverless"

if os.environ.get("PLATFORM", PLATFORM_HOSTED) == PLATFORM_SERVERLESS:
    COMFYUI_PATH = os.environ["COMFYUI_PATH"]
else:
    COMFYUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI")
    os.environ["COMFYUI_PATH"] = COMFYUI_PATH

    os.environ["UV_LINK_MODE"] = "copy"
    os.environ["PIP_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache/pip")
    os.makedirs(os.environ["PIP_CACHE_DIR"], exist_ok=True)

print("ComfyUI path: ", COMFYUI_PATH)
# for network volume
os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"


def pip_install(platform, cmd):
    if platform == PLATFORM_SERVERLESS:
        return f"{cmd} --system --compile-bytecode"
    else:
        return f"{cmd} --cache-dir {os.environ['PIP_CACHE_DIR']}"


def run_cmd(command):
    """Run a shell command"""
    print(f"🔄 Running: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"❌ Command failed: {command} (Exit Code: {exit_code})")
        exit(1)


def install_dependencies():
    """Install system dependencies and Python packages."""
    print("📦 Installing dependencies...")
    run_cmd("apt-get update && apt-get install -y git wget curl libgl1-mesa-glx libglib2.0-0 tmux emacs git-lfs")
    print("✅ Dependencies installed.")


def install_git_repo(repo_url, install_path, requirements=False, submodules=False, platform=PLATFORM_HOSTED):
    """Clone or update a git repository and handle its dependencies"""
    original_dir = os.getcwd()

    if not os.path.exists(install_path) or not os.path.isdir(install_path) or not os.path.exists(
            os.path.join(install_path, ".git")):
        print(f"📂 Cloning {os.path.basename(install_path)}...")
        run_cmd(f"git clone {repo_url} {install_path}")
    else:
        print(f"🔄 {os.path.basename(install_path)} exists. Checking for updates...")

    # Change to repo directory and update
    os.chdir(install_path)
    run_cmd("git pull")

    if submodules:
        run_cmd("git submodule update --init --recursive")
    if requirements:
        run_cmd(pip_install(platform, "uv pip install -r requirements.txt"))

    print(f"✅ {os.path.basename(install_path)} installed and updated.")
    os.chdir(original_dir)


def install_modules(platform=PLATFORM_HOSTED):
    """Set up ComfyUI and other git submodules."""

    if platform == PLATFORM_SERVERLESS:
        print("📂 Serverless platform: Skipping git operations, using pre-copied submodules")

        # For serverless, the submodule directories are already copied by Modal's add_local_dir
        # We just need to install their dependencies
        print("📦 Installing dependencies for pre-copied submodules...")
        original_dir = os.getcwd()

        # Known submodule directories that should be present
        submodule_dirs = ["MV_Adapter", "ComfyUI_AutoCropFaces", "ComfyUI", "LoRACaptioner"]

        for submodule_dir in submodule_dirs:
            if os.path.exists(submodule_dir):
                # Skip ai_toolkit dependencies
                if submodule_dir == "ai_toolkit":
                    print(f"⏭️ Skipping dependencies for {submodule_dir}...")
                    continue

                requirements_path = os.path.join(submodule_dir, "requirements.txt")
                if os.path.exists(requirements_path):
                    print(f"📦 Installing dependencies for {submodule_dir}...")
                    os.chdir(submodule_dir)
                    run_cmd(pip_install(platform, "uv pip install -r requirements.txt"))
                    os.chdir(original_dir)
                else:
                    print(f"ℹ️ No requirements.txt found for {submodule_dir}")
            else:
                print(f"⚠️ Expected submodule directory not found: {submodule_dir}")

    else:
        print("📂 Initializing and updating git submodules...")

        # Initialize and update all submodules defined in .gitmodules to their latest versions
        run_cmd("git submodule init")
        run_cmd("git submodule update --init --recursive --remote")

        # Install dependencies for each submodule
        print("📦 Installing dependencies for submodules...")
        original_dir = os.getcwd()

        # Get list of submodules from .gitmodules
        with open(".gitmodules", "r") as f:
            for line in f:
                if line.strip().startswith("path = "):
                    submodule_path = line.split("=")[1].strip()
                    # Skip ai_toolkit dependencies
                    if submodule_path == "ai_toolkit":
                        print(f"⏭️ Skipping dependencies for {submodule_path}...")
                        continue
                    if os.path.exists(os.path.join(submodule_path, "requirements.txt")):
                        print(f"📦 Installing dependencies for {submodule_path}...")
                        os.chdir(submodule_path)
                        run_cmd(pip_install(platform, "uv pip install -r requirements.txt"))
                        os.chdir(original_dir)

    # Make CLI wrapper scripts executable
    if os.path.exists("scripts"):
        run_cmd("chmod +x scripts/*.sh")

    print("✅ All submodules processed and dependencies installed.")


def download_huggingface_models(cache_models=True):
    """Download required models from Hugging Face."""
    from huggingface_hub import hf_hub_download
    hf_models = [
        {"repo_id": "Comfy-Org/flux1-dev", "filename": "flux1-dev-fp8.safetensors", "folder": "checkpoints"},
        {"repo_id": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", "filename": "diffusion_pytorch_model.safetensors",
         "folder": "controlnet"},
        {"repo_id": "skbhadra/ClearRealityV1", "filename": "4x-ClearRealityV1.pth", "folder": "upscale_models"},
        {"repo_id": "guozinan/PuLID", "filename": "pulid_flux_v0.9.1.safetensors", "folder": "pulid"},
        {"repo_id": "lllyasviel/ic-light", "filename": "iclight_sd15_fbc.safetensors", "folder": "unet"},
        {"repo_id": "Bingsu/adetailer", "filename": "face_yolov8m.pt", "folder": "ultralytics/bbox"},
    ]

    # Dictionary mapping repo_ids to specific filenames
    filename_mappings = {
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro": "Flux_Dev_ControlNet_Union_Pro_ShakkerLabs.safetensors",
        "Bingsu/adetailer": "face_yolov8m.pt",
    }

    for model in hf_models:
        try:
            target_dir = os.path.join(COMFYUI_PATH, "models", model["folder"])
            os.makedirs(target_dir, exist_ok=True)

            # Use mapping if it exists, otherwise use original filename
            file_name_only = filename_mappings.get(model["repo_id"], os.path.basename(model["filename"]))
            target_path = os.path.join(target_dir, file_name_only)

            if os.path.exists(target_path):
                print(f"✅ Already exists: {file_name_only}")
                continue

            if cache_models:
                # Download to HF_HOME cache and create symlink
                model_path = hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=model["filename"],
                    cache_dir=os.environ.get("HF_HOME"),
                    repo_type=model.get("repo_type", "model"),
                    token=os.environ.get("HF_TOKEN")
                )
                os.symlink(model_path, target_path)
                print(f"✅ Linked: {file_name_only}")
            else:
                hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=model["filename"],
                    local_dir=target_dir,
                    repo_type=model.get("repo_type", "model"),
                    token=os.environ.get("HF_TOKEN")
                )
                print(f"✅ Downloaded: {file_name_only} directly to {target_dir}")
        except Exception as e:
            print(f"❌ Failed to download {model['filename']}: {e}")


def download_external_models(cache_models=True):
    """Download required models from CivitAI."""
    import requests
    civitai_token = os.environ.get("CIVITAI_API_KEY")
    if not civitai_token:
        print("⚠️ Warning: CIVITAI_API_KEY not found in environment variables")
        return

    # Create cache directory if using cache_models
    cache_dir = os.environ.get("HF_HOME", "./cache")
    if cache_models:
        os.makedirs(cache_dir, exist_ok=True)

    # Models not available on Hugging Face
    external_models = [
        {
            "url": f"https://civitai.com/api/download/models/782002?type=Model&format=SafeTensor&size=full&fp=fp16&token={civitai_token}",
            "folder": "checkpoints", "filename": "juggernaut-xl.safetensors"},
        {
            "url": f"https://civitai.com/api/download/models/90072?type=Model&format=SafeTensor&size=pruned&fp=fp16&token={civitai_token}",
            "folder": "checkpoints", "filename": "photon.safetensors"},
    ]

    for model in external_models:
        target_dir = os.path.join(COMFYUI_PATH, "models", model["folder"])
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, model["filename"])

        if os.path.exists(file_path):
            print(f"✅ Model already exists: {model['filename']}")
            continue

        try:
            if cache_models:
                # Download to cache directory and create symlink
                cache_path = os.path.join(cache_dir, model["filename"])

                if not os.path.exists(cache_path):
                    print(f"📥 Downloading to cache: {model['filename']}")
                    response = requests.get(model["url"], stream=True)
                    response.raise_for_status()
                    with open(cache_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Create symlink from cache to target
                os.symlink(cache_path, file_path)
                print(f"✅ Linked: {model['filename']}")
            else:
                # Download directly to target directory
                print(f"📥 Downloading: {model['filename']}")
                response = requests.get(model["url"], stream=True)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded: {model['filename']} directly to {target_dir}")
        except requests.RequestException as e:
            print(f"❌ Failed to download {model['filename']} from {model['url']}: {e}")

    download_and_extract_antelopev2()


def download_and_extract_antelopev2():
    """Download and extract AntelopeV2 model for insightface."""
    import zipfile, requests, shutil

    base_path = os.path.join(COMFYUI_PATH, "models", "insightface/models")
    model_target_path = os.path.join(base_path, "antelopev2")
    download_url = "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"
    zip_path = os.path.join(base_path, "antelopev2.zip")
    temp_extract_path = os.path.join(base_path, "temp_antelopev2")

    os.makedirs(base_path, exist_ok=True)

    if not os.path.exists(model_target_path) or not os.listdir(model_target_path):
        # First, remove any existing problematic directories
        if os.path.exists(model_target_path):
            shutil.rmtree(model_target_path)

        print(f"📥 Downloading AntelopeV2 model...")
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("✅ Download complete.")

            # Create a temporary extraction directory
            os.makedirs(temp_extract_path, exist_ok=True)

            print("📂 Extracting AntelopeV2 model...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_extract_path)
            print("✅ Extraction complete.")

            # Create the target directory
            os.makedirs(model_target_path, exist_ok=True)

            # Move the model files to the correct location
            # The ZIP contains a nested antelopev2 directory we need to move files from
            nested_model_dir = os.path.join(temp_extract_path, "antelopev2")
            if os.path.exists(nested_model_dir):
                for item in os.listdir(nested_model_dir):
                    source = os.path.join(nested_model_dir, item)
                    target = os.path.join(model_target_path, item)
                    shutil.move(source, target)

            # Clean up
            if os.path.exists(temp_extract_path):
                shutil.rmtree(temp_extract_path)
            if os.path.exists(zip_path):
                os.remove(zip_path)

            print("🗑️ Cleaned up temporary files.")
            print("✅ AntelopeV2 model installed correctly.")

        except Exception as e:
            print(f"❌ Failed to download/extract AntelopeV2: {e}")
    else:
        print("✅ AntelopeV2 model already exists")


def install_custom_nodes(platform=PLATFORM_HOSTED):
    """Install all custom nodes for ComfyUI."""
    # Make sure we have the absolute path to ComfyUI_PATH
    original_dir = os.getcwd()
    custom_nodes_path = os.path.join(COMFYUI_PATH, "custom_nodes")

    # Install ComfyUI-Manager first
    install_git_repo(
        "https://github.com/ltdrdata/ComfyUI-Manager",
        os.path.join(custom_nodes_path, "ComfyUI-Manager"),
        requirements=True,
        platform=platform
    )

    # List of custom nodes to install via comfy node install
    custom_nodes = [
        "comfyui_essentials",
        "comfyui-advancedliveportrait",
        "comfyui-ic-light",
        "comfyui-impact-pack",
        "comfyui-custom-scripts",
        "rgthree-comfy",
        "comfyui-easy-use",
        "comfyui-impact-subpack",
    ]

    # Now update all existing nodes
    run_cmd(f"comfy --here --skip-prompt node update all")

    # Then install any missing nodes
    for node in custom_nodes:
        run_cmd(f"comfy --here --skip-prompt node install {node}")

    print("✅ Installed and updated all ComfyUI registry nodes.")

    # List of custom nodes to install from git
    custom_nodes_git = [
        {
            "repo": "https://github.com/WASasquatch/was-node-suite-comfyui.git",
            "name": "was-node-suite-comfyui",
            "requirements": True
        },
        {
            "repo": "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git",
            "name": "ComfyUI_UltimateSDUpscale",
            "submodules": True
        },
        {
            "repo": "https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git",
            "name": "ComfyUI-PuLID-Flux-Enhanced",
            "requirements": True
        },
        {
            "repo": "https://github.com/giriss/comfy-image-saver.git",
            "name": "comfy-image-saver",
            "requirements": True
        },
        {
            "repo": "https://github.com/spacepxl/ComfyUI-Image-Filters.git",
            "name": "ComfyUI-Image-Filters",
            "requirements": True
        },
        {
            "repo": "https://github.com/Jonseed/ComfyUI-Detail-Daemon",
            "name": "ComfyUI-Detail-Daemon",
            "requirements": True
        },
        {
            "repo": "https://github.com/kijai/ComfyUI-KJNodes.git",
            "name": "ComfyUI-KJNodes",
            "requirements": True
        },
    ]

    for node in custom_nodes_git:
        repo_name = node["name"]
        repo_path = os.path.join(custom_nodes_path, repo_name)
        install_git_repo(
            node["repo"],
            repo_path,
            requirements=node.get("requirements", False),
            submodules=node.get("submodules", False),
            platform=platform
        )

        # Handle any post-install commands
        if "post_install" in node:
            current_dir = os.getcwd()
            os.chdir(repo_path)
            for command in node["post_install"]:
                run_cmd(command)
            os.chdir(current_dir)

    os.chdir(original_dir)


def final_steps(platform=PLATFORM_HOSTED):
    """Force install torch 2.5.1 for stability of the ComfyUI custom nodes"""
    run_cmd(pip_install(platform,
                        "uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"))
    # Delete ComfyUI Manager for performance
    manager_path = os.path.join(COMFYUI_PATH, "custom_nodes", "ComfyUI-Manager")
    if os.path.exists(manager_path):
        print("🗑️ Deleting ComfyUI Manager...")
        import shutil
        shutil.rmtree(manager_path)
        print("✅ ComfyUI Manager deleted.")


def full_install(cache_models=True, platform=PLATFORM_HOSTED):
    if platform == PLATFORM_HOSTED:
        install_dependencies()
    install_modules(platform)
    install_custom_nodes(platform)
    if platform == PLATFORM_HOSTED:
        download_huggingface_models(cache_models)
        download_external_models(cache_models)
    final_steps(platform)
    print("✅ Setup Complete!")


def main():
    parser = argparse.ArgumentParser(description="CharForge installation script")
    parser.add_argument("--no-cache-models", action="store_false", dest="cache_models",
                        help="Download models directly instead of using cache")
    parser.add_argument("--platform", type=str, default=PLATFORM_HOSTED, choices=[PLATFORM_HOSTED, PLATFORM_SERVERLESS],
                        help="Platform to configure for (hosted, serverless)")

    args = parser.parse_args()

    full_install(
        cache_models=args.cache_models,
        platform=args.platform
    )


if __name__ == "__main__":
    main()
