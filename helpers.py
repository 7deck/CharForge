import os
import subprocess


def run_subprocess(cmd, work_dir=None):
    """Run a subprocess command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        output = []
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            output.append(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        return '\n'.join(output)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        raise


def find_character_lora(name, work_dir=None):
    """
    Find the LoRA model for a character.
    
    Args:
        name (str): Character name
        work_dir (str, optional): Working directory path. If None, uses ./scratch/{name}/
    
    Returns:
        str: Path to the LoRA model file
    """
    # Set default work_dir if not provided
    if work_dir is None:
        # Use the APP_PATH environment variable as the base path if available
        app_path = os.environ.get('APP_PATH', os.getcwd())
        work_dir = os.path.join(app_path, 'scratch', name)

    # Check if work_dir exists
    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"Character directory not found: {work_dir}")

    # Find LoRA directory
    lora_dir = os.path.join(work_dir, "char")
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    # Look for specific LoRA file - char.safetensors
    lora_file = "char.safetensors"
    lora_path = os.path.join(lora_dir, lora_file)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA model file not found: {lora_path}")

    return lora_path


def optimize_prompt(user_prompt, character_name, use_reference_image, work_dir=None):
    """
    Optimize the user prompt using LoRACaptioner.
    
    Args:
        user_prompt (str): Raw user prompt
        character_name (str): Character name
        use_reference_image (bool): Whether to use the --reference_image flag.
        work_dir (str, optional): Working directory path. If None, uses ./scratch/{name}/
    
    Returns:
        str: Optimized prompt
    """
    if work_dir is None:
        # Use the APP_PATH environment variable as the base path if available
        app_path = os.environ.get('APP_PATH', os.getcwd())
        work_dir = os.path.join(app_path, 'scratch', character_name)

    # Get captions directory (should be in the sheet folder)
    captions_dir = os.path.join(work_dir, "sheet")

    if not os.path.exists(captions_dir):
        print(f"Warning: Captions directory not found: {captions_dir}")
        return user_prompt

    # Convert captions_dir to absolute path
    captions_dir = os.path.abspath(captions_dir)

    # Find the LoRACaptioner directory using APP_PATH
    app_path = os.environ.get('APP_PATH', os.getcwd())
    loracaptioner_dir = os.path.join(app_path, "LoRACaptioner")

    if not os.path.exists(loracaptioner_dir):
        print(f"Warning: LoRACaptioner directory not found: {loracaptioner_dir}")
        return user_prompt

    # Use absolute path to prompt.py
    prompt_script = os.path.join(loracaptioner_dir, "prompt.py")

    # Check for reference image (original.png) in the captions directory
    reference_image_path = os.path.join(captions_dir, "original.png")

    # Run LoRACaptioner prompt optimization via CLI
    cmd = [
        "python", prompt_script,
        "--prompt", user_prompt,
        "--captions", captions_dir,
    ]
    if use_reference_image:
        cmd += ["--reference_image", reference_image_path]

    try:
        print(f"Optimizing prompt using LoRACaptioner...")
        print(f"Running: {' '.join(cmd)} in {loracaptioner_dir}")
        output = run_subprocess(cmd, loracaptioner_dir)

        # Extract the optimized prompt from the output
        lines = output.split('\n')
        for i in range(len(lines)):
            if "Optimized Prompt:" in lines[i] and i + 1 < len(lines):
                optimized_prompt = lines[i + 1].strip()
                return optimized_prompt

        # If we couldn't find the optimized prompt in the output
        print("Warning: Could not parse optimized prompt from output")
        return user_prompt

    except Exception as e:
        print(f"Error optimizing prompt: {e}")
        return user_prompt
