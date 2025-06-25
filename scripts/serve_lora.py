import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI()

# Get RunPod environment variables
PORT = 8889
POD_ID = os.environ["RUNPOD_POD_ID"]

character_name = "character_name"

# Mount the scratch directory
scratch_dir = "/workspace/CharForge/scratch"
app.mount("/loras", StaticFiles(directory=scratch_dir), name="loras")


def get_public_url_base() -> str:
    """Get the base URL for the RunPod proxy"""
    return f"https://{POD_ID}-{PORT}.proxy.runpod.net"


@app.get("/")
async def root():
    """Root endpoint that returns the base URL and instructions"""
    base_url = get_public_url_base()
    return {
        "message": "LoRA server is running",
        "base_url": base_url,
        "instructions": f"Access your LoRAs at {base_url}/loras/{character_name}/char/char.safetensors"
    }


if __name__ == "__main__":
    print(f"Server running at {get_public_url_base()}")
    print(f"{get_public_url_base()}/loras/{character_name}/char/char.safetensors")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
