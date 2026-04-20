"""
audio_generator.py
------------------
Handles generation of marketing scripts and ElevenLabs audio synthesis.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize client only if key is present
try:
    client = ElevenLabs(api_key=API_KEY) if API_KEY else None
except Exception as e:
    logging.getLogger(__name__).error(f"Error initializing ElevenLabs client: {e}")
    client = None

# Default Voice: "Matilda" (Professional) - Safe for Free users
DEFAULT_VOICE_ID = "XrExE9yKIg1WjnnlVkGX"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def generate_marketing_script(product: dict) -> str:
    """
    Creates a punchy 10-15 second marketing script from product metadata.
    """
    title    = product.get("title", "Product")
    category = product.get("category", "E-Commerce")
    color    = product.get("color", "")
    style    = product.get("style", "modern")
    
    # Clean up title for reading (remove reg marks, etc)
    clean_title = title.replace("&reg;", "").replace("(R)", "").strip()
    
    script = (
        f"Introducing the all-new {color} {clean_title}. "
        f"A standout in our {category} collection, it combines {style} design "
        f"with premium materials. Elevate your lifestyle with this exceptional product, "
        f"now available for you."
    )
    return script


def synthesize_audio(text: str, output_path: str | Path, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    """
    Calls ElevenLabs API to generate audio from text.
    Saves the result to output_path.
    """
    if not client:
        return False
        
    try:
        # ElevenLabs v1.x SDK uses client.text_to_speech.convert
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # The SDK returns a generator of bytes
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
                
        return True
    except Exception as e:
        # Fallback for "Payment Required" or "Library Voice" restrictions
        if "402" in str(e) or "paid_plan_required" in str(e):
            logger.warning(f"Voice {voice_id} restricted. Attempting fallback voice.")
            return _synthesize_fallback(text, output_path)
            
        logger.error(f"Failed to synthesize audio: {e}")
        return False


def _synthesize_fallback(text: str, output_path: str | Path) -> bool:
    """Fallback to the first available voice on the account."""
    try:
        voices = client.voices.get_all().voices
        if not voices:
            return False
        fallback_id = voices[0].voice_id
        
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=fallback_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Fallback synthesis failed: {e}")
        return False


def generate_product_audio(product: dict, output_dir: Path) -> dict:
    """
    Master function to script and synthesize audio for a product.
    Saves to <output_dir>/<product_id>_narration.mp3
    """
    pid    = product.get("id") or product.get("product_id") or "unknown_product"
    script = generate_marketing_script(product)
    opath  = Path(output_dir) / f"{pid}_narration.mp3"
    
    # Ensure dir exists
    opath.parent.mkdir(parents=True, exist_ok=True)
    
    success = synthesize_audio(script, opath)
    
    return {
        "product_id": pid,
        "script": script,
        "audio_path": str(opath) if success else None,
        "success": success
    }
