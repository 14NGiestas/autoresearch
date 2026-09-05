# /// script
# requires-python = ">=3.11, <3.13"
# dependencies = [
#     "kokoro",
#     "ollama",
#     "soundfile",
#     "torch; sys_platform == 'linux'",
#     "numpy",
# ]
# ///


import subprocess
import time
from pathlib import Path

import numpy as np
import ollama
import soundfile as sf
from kokoro import KPipeline

# ─── Configuration ───────────────────────────────────────────────────────
OLLAMA_MODEL = "fermi-mini:latest"
KOKORO_LANG_CODE = "p"              # ← 'p' = Portuguese (Brasil)
KOKORO_VOICE = "pm_santa"            # ← Voz masculina BR (ou "pf_dora" para feminina)
KOKORO_SPEED = 1.1
SAMPLE_RATE = 24000
OUTPUT_FILE = Path(f"output-{KOKORO_VOICE}.wav")
USER_PROMPT = "Tell me one sentence about why AVX-512 is great for AI."


def generate_response(model: str, prompt: str) -> str:
    """Generate text response via Ollama."""
    print("--- Ollama Thinking ---")
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]


def synthesize_and_save(text: str, output_path: Path) -> None:
    """Synthesize speech and save to WAV."""
    print(f"--- Synthesizing (AVX-512) ---\nText: {text}")
    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
    start = time.time()

    chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text=text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        chunks.append(audio)

    if not chunks:
        print("⚠️ No audio generated.")
        return

    full_audio = np.concatenate(chunks)
    sf.write(str(output_path), full_audio, SAMPLE_RATE)
    print(f"✅ Saved to {output_path.resolve()} ({time.time() - start:.2f}s)")


def try_playback(path: Path) -> bool:
    """Attempt playback if a real audio sink exists."""
    try:
        out = subprocess.run(["wpctl", "status"], capture_output=True, text=True, check=True)
        # Ignora 'auto_null' ou 'Saída de falsa'
        has_real_sink = any(
            line.strip() and "auto_null" not in line and "falsa" not in line.lower()
            for line in out.stdout.splitlines()
            if "Sinks:" in out.stdout.split(out.stdout.split("Sinks:")[1])[0]
        )
        if not has_real_sink:
            print("ℹ️ No physical audio sink detected. Playback skipped.")
            return False

        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=True)
        return True
    except Exception:
        return False


def main() -> None:
    """Main execution flow."""
    try:
        text = generate_response(OLLAMA_MODEL, USER_PROMPT)
        synthesize_and_save(text, OUTPUT_FILE)
        try_playback(OUTPUT_FILE)
    except Exception as err:
        print(f"❌ Failed: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
