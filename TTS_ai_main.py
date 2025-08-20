# --- Unified TTS Controller (stream or no-stream) ---
import os
import sys
import threading
import queue
import time
from os.path import join as dirjoin

import torch
import soundfile as sf
import sounddevice as sd
from TTS.api import TTS as XTTS
import ollama
import pytchat

# Workspace paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

# Ensure RVC-beta on path early (used indirectly by rvc_python in xtts_rvc_infer)
rvc_beta_path = os.path.join(WORKSPACE_ROOT, "RVC-beta0717")
if rvc_beta_path not in sys.path:
    sys.path.insert(0, rvc_beta_path)

# Shared helpers
from xtts_rvc_infer import (
    process_text,
    convert_voice,
)

# --- Config ---
voice = 'Lisa'
rvc_model_path = "D:/program/RVC-TTS-PIPELINE/RVCModels/"
rvc_model_name = 'lisa-genshin'
prompt_file_path = "D:/program/RVC-TTS-PIPELINE/RVCPrompts/"
prompt_file_name = 'Lisa_Prompt'
generated_sounds_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"
ollama_model_name = "mistral"

# RVC settings
rvc_f0_key = -8

# Globals
audio_path = generated_sounds_path
sentences_queue = queue.Queue()


def output_handler():
    """Play and then delete the audio file enqueued by either mode."""
    while True:
        user_text, ai_generated, filename = sentences_queue.get()
        # filename may be absolute or base name
        file_path = filename if os.path.isabs(filename) else dirjoin(audio_path, filename)
        print(f"Playing: {file_path}")
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
            print(f"Finished: {file_path}")
        except Exception as e:
            print(f"Playback error for {file_path}: {e}")
        finally:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")
            sentences_queue.task_done()


def init_common():
    # Ensure output directory
    os.makedirs(generated_sounds_path, exist_ok=True)

    # Init TTS (high-level API)
    print("Initializing XTTS model...")
    tts = XTTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=torch.cuda.is_available(),
    )

    # Resolve RVC model path
    model_pth = os.path.join(rvc_model_path, f"{rvc_model_name}.pth")
    if os.path.exists(model_pth):
        print(f"Using RVC model: {model_pth}")
    else:
        print(f"Warning: RVC model not found at {model_pth}. Voice conversion will be disabled.")
        model_pth = None

    # Configure audio output device best-effort
    try:
        current_devices = sd.query_devices()
        # Choose first device with output channels
        out_idx = None
        for i, dev in enumerate(current_devices):
            if dev.get('max_output_channels', 0) > 0:
                out_idx = i
                break
        if out_idx is not None:
            sd.default.device = (sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device, out_idx)
            print(f"Using audio output device {out_idx}: {current_devices[out_idx]['name']}")
    except Exception as e:
        print(f"Warning: Could not set up audio device: {e}")

    # Resolve speaker reference wav
    _speaker_wav = ""
    voices_root = os.path.join(WORKSPACE_ROOT, "voices")
    exts = (".wav", ".mp3", ".flac", ".m4a")

    voice_dir = os.path.join(voices_root, voice)
    if os.path.isdir(voice_dir):
        for fname in os.listdir(voice_dir):
            if fname.lower().endswith(exts):
                _speaker_wav = os.path.join(voice_dir, fname)
                break
    if not _speaker_wav:
        for folder_name in os.listdir(voices_root):
            folder_path = os.path.join(voices_root, folder_name)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(exts):
                        _speaker_wav = os.path.join(folder_path, fname)
                        break
                if _speaker_wav:
                    break

    # Load prompt
    try:
        with open(dirjoin(prompt_file_path, prompt_file_name + ".txt"), encoding="utf-8") as f:
            prompt = f.read()
    except Exception as e:
        print(f"Prompt file not found: {e}")
        prompt = ""

    return tts, model_pth, _speaker_wav, prompt


def handle_sentence(text, conversation, tts, speaker_wav, model_pth):
    """Generate TTS, optionally convert via RVC, enqueue for playback."""
    audio_data, sample_rate, output_file_path = process_text(
        text=text,
        speaker_wav=speaker_wav,
        tts=tts,
        output_dir=generated_sounds_path,
    )

    if audio_data is None or not output_file_path:
        print("Failed to generate TTS audio")
        return

    # Try RVC conversion
    if model_pth is not None:
        try:
            converted_file = convert_voice(
                tts_wav_path=output_file_path,
                model_pth_path=model_pth,
                f0_up_key=rvc_f0_key,
            )
            if converted_file and os.path.exists(converted_file):
                sentences_queue.put((conversation[-1]["content"], text, converted_file))
                print(f"Converted voice saved to: {converted_file}")
                # Delete the original TTS file
                try:
                    if os.path.exists(output_file_path):
                        os.remove(output_file_path)
                        print(f"Deleted original TTS file: {output_file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete original TTS file {output_file_path}: {e}")
                return
            else:
                print("Voice conversion failed, using original TTS audio")
        except Exception as e:
            print(f"Error in voice conversion: {e}")

    # Enqueue original TTS
    sentences_queue.put((conversation[-1]["content"], text, output_file_path))


# --- Modes ---

def run_nostream():
    tts, model_pth, speaker_wav, prompt = init_common()
    conversation = []

    t = threading.Thread(target=output_handler, daemon=True)
    t.start()

    first = True
    new_line_keywords = [".", "?", "!"]
    stop_keywords = ["stop", "nope", "good bye"]
    reset_keywords = ["reset", "new", "next"]

    while True:
        try:
            message = input("user input:")
        except KeyboardInterrupt:
            break
        if message in stop_keywords:
            break
        if message in reset_keywords:
            first = True
            continue
        if first:
            message = prompt + message
            first = False
        conversation.append({"role": "user", "content": message})
        sentences = [""]

        stream = ollama.chat(model=ollama_model_name, messages=conversation, stream=True)
        print("user input:" + conversation[-1]["content"])
        idx = 0
        for chunk in stream:
            chunk = chunk["message"]["content"]
            sentences[-1] += chunk
            if any(k in chunk for k in new_line_keywords) or (sentences[-1] and any(sentences[-1].endswith(k) for k in new_line_keywords)):
                try:
                    handle_sentence(sentences[-1], conversation, tts, speaker_wav, model_pth)
                    sentences.append("")
                    idx += 1
                except Exception as e:
                    print(f"Error processing text: {e}")
                    sentences.append("")
        conversation.append({"role": "assistant", "content": "".join(sentences)})
        sentences_queue.join()


def sentence_boundary_hit(chunk, buffer, keywords):
    if any(k in chunk for k in keywords):
        return True
    if buffer and any(buffer.endswith(k) for k in keywords):
        return True
    return False


def run_stream():
    tts, model_pth, speaker_wav, prompt = init_common()

    video_id = input("Enter YouTube video ID: ")
    chat = pytchat.create(video_id=video_id)
    if not chat.is_alive():
        print("Stream is not alive")
        return

    conversation = []

    t = threading.Thread(target=output_handler, daemon=True)
    t.start()

    first = True
    new_line_keywords = [".", "?", "!"]
    stop_keywords = ["stop", "nope", "good bye"]
    reset_keywords = ["reset", "new", "next"]

    while True:
        for c in chat.get().sync_items():
            if c.message in stop_keywords:
                return
            if c.message in reset_keywords:
                first = True
                continue
            # Ask for user confirmation
            try:
                if input(f"{c.datetime} [{c.author.name}]- {c.message} accept? (y/n): ").lower() != 'y':
                    continue
            except KeyboardInterrupt:
                return

            if first:
                c.message = prompt + c.message
                first = False

            conversation.append({"role": "user", "content": c.message})
            sentences = [""]
            idx = 0

            stream = ollama.chat(model=ollama_model_name, messages=conversation, stream=True)
            print("user input:" + conversation[-1]["content"])
            for chunk in stream:
                chunk = chunk["message"]["content"]
                sentences[-1] += chunk
                if sentence_boundary_hit(chunk, sentences[-1], new_line_keywords):
                    try:
                        handle_sentence(sentences[-1], conversation, tts, speaker_wav, model_pth)
                        sentences.append("")
                        idx += 1
                    except Exception as e:
                        print(f"Error processing text: {e}")
            conversation.append({"role": "assistant", "content": "".join(sentences)})
            sentences_queue.join()
        time.sleep(1)


if __name__ == "__main__":
    print("Select mode: [1] Stream (YouTube)  [2] No-Stream (console)")
    choice = input("Enter 1 or 2: ").strip()
    try:
        mode = int(choice)
    except Exception:
        mode = 2

    if mode == 1:
        run_stream()
    else:
        run_nostream()
