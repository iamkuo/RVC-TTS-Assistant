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
from xtts_rvc_infer import (
    process_text,
    rvc_pipe_convert,
    init_xtts_model_with_deepspeed,
)
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

# --- Config ---
out_idx = 6
voice = 'Lisa'
rvc_model_path = "D:/program/RVC-TTS-PIPELINE/RVCModels/"
rvc_model_name = 'lisa-genshin'
prompt_file_path = "D:/program/RVC-TTS-PIPELINE/RVCPrompts/"
prompt_file_name = 'Lisa_Prompt'
generated_sounds_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"
ollama_model_name = "mistral"
ai_response_file = os.path.join(WORKSPACE_ROOT, "ai_response.txt")
user_input_file = os.path.join(WORKSPACE_ROOT, "user_input.txt")

# RVC settings
rvc_f0_key = -8

# Globals
audio_path = generated_sounds_path
sentences_queue = queue.Queue()


def write_text_file(path, text):
    """Write text to a UTF-8 file (overwrite)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Warning: could not write to {path}: {e}")


def output_handler():
    """Play and then delete the audio file enqueued by either mode."""
    while True:
        user_text, ai_generated, filename = sentences_queue.get()
        # filename may be absolute or base name
        file_path = filename if os.path.isabs(filename) else dirjoin(audio_path, filename)
        print(f"Playing: {file_path}")
        try:
            # Update ai_response.txt with the sentence that is about to play
            try:
                write_text_file(ai_response_file, ai_generated)
            except Exception as e:
                print(f"Warning: could not update ai_response file at playback start: {e}")
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

    # Clear output file at startup to avoid stale text
    try:
        write_text_file(ai_response_file, "")
    except Exception as e:
        print(f"Warning: could not clear {ai_response_file} at startup: {e}")

    # Init TTS via shared helper (high-level API path by default)
    print("Initializing XTTS model via xtts_rvc_infer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts = init_xtts_model_with_deepspeed(
        config_path="",  # unused when use_deepspeed=False
        checkpoint_path="",  # unused when use_deepspeed=False
        device=device,
        use_deepspeed=False,
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
        sd.default.device = (sd.default.device[6] if isinstance(sd.default.device, tuple) else sd.default.device, out_idx)
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
            converted_file = rvc_pipe_convert(
                input_wav_path=output_file_path,
                model_pth_path=model_pth
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
        # Clear output file at the start of a new question
        write_text_file(ai_response_file, "")
        # Write the user's raw input to file immediately
        write_text_file(user_input_file, message)
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

    # Ask once whether to manually verify each comment
    try:
        verify_each = input("Manually verify each comment? (y/n): ").strip().lower().startswith('y')
    except KeyboardInterrupt:
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
            # Only moderators/owner can issue stop/reset commands
            is_mod = bool(getattr(c.author, "isChatModerator", False) or getattr(c.author, "isChatOwner", False))
            if is_mod and c.message in stop_keywords:
                return
            if c.message in reset_keywords:
                first = True
                continue

            # Manual verification if enabled
            if verify_each:
                try:
                    if input(f"{c.datetime} [{c.author.name}]- {c.message} accept? (y/n): ").strip().lower() != 'y':
                        continue
                except KeyboardInterrupt:
                    return

            # Write raw chat message to user file
            # Clear output file at the start of a new accepted message
            write_text_file(ai_response_file, "")
            write_text_file(user_input_file, c.message)
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
