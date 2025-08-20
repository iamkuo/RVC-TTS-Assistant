# --- Imports ---
import os
import sys
import threading
import queue
import time
from os.path import join as dirjoin

print("Python executable:", sys.executable)
# Add local workspace root to sys.path for local modules (e.g., rvc_python)
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

# Optional/third-party modules (no try/except, let errors show)
from TTS.api import TTS as XTTS
import sounddevice as sd
import soundfile as sf
import ollama
import pytchat
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
audio_path = generated_sounds_path

sentences_queue = queue.Queue()
new_line_keywords = [".","?","!"]
stop_keywords = ["stop" , "nope" , "good bye"] #lower case
reset_keywords = ["reset","new","next"]
first = True

# Clear user/AI text files at start
for fname in ("user_input.txt", "ai_response.txt"):
    with open(fname, "w", encoding="utf-8") as f:
        pass

def sentence_boundary_hit(chunk, buffer, keywords):
    # Trigger on any punctuation appearing in the new chunk, or buffer ending with it
    if any(k in chunk for k in keywords):
        return True
    if buffer and any(buffer.endswith(k) for k in keywords):
        return True
    return False

def output_handler():
    while True:
        user_text, ai_generated, filename = sentences_queue.get()
        # filename is basename; reconstruct full path
        file_path = filename if os.path.isabs(filename) else dirjoin(audio_path, filename)
        with open("user_input.txt", "w", encoding="utf-8") as user_file:
            user_file.write("User:" + user_text)
            user_file.flush()
        with open("ai_response.txt", "w", encoding="utf-8") as ai_file:
            ai_file.write(ollama_model_name + ":")
            ai_file.flush()
        print(f"Playing: {file_path}")
        if sf is not None and sd is not None:
            try:
                data, samplerate = sf.read(file_path)
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print(f"Audio playback error: {e}")
        else:
            print("Audio playback skipped (missing soundfile or sounddevice)")
        print(f"Finished: {file_path}")
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")
        sentences_queue.task_done()

# --- Main logic ---
rvc_model_path_full = rvc_model_path + rvc_model_name + ".pth"
# Resolve RVC model .pth path (align with nostream)
if os.path.exists(rvc_model_path_full):
    print(f"Using RVC model: {rvc_model_path_full}")
    model_pth = rvc_model_path_full
else:
    print(f"Warning: RVC model not found at {rvc_model_path_full}. Voice conversion will be disabled.")
    model_pth = None

# Initialize audio device
os.makedirs(generated_sounds_path, exist_ok=True)
if sd is not None:
    try:
        # set only output device; keep input as system default
        current_devices = sd.default.device
        input_device = current_devices[0] if current_devices and current_devices[0] is not None else None
        # Use type: ignore to suppress the type checker warning
        sd.default.device = (input_device, 4)  # type: ignore
    except Exception as e:
        print(f"Warning: Could not set sounddevice default device: {e}")
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, dev in enumerate(devices):
                print(f"  [{i}] {dev}")
        except Exception:
            pass

# Initialize TTS
try:
    tts = XTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
except Exception as e:
    print(f"XTTS init error: {e}")
    tts = None

# Resolve speaker reference
_speaker_wav = ""
try:
    voices_root = os.path.join(WORKSPACE_ROOT, "voices")
    exts = (".wav", ".mp3", ".flac", ".m4a")
    
    # 1) folder matching the selected voice
    voice_dir = os.path.join(voices_root, voice)
    if os.path.isdir(voice_dir):
        for fname in os.listdir(voice_dir):
            if fname.lower().endswith(exts):
                _speaker_wav = os.path.join(voice_dir, fname)
                break
    
    # 2) search in any voice folder if specified voice not found
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
except Exception:
    pass

try:
    with open(dirjoin(prompt_file_path, prompt_file_name + ".txt"), encoding="utf-8") as f:
        prompt = f.read()
except Exception as e:
    print(f"Prompt file not found: {e}")
    prompt = ""

# Initialize YouTube chat
video_id = input("Enter YouTube video ID: ")
chat = pytchat.create(video_id=video_id)
if not chat.is_alive():
    print("Stream is not alive")
    exit()

conversation = []
tts_thread = threading.Thread(target=output_handler, daemon=True)
tts_thread.start()

while True:
    for c in chat.get().sync_items():
        # Handle one input sentence
        if c.message in stop_keywords:
            break
        if c.message in reset_keywords:
            first = True
            continue
        
        # Ask for user confirmation
        if input(f"{c.datetime} [{c.author.name}]- {c.message} accept? (y/n): ").lower() != 'y':
            continue
            
        if first:
            c.message = prompt + c.message
            first = False
            
        conversation.append({"role": "user", "content": c.message})
        sentences = [""]
        idx = 0
        
        if ollama is not None:
            stream = ollama.chat(
                model=ollama_model_name,
                messages=conversation, stream=True)
            print("user input:" + conversation[-1]["content"])
            
            for chunk in stream:
                chunk = chunk["message"]["content"]
                sentences[-1] += chunk
                if sentence_boundary_hit(chunk, sentences[-1], new_line_keywords):
                    if tts is not None:
                        try:
                            # Generate TTS using shared helper (saves file and returns its path)
                            audio_data, sample_rate, output_file_path = process_text(
                                text=sentences[-1],
                                speaker_wav=_speaker_wav if _speaker_wav else "",
                                tts=tts,
                                output_dir=generated_sounds_path,
                            )
                        except Exception as e:
                            print(f"XTTS generation error: {e}")
                            audio_data, sample_rate, output_file_path = None, None, ""
                    else:
                        print("XTTS not available, skipping audio generation.")
                        audio_data, sample_rate, output_file_path = None, None, ""

                    # If TTS succeeded, try RVC conversion when available
                    enqueued_path = None
                    if audio_data is not None and output_file_path:
                        if model_pth is not None:
                            try:
                                converted_file = convert_voice(
                                    tts_wav_path=output_file_path,
                                    model_pth_path=model_pth,
                                    f0_up_key=-8,
                                )
                                if converted_file and os.path.exists(converted_file):
                                    print(f"Converted voice saved to: {converted_file}")
                                    enqueued_path = converted_file
                                    # Delete original TTS when conversion succeeds
                                    try:
                                        if os.path.exists(output_file_path):
                                            os.remove(output_file_path)
                                            print(f"Deleted original TTS file: {output_file_path}")
                                    except Exception as e:
                                        print(f"Warning: Could not delete original TTS file {output_file_path}: {e}")
                                else:
                                    print("Voice conversion failed, using original TTS audio")
                                    enqueued_path = output_file_path
                            except Exception as e:
                                print(f"RVC error: {e}")
                                enqueued_path = output_file_path
                        else:
                            print("RVC model not available, using TTS audio directly")
                            enqueued_path = output_file_path

                    if enqueued_path:
                        # queue absolute or relative path; handler supports both
                        completed_sentence = sentences[-1]
                        sentences.append("")
                        idx += 1
                        sentences_queue.put((conversation[-1]["content"], completed_sentence, enqueued_path))
            else:
                print("Ollama not available, skipping AI chat.")
                
        conversation.append({"role": "assistant", "content": "".join(sentences)})
        sentences_queue.join()
    else:
        time.sleep(1)
        continue
    break

# Clear user/AI text files at end
for fname in ("user_input.txt", "ai_response.txt"):
    with open(fname, "w", encoding="utf-8") as f:
        pass 