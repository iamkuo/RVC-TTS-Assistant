# --- Imports ---
import os
import sys
import threading
import queue
import torch
import soundfile as sf
from os.path import join as dirjoin
import time

# Set environment variables for PyTorch and DeepSpeed
os.environ["TORCH_WEIGHTS_ONLY"] = "False"

# 修復 PyTorch 2.6 相容性問題

print("Python executable:", sys.executable)
print(f"PyTorch version: {torch.__version__}")
# Add local workspace root to sys.path for local modules (e.g., rvc_python)
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

# Optional/third-party modules
from TTS.api import TTS as XTTS
import sounddevice as sd
import ollama

# Add local RVC-beta to path BEFORE importing rvc_infer (contains dependencies)
rvc_beta_path = os.path.join(WORKSPACE_ROOT, "RVC-beta0717")
if rvc_beta_path not in sys.path:
    sys.path.insert(0, rvc_beta_path)

# (Removed unused rvc_infer import and src/rvc-tts-pipe path insertion)

# Import AI processing helpers directly
from xtts_rvc_infer import (
    process_text,
    convert_voice,
)

# --- Audio Processing Functions ---
# (All processing functions are now imported directly from xtts_rvc_infer)

# --- Config ---
voice = 'Lisa'
rvc_model_path = "D:/program/RVC-TTS-PIPELINE/RVCModels/"
rvc_model_name = 'lisa-genshin'
prompt_file_path = "D:/program/RVC-TTS-PIPELINE/RVCPrompts/"
prompt_file_name = 'Lisa_Prompt'
generated_sounds_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"
ollama_model_name = "mistral"

# RVC settings
rvc_f0_key = -8  # Pitch adjustment for voice conversion
audio_path = generated_sounds_path
sentences_queue = queue.Queue()
new_line_keywords = [".","?","!"]
stop_keywords = ["stop" , "nope" , "good bye"] #lower case
reset_keywords = ["reset","new","next"]
first = True

def output_handler():
    while True:
        user_text, ai_generated, filename = sentences_queue.get()
        # filename is basename; reconstruct full path
        file_path = filename if os.path.isabs(filename) else dirjoin(audio_path, filename)
        print(f"Playing: {file_path}")
        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait()
        print(f"Finished: {file_path}")
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        sentences_queue.task_done()

# --- Main logic ---
# Initialize XTTS model
print("Initializing XTTS model...")
try:
    tts = XTTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=torch.cuda.is_available()
    )
    print("XTTS model initialized")
except Exception as e:
    print(f"Failed to initialize XTTS model: {str(e)}")
    sys.exit(1)

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resolve RVC model .pth path (used by rvc_python.infer under the hood)
model_pth = os.path.join(rvc_model_path, f"{rvc_model_name}.pth")
if os.path.exists(model_pth):
    print(f"Using RVC model: {model_pth}")
else:
    print(f"Warning: RVC model not found at {model_pth}. Voice conversion will be disabled.")
    model_pth = None

# Initialize audio device
os.makedirs(generated_sounds_path, exist_ok=True)
try:
    current_devices = sd.query_devices()
    print("Available audio devices:")
    for i, device in enumerate(current_devices):
        print(f"{i}: {device['name']} (Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']})")
    
    # Try to find a suitable output device (first device with output channels)
    output_device = None
    for i, device in enumerate(current_devices):
        if device['max_output_channels'] > 0:
            output_device = i
            break
    
    if output_device is not None:
        print(f"Using audio output device {output_device}: {current_devices[output_device]['name']}")
        sd.default.device = (sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device, 
                           output_device)
    else:
        print("Warning: No suitable audio output device found")
        
except Exception as e:
    print(f"Warning: Could not set up audio device: {str(e)}")
    print("Audio output may not work correctly")

# Resolve speaker reference
_speaker_wav = ""
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

with open(dirjoin(prompt_file_path, prompt_file_name + ".txt"), encoding="utf-8") as f:
    prompt = f.read()

conversation = []
tts_thread = threading.Thread(target=output_handler, daemon=True)
tts_thread.start()

while True:
    message = input("user input:")
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
    idx = 0
    stream = ollama.chat(
        model=ollama_model_name,
        messages=conversation, stream=True)
    print("user input:" + conversation[-1]["content"])
    for chunk in stream:
        chunk = chunk["message"]["content"]
        sentences[-1] += chunk
        # Inlined sentence_boundary_hit logic
        if any(k in chunk for k in new_line_keywords) or (sentences[-1] and any(sentences[-1].endswith(k) for k in new_line_keywords)):
            try:
                # Generate TTS audio
                audio_data, sample_rate, output_file_path = process_text(
                    text=sentences[-1],
                    speaker_wav=_speaker_wav,
                    tts=tts,
                    output_dir=generated_sounds_path,
                )
                
                if audio_data is not None and output_file_path and model_pth is not None:
                    try:
                        # Convert the just-saved TTS file with RVC
                        converted_file = convert_voice(
                            tts_wav_path=output_file_path,
                            model_pth_path=model_pth,
                            f0_up_key=rvc_f0_key,
                        )
                        if converted_file and os.path.exists(converted_file):
                            sentences_queue.put((
                                conversation[-1]["content"],
                                sentences[-1],
                                converted_file
                            ))
                            print(f"Converted voice saved to: {converted_file}")
                            # Delete the original TTS file since we'll use the converted one
                            try:
                                if output_file_path and os.path.exists(output_file_path):
                                    os.remove(output_file_path)
                                    print(f"Deleted original TTS file: {output_file_path}")
                            except Exception as e:
                                print(f"Warning: Could not delete original TTS file {output_file_path}: {e}")
                        else:
                            # Fallback to original TTS if conversion fails
                            print("Voice conversion failed, using original TTS audio")
                            sentences_queue.put((
                                conversation[-1]["content"],
                                sentences[-1],
                                output_file_path
                            ))
                    except Exception as e:
                        print(f"Error in voice conversion: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to original TTS on error
                        sentences_queue.put((
                            conversation[-1]["content"],
                            sentences[-1],
                            output_file_path
                        ))
                elif audio_data is not None and output_file_path:
                    # If RVC model is not available, use the original TTS audio
                    if model_pth is None:
                        print("RVC model not available, using TTS audio directly")
                    sentences_queue.put((
                        conversation[-1]["content"],
                        sentences[-1],
                        output_file_path
                    ))
                else:
                    print("Failed to generate TTS audio")
                
                # Start a new sentence
                sentences.append("")
                idx += 1
                
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                import traceback
                traceback.print_exc()
                # Ensure we start a new sentence even if there was an error
                sentences.append("")
    
    conversation.append({"role": "assistant", "content": "".join(sentences)})
    sentences_queue.join()
