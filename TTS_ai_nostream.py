# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from rvc_python.infer import RVCInference

import os
from os.path import join as dirjoin

#import wave
import pyaudio

import ollama

import time
#import multiprocessing
import threading
import queue
import sounddevice as sd
import soundfile as sf
import sys  # 加入 sys 模組

voice = 'train_kennard'
rvc_model_path = "D:/program/RVC-TTS-PIPELINE/RVCModels/"
rvc_model_name = 'lisa-genshin'
prompt_file_path = "D:/program/RVC-TTS-PIPELINE/RVCPrompts/"
prompt_file_name = 'Lisa_Prompt'
generated_sounds_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"
ollama_model_name = "mistral"

sd.default.device = "CABLE Input (VB-Audio Virtual Cable), Windows DirectSound"
audio_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"
sentences_queue = queue.Queue()

new_line_keywords = [".","?","!"]
stop_keywords = ["stop" , "nope" , "good bye"] #lower case
reset_keywords = ["reset","new","next"]

first = True

with open("user_input.txt", "w", encoding="utf-8") as user_file: pass
with open("ai_response.txt", "w", encoding="utf-8") as user_file: pass

def output_handler():
    while True:
        user_text,ai_generated, filename = sentences_queue.get()  # 取得佇列中的任務
        file_path = dirjoin(audio_path,filename)
        with open("user_input.txt", "w", encoding="utf-8") as user_file:
            user_file.write("User:" + user_text)
            user_file.flush()
        with open("ai_response.txt","w",encoding="utf-8") as ai_file:
            ai_file.write(ollama_model_name+":")  # 持續寫入 AI 回應
            ai_file.flush()  # 確保 OBS 能即時讀取
        print(f"Playing: {file_path}")
        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait()
        print(f"Finished: {file_path}")
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        sentences_queue.task_done()  # 標記任務完成

# This will download all the models used by Tortoise from the HF hub.
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(device="cuda:0",use_deepspeed=True, kv_cache=True)

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
#preset = "fast"

# Pick one of the voices from the output above

#text = 'Hello you have reached the voicemail of myname, please leave a message'
# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)


rvc = RVCInference(device="cuda:0")
rvc.f0up_key = -8
rvc.load_model(rvc_model_path + rvc_model_name + ".pth")


with open("RVCPrompts/" + prompt_file_name+ ".txt",encoding="utf-8") as f: prompt = f.read()
conversation = []

tts_thread = threading.Thread(target=output_handler, daemon=True)
tts_thread.start()

while True:
    message = input("user input:")
    #handle one input sentence
    if message in stop_keywords: break
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
    messages=conversation,stream=True)
    print("user input:" + conversation[-1]["content"])
    #connect ollama output stream into sentences and generate output
    for chunk in stream:
        chunk = chunk["message"]["content"]
        sentences[-1] += chunk
        if(chunk in new_line_keywords):
            #generate audio output
            tortoise_generated_path = dirjoin(generated_sounds_path, "tortoise_generated_" + str(idx) + ".wav")
            output_path = dirjoin(generated_sounds_path, "output" + str(idx) + ".wav")
            gen = tts.tts(sentences[-1], voice_samples=voice_samples, conditioning_latents=conditioning_latents,verbose=False)
            torchaudio.save(tortoise_generated_path, gen.squeeze(0).cpu(), 24000)
            print("completed tortoise generation")
            rvc.infer_file(tortoise_generated_path, output_path)
            print("completed RVC generation")
            
            #handle output
            sentences.append("")
            idx += 1
            sentences_queue.put((conversation[-1]["content"], sentences[-1], output_path))
    
    conversation.append({"role": "assistant", "content": "".join(sentences)})
    sentences_queue.join()
with open("user_input.txt", "w", encoding="utf-8") as user_file: pass
with open("ai_response.txt", "w", encoding="utf-8") as user_file: pass