# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

from rvc_python.infer import RVCInference

import os

import wave
import pyaudio

import ollama

"""
def get_audio_files(directory, extensions=(".wav", ".mp3", ".flac", ".ogg")):
    #掃描指定資料夾，取得所有符合副檔名的音檔路徑。
    #:param directory: 要掃描的資料夾路徑
    #:param extensions: 允許的音檔副檔名 (預設包含常見格式)
    #:return: 包含音檔完整路徑的清單
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

# 指定要掃描的資料夾
folder_path = "your/audio/folder/path"  # 這裡換成你的資料夾路徑
# 取得音檔列表
audio_file_list = get_audio_files(folder_path)
"""

# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(use_deepspeed=True, kv_cache=True)

# This is the text that will be spoken.
text = "Joining two modalities results in a surprising increase in generalization! What would happen if we combined them all?"

# Here's something for the poetically inclined.. (set text=)
"""
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,"""

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
#preset = "fast"

# Pick one of the voices from the output above
voice = 'Noelle'
text = 'Hello you have reached the voicemail of myname, please leave a message'
# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)

rvc = RVCInference(device="cuda:0")
rvc.f0up_key = -8

prompt_file_name = 'Lisa_Prompt'
with open("RVCPrompts/" + prompt_file_name+ ".txt") as f: question_1 = f.read()
conversation = [{"role": "user", "content": question_1+input("User Input:")}]


while True:
    if conversation[-1]["content"] == "Stop": break
    conversation.append(ollama.chat(
    model="mistral",
    messages=conversation,stream=False))
    gen = tts.tts(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents)
    #gen = torch.cat(list(gen))
    torchaudio.save('tortoise_generated.wav', gen.squeeze(0).cpu(), 24000)
    print("completed tortoise generation")
    rvc.load_model("RVCModels/lisa-genshin.pth")
    rvc.infer_file("tortoise_generated.wav", "output.wav")
    conversation.append({"role": "user", "content": input("User Input:")})


    


