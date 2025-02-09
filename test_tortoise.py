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



# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(device="cuda:0",use_deepspeed=True, kv_cache=True)

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

rvc_model_name = 'lisa-genshin'
rvc = RVCInference(device="cuda:0")
rvc.f0up_key = -8
rvc.load_model("RVCModels/" + rvc_model_name + ".pth")

prompt_file_name = 'Lisa_Prompt'
with open("RVCPrompts/" + prompt_file_name+ ".txt") as f: question_1 = f.read()
conversation = [{"role": "user", "content": question_1+input("User Input:")}]

generated_sounds_path = "generated_sounds/"

ollama_model_name = "mistral"

while True:
    with open("user_input.txt", "w", encoding="utf-8") as user_file: user_file.write("User:" + conversation[-1]["content"])
    if conversation[-1]["content"] == "Stop": break
    sentences = [""]
    idx = 0
    
    stream = ollama.chat(
    model=ollama_model_name,
    messages=conversation,stream=True)
    print(conversation[-1]["content"])
    with open("ai_response.txt","w",encoding="utf-8") as ai_file:
        ai_file.write(ollama_model_name+":")  # 持續寫入 AI 回應
        ai_file.flush()  # 確保 OBS 能即時讀取
        for chunk in stream:
            chunk = chunk["message"]["content"]
            ai_file.write(chunk)  # 持續寫入 AI 回應
            ai_file.flush()  # 確保 OBS 能即時讀取
            sentences[-1] += chunk
            if(chunk == "."):
                gen = tts.tts(sentences[-1], voice_samples=voice_samples, conditioning_latents=conditioning_latents,verbose=False)
                torchaudio.save(generated_sounds_path + "tortoise_generated_" + str(idx) + ".wav", gen.squeeze(0).cpu(), 24000)
                print("completed tortoise generation")
                rvc.infer_file(generated_sounds_path + 'tortoise_generated_' + str(idx) + '.wav',generated_sounds_path + "output" + str(idx) + ".wav")
                print("completed RVC generation")
                sentences.append("")
                idx += 1
                ai_file.seek(0)
                ai_file.truncate()
                ai_file.flush()
    conversation.append({"role": "assistant", "content": "".join(sentences)})
    conversation.append({"role": "user", "content": input("User Input:")})


    


