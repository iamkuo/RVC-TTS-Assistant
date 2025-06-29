# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

from rvc_python.infer import RVCInference

import os
from os.path import join as dirjoin

import wave
import pyaudio

import ollama

import time
#import multiprocessing
import threading
import queue
import sounddevice as sd
import soundfile as sf
import sys  # 加入 sys 模組
import pytchat
print("yay")