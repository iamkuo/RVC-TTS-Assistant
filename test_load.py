import torch
print(torch.cuda.is_available())
import ffmpeg
from pydub import AudioSegment
from pydub.playback import play