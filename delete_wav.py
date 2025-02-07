import os
import glob

# 取得當前腳本所在目錄
directory = "/mnt/d/RVC-TTS-PIPELINE/generated_sounds"

# 找到所有包含 'output' 或 'tortoise_generated' 的 .wav 檔案
files = glob.glob(os.path.join(directory, "*output*.wav")) + \
        glob.glob(os.path.join(directory, "*tortoise_generated*.wav"))

for file in files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")
