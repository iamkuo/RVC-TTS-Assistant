import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydub import AudioSegment

# 設定監控的資料夾
watch_folder = "D:\RVC-TTS-PIPELINE\generated_sounds"  # 改成你要監控的資料夾路徑

# 播放音訊的函式
def play_audio(filename):
    print(f"Playing {filename}...")
    audio = AudioSegment.from_wav(filename)  # 使用 pydub 讀取 WAV 檔案
    samples = np.array(audio.get_array_of_samples())  # 轉換為 numpy 陣列
    rate = audio.frame_rate  # 音訊的採樣率
    sd.play(samples, rate)  # 播放音訊
    sd.wait()  # 等待音訊播放完成
    print("Playback finished.")
    
    # 播放完成後刪除檔案
    os.remove(filename)
    print(f"Deleted {filename}.")

# 監控新檔案加入的事件處理器
class WatcherHandler(FileSystemEventHandler):
    def __init__(self):
        self.is_playing = False  # 控制是否正在播放

    def on_created(self, event):
        # 檢查是否為新增的 .wav 檔案
        if event.is_directory:
            return
        if event.src_path.endswith('.wav'):
            if not self.is_playing:  # 如果正在播放音訊，則不處理
                self.is_playing = True
                play_audio(event.src_path)
                self.is_playing = False

# 設定監控器並啟動監控
def start_watching():
    event_handler = WatcherHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)  # 每秒檢查一次資料夾
            print("yee")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 開始監控
start_watching()
