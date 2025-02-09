import os
import time
import signal
import multiprocessing
import sounddevice as sd
import soundfile as sf
import sys  # 加入 sys 模組

sd.default.device = "CABLE Input (VB-Audio Virtual Cable), Windows DirectSound"
WATCH_FOLDER = "D:\RVC-TTS-PIPELINE\generated_sounds"
stop_flag = multiprocessing.Event()  # 使用 multiprocessing 的 Event 來控制結束
played_files = set()


def play_audio(file_path):
    print(f"Playing: {file_path}")
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()
    print(f"Finished: {file_path}")

    os.remove(file_path)
    played_files.remove(file_path)
    print(f"Deleted: {file_path}")

def watch_folder():

    while not stop_flag.is_set():
        files = [f for f in os.listdir(WATCH_FOLDER) if f.endswith(".wav") and "output" in f]
        files.sort()

        for file in files:
            if stop_flag.is_set():
                break
            file_path = os.path.join(WATCH_FOLDER, file)
            if file_path not in played_files:
                played_files.add(file_path)
                play_audio(file_path)
        
        time.sleep(1)

# 處理 `CTRL+C` 訊號
def signal_handler(sig, frame):
    print("\n偵測到 CTRL+C，正在停止程式...")
    stop_flag.set()  # 設定事件，通知 `watch_folder()` 停止
    sd.stop()  # 停止播放
    sys.exit(0)  # 確保程式完全終止

# 設定 `CTRL+C` 訊號監聽
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Watching folder for new .wav files... (按 CTRL+C 停止)")

    watch_process = multiprocessing.Process(target=watch_folder)
    watch_process.start()

    try:
        watch_process.join()  # 等待監測進程結束
    except KeyboardInterrupt:
        print("手動終止程式...")
        stop_flag.set()
        watch_process.terminate()
        watch_process.join()
        sys.exit(0)
