#!/usr/bin/env python3
"""
測試腳本：驗證所有組件是否正常工作
"""

import os
import sys
from datetime import datetime

def test_components():
    """測試所有組件"""
    print("=== 組件測試 ===")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路徑: {sys.executable}")
    print(f"工作目錄: {os.getcwd()}")
    print(f"當前時間: {datetime.now()}")
    
    # 測試基本模組
    print("\n--- 基本模組測試 ---")
    try:
        import sounddevice as sd
        print("✓ sounddevice 導入成功")
    except ImportError as e:
        print(f"✗ sounddevice 導入失敗: {e}")
    
    try:
        import soundfile as sf
        print("✓ soundfile 導入成功")
    except ImportError as e:
        print(f"✗ soundfile 導入失敗: {e}")
    
    try:
        import pyaudio
        print("✓ pyaudio 導入成功")
    except ImportError as e:
        print(f"✗ pyaudio 導入失敗: {e}")
    
    try:
        import ollama
        print("✓ ollama 導入成功")
    except ImportError as e:
        print(f"✗ ollama 導入失敗: {e}")
    
    # 測試 TTS
    print("\n--- TTS 測試 ---")
    try:
        from TTS.api import TTS as XTTS
        print("✓ TTS 導入成功")
    except ImportError as e:
        print(f"✗ TTS 導入失敗: {e}")
    
    # 測試 RVC
    print("\n--- RVC 測試 ---")
    try:
        # 添加 RVC 路徑
        rvc_tts_pipe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "rvc-tts-pipe")
        if rvc_tts_pipe_path not in sys.path:
            sys.path.insert(0, rvc_tts_pipe_path)
        
        from rvc_infer import rvc_convert   # type: ignore
        print("✓ RVC 導入成功")
    except ImportError as e:
        print(f"✗ RVC 導入失敗: {e}")
    
    # 測試檔案權限
    print("\n--- 檔案權限測試 ---")
    test_file = "test_permission.txt"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("測試檔案\n")
        print(f"✓ 成功創建檔案: {test_file}")
        
        os.remove(test_file)
        print(f"✓ 成功刪除檔案: {test_file}")
    except Exception as e:
        print(f"✗ 檔案權限測試失敗: {e}")
    
    # 測試目錄權限
    print("\n--- 目錄權限測試 ---")
    generated_sounds_path = "generated_sounds"
    try:
        os.makedirs(generated_sounds_path, exist_ok=True)
        print(f"✓ 成功創建/訪問目錄: {generated_sounds_path}")
    except Exception as e:
        print(f"✗ 目錄權限測試失敗: {e}")
    
    print("\n=== 測試完成 ===")
    return True

if __name__ == "__main__":
    test_components() 