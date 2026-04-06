import sys
import os

print("=== 環境檢查 ===")
print(f"Python 執行檔: {sys.executable}")
print(f"Python 版本: {sys.version}")

# 檢查是否使用正確的 conda 環境
if ".conda" not in sys.executable:
    print("警告: 未使用 conda 環境的 Python!")
    print("請執行: conda activate rvc-tts-pipeline")

# 添加 rvc_python 路徑到 Python 路徑
conda_path = os.path.join(os.getcwd(), ".conda", "Lib", "site-packages")
if conda_path not in sys.path:
    sys.path.insert(0, conda_path)
    print(f"已添加路徑: {conda_path}")

print("\n=== 測試依賴模組 ===")
# 先測試 cffi
try:
    import _cffi_backend
    print("✓ _cffi_backend 匯入成功")
except ImportError as e:
    print(f"✗ _cffi_backend 匯入失敗: {e}")
    print("請執行: pip install cffi")

# 測試 rvc_python
try:
    import rvc_python
    print("✓ rvc_python 模組匯入成功")
    
    from rvc_python.infer import RVCInference
    print("✓ RVCInference 匯入成功")
    
    # 測試建立實例
    rvc = RVCInference(device="cpu:0")
    print("✓ RVCInference 實例建立成功")
    
except ImportError as e:
    print(f"✗ 匯入失敗: {e}")
except Exception as e:
    print(f"✗ 其他錯誤: {e}")

print("\n=== 建議解決方案 ===")
print("1. 確保使用 conda 環境: conda activate rvc-tts-pipeline")
print("2. 安裝缺少的依賴: pip install cffi pycparser")
print("3. 重新安裝 rvc_python: pip install --force-reinstall rvc_python")