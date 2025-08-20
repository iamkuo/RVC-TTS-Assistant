import os

# 清理 generated_sounds 目錄
generated_sounds_path = "D:/program/RVC-TTS-PIPELINE/generated_sounds/"

if os.path.exists(generated_sounds_path):
    for item in os.listdir(generated_sounds_path):
        item_path = os.path.join(generated_sounds_path, item)
        try:
            if os.path.isdir(item_path):
                os.rmdir(item_path)
                print(f"已刪除目錄: {item}")
            else:
                os.remove(item_path)
                print(f"已刪除檔案: {item}")
        except Exception as e:
            print(f"無法刪除 {item}: {e}")

print("清理完成！") 