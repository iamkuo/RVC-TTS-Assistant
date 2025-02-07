import os

# 獲取當前程式碼所在的資料夾
current_folder = os.path.dirname(os.path.abspath(__file__))

# 指定要查找的子資料夾名稱
target_subfolder_name = "generated_sounds"  # 改為你想查找的子資料夾名稱

# 獲得指定名稱的子資料夾路徑
target_subfolder_path = None
for subfolder in os.listdir(current_folder):
    subfolder_path = os.path.join(current_folder, subfolder)
    if os.path.isdir(subfolder_path) and subfolder == target_subfolder_name:
        target_subfolder_path = subfolder_path
        break

# 顯示結果
if target_subfolder_path:
    print(f"找到指定的子資料夾: {target_subfolder_path}")
else:
    print(f"未找到名為 '{target_subfolder_name}' 的子資料夾。")