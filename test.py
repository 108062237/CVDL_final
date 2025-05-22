import os
import pandas as pd

# 讀取 CSV
df = pd.read_csv('data/train.csv')

# 取得 image_id 清單
image_ids_from_csv = set(df['image_id'])

# 取得資料夾中所有的 .tiff 檔名（去除副檔名）
image_ids_from_folder = set([
    fname.replace('.tiff', '')
    for fname in os.listdir('data/train_images')
    if fname.endswith('.tiff')
])

# 檢查 CSV 中有但資料夾沒有的 image_id
missing_in_folder = image_ids_from_csv - image_ids_from_folder

# 檢查資料夾中有但 CSV 沒有的 image_id
extra_in_folder = image_ids_from_folder - image_ids_from_csv

# 結果輸出
print(f"✅ Total CSV image_ids: {len(image_ids_from_csv)}")
print(f"✅ Total folder image_ids: {len(image_ids_from_folder)}")
print(f"❌ Missing in folder: {len(missing_in_folder)}")
if missing_in_folder:
    print("   Example missing:", list(missing_in_folder)[:5])

print(f"❌ Extra in folder: {len(extra_in_folder)}")
if extra_in_folder:
    print("   Example extra:", list(extra_in_folder)[:5])
