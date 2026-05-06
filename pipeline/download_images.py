import json, requests, os
from pathlib import Path

# Đọc file data của LFVQA
with open('/home/nhan/Workspaces/VQA-Project/data/all.json') as f:
    data = json.load(f)

# Folder chứa ảnh tải về
out = Path('data/images_original')
out.mkdir(parents=True, exist_ok=True)

downloaded = 0
limit = 150  # Giới hạn 150 ảnh

for entry_id, entry in data.items():
    if downloaded >= limit:
        break  # Dừng vòng lặp khi đủ 150 ảnh
    # Lấy URL ảnh từ data
    url = entry.get('image_url', '')
    if not url:
        continue

    # Lấy tên file từ URL
    fname = url.split('/')[-1]
    dest = out / fname

    # Bỏ qua nếu đã tải rồi
    if dest.exists():
        continue

    try:
        r = requests.get(url, timeout=15)
        dest.write_bytes(r.content)
        downloaded += 1
        print(f'[{downloaded}] Tải xong: {fname}')
    except Exception as e:
        print(f'Lỗi {fname}: {e}')

print(f'\nHoàn thành! Tải được {downloaded} ảnh.')