import json
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# 1. Tải mô hình AI cục bộ (BLIP)
print("Đang tải mô hình AI...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 2. Đọc dữ liệu gốc từ bài báo
print("Đang đọc dữ liệu all.json...")
with open("data/all.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 3. Đánh giá thử trên 3 câu hỏi đầu tiên
count = 0
for image_id, data in dataset.items():
    if count >= 3: # Đổi số này để test nhiều ảnh hơn
        break
        
    question = data["question"]
    image_url = data["image_url"]
    expert_answer = data["long_answers"]["Expert"]["answer_paragraph"]
    
    # Tải ảnh từ URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    except:
        continue

    # Yêu cầu AI cục bộ trả lời (Thêm prompt ép AI trả lời chi tiết hoặc từ chối)
    prompt = f"Question: {question} Answer in detail. If the image is too blurry, say 'I cannot answer'."
    inputs = processor(image, prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    ai_answer = processor.decode(out[0], skip_special_tokens=True)

    # In kết quả so sánh
    print(f"\n--- Ảnh ID: {image_id} ---")
    print(f"Câu hỏi: {question}")
    print(f"Chuyên gia (Ground Truth): {expert_answer}")
    print(f"AI Cục bộ (BLIP): {ai_answer}")
    
    count += 1