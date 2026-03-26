import gradio as gr
from google import genai
from google.genai import errors
import json
import requests
from PIL import Image

# 1. Cấu hình Gemini API (THAY BẰNG KEY MỚI CỦA BẠN TẠI ĐÂY)
client = genai.Client(api_key="AIzaSyD8sEoGPrRI5RZSSv6FadfC7mpLe-XaRIU")

# 2. Đọc dữ liệu từ file all.json
with open("data/all.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
image_keys = list(dataset.keys())

def evaluate_image(image_id):
    data = dataset[image_id]
    question = data["question"]
    image_url = data["image_url"]
    expert_answer = data["long_answers"]["Expert"]["answer_paragraph"]
    
    # Tải ảnh
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    prompt = f"Question: {question}. Please provide a detailed, long-form answer. If the image is too blurry to determine, say 'I cannot answer'."
    
    # Yêu cầu Gemini phân tích an toàn
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt, image]
        )
        ai_answer = response.text
    except errors.ClientError as e:
        ai_answer = f"Lỗi API: Đã hết lượt dùng miễn phí hoặc Key bị lỗi. Vui lòng tạo Key mới!"
    except Exception as e:
        ai_answer = f"Lỗi không xác định: {str(e)}"
    
    return image, question, expert_answer, ai_answer

# 3. Tạo giao diện web
demo = gr.Interface(
    fn=evaluate_image,
    inputs=[gr.Dropdown(choices=image_keys[:20], label="Chọn ID Ảnh để test (1-20)", value=image_keys[0])],
    outputs=[
        gr.Image(type="pil", label="Ảnh chụp bởi người khiếm thị"),
        gr.Textbox(label="Câu hỏi ban đầu"),
        gr.Textbox(label="Câu trả lời của Chuyên gia"),
        gr.Textbox(label="Câu trả lời của AI (Gemini)")
    ],
    title="Đánh giá mô hình VQA (Mô phỏng VizWiz-LF)"
)

if __name__ == "__main__":
    demo.launch()