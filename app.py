import os
import gradio as gr
from google import genai
import json
import requests
import base64
from PIL import Image

# ==========================================
# 1. CẤU HÌNH API
# ==========================================
API_KEY = "YOUR_API_KEY_HERE" # Nhớ thay mã của bạn vào
client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. XỬ LÝ LỖI DATASET (DỮ LIỆU GIẢ LẬP ĐỂ TEST)
# ==========================================
try:
    with open("data/all.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    image_keys = list(dataset.keys())
except FileNotFoundError:
    print("⚠️ Không tìm thấy file data/all.json. Đang sử dụng Dữ liệu giả lập để test giao diện...")
    # Tạo một dữ liệu giả nếu không có file thật để web không bị sập
    dataset = {
        "TEST_001": {
            "question": "Mô tả chi tiết bức ảnh này?",
            "image_url": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=400", # Ảnh chó con
            "long_answers": {
                "Expert": {"answer_paragraph": "Đây là câu trả lời mẫu của chuyên gia vì bạn chưa tải dataset."},
                "Gemini": {"answer_paragraph": "Đây là câu trả lời mẫu của AI lưu trong file."}
            }
        }
    }
    image_keys = list(dataset.keys())

# ==========================================
# 3. HÀM XỬ LÝ AI
# ==========================================
def evaluate_offline(image_id):
    data = dataset[image_id]
    try:
        image = Image.open(requests.get(data["image_url"], stream=True).raw).convert('RGB')
    except:
        image = None
    expert = data["long_answers"].get("Expert", {}).get("answer_paragraph", "")
    gemini = data["long_answers"].get("Gemini", {}).get("answer_paragraph", "")
    return image, data["question"], expert, gemini

def live_demo(image, question):
    if image is None: return "Vui lòng tải ảnh."
    prompt = f"Question: {question}. Provide a detailed answer. If blurry, say 'I cannot answer'."
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, image])
        return response.text
    except Exception as e:
        return f"Lỗi API: {str(e)}"

# Hàm chuyển trang
def go_to_main():
    return gr.update(visible=False), gr.update(visible=True)
def go_to_welcome():
    return gr.update(visible=True), gr.update(visible=False)


def img_to_base64(path):
    """Đọc file ảnh và trả về chuỗi base64 để nhúng vào HTML"""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        # Tự detect đuôi file để chọn đúng mime type
        ext = path.split(".")[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file: {path}")
        return ""  # Trả về rỗng nếu không có file

# Convert tất cả ảnh cutout của bạn
PIC1 = img_to_base64("pic1.png")
PIC2 = img_to_base64("pic2.png")
PIC3 = img_to_base64("pic3.png")
PIC4 = img_to_base64("pic4.png")
PIC5 = img_to_base64("pic5.png")
PIC6 = img_to_base64("pic6.png")

# ==========================================
# 4. GIAO DIỆN SIÊU ĐẸP (CSS & HTML THEO BẢN THIẾT KẾ CỦA BẠN)
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Space+Mono:wght@400;700&display=swap');

/* NỀN VÀ FONT TỔNG THỂ */
body, .gradio-container { 
    background-color: #F7F5F0 !important; 
    font-family: 'Space Mono', monospace !important; 
}

/* ------------------- TRANG CHÀO MỪNG ------------------- */
#welcome-wrapper {
    background-color: #F7F5F0 !important;
    background-image: linear-gradient(#D6DCE4 1px, transparent 1px), linear-gradient(90deg, #D6DCE4 1px, transparent 1px) !important;
    background-size: 44px 44px !important;
    height: 85vh !important;
    position: relative;
    border-radius: 12px;
    border: 1px solid #D6DCE4;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

/* NÚT ENTER APP (Giữa màn hình) */
#btn-enter {
    background: #1A1A1A !important;
    color: #fff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 15px 40px !important;
    border-radius: 50px !important;
    box-shadow: 4px 4px 0 #E63946 !important;
    transition: 0.2s !important;
    border: none !important;
    z-index: 100;
    width: auto !important;
}
#btn-enter:hover { transform: translate(-2px, -2px) !important; box-shadow: 6px 6px 0 #E63946 !important; }
#btn-enter:active { transform: translate(2px, 2px) !important; box-shadow: 2px 2px 0 #E63946 !important; }

/* NÚT NGHE GIỌNG NÓI */
#btn-speak {
    background: #fff !important;
    color: #1A1A1A !important;
    border: 2px solid #1A1A1A !important;
    border-radius: 50px !important;
    box-shadow: 3px 3px 0 #C9CDD4 !important;
    margin-top: 15px !important;
    width: auto !important;
    z-index: 100;
}
#btn-speak:hover { transform: translate(-1px,-1px) !important; box-shadow: 5px 5px 0 #C9CDD4 !important; }

/* VĂN BẢN VÀ ICON BAY LƠ LỬNG */
.scrapbook-item {
    position: absolute;
    font-size: 50px;
    transition: transform 0.5s;
    filter: drop-shadow(2px 5px 10px rgba(0,0,0,0.18));
}
.scrapbook-item:hover { transform: scale(1.15) rotate(5deg); z-index: 50; }

.bottom-block {
    position: absolute;
    bottom: -300px; /* Giảm từ 30px xuống 10px hoặc 5px */
    left: 30px;
    right: 30px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    pointer-events: none;
}
.tagline {
    font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.12em; line-height: 1.7;
}
.site-title { text-align: right; line-height: 1; }
.site-title .pre { display: block; font-size: 24px; color: #6B7280; letter-spacing: -0.02em; }
.site-title .main { display: block; font-family: 'DM Serif Display', serif; font-size: 70px; color: #1A1A1A; font-style: italic; letter-spacing: -0.03em; line-height: 0.9;}

/* ------------------- TRANG CHÍNH APP ------------------- */
#main-app { padding: 20px; }
h1 { font-family: 'DM Serif Display', serif !important; font-size: 40px !important; color: #1A1A1A !important; margin-bottom: 0px !important;}
p.subtitle { font-family: 'Space Mono', monospace; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0px;}

#btn-back { background: transparent !important; border: 2px solid #1A1A1A !important; border-radius: 30px !important; color: #1A1A1A !important; width: 180px !important; margin-bottom: 20px !important;}
#btn-back:hover { background: #1A1A1A !important; color: #FFF !important; }

/* Form nhập liệu */
.gr-box, .gr-block, input, textarea, .gr-dropdown {
    background: #FFF !important;
    border: 2px solid #D6DCE4 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
button.primary { background: #1A1A1A !important; color: #FFF !important; border-radius: 30px !important; font-weight: bold !important; }
"""



# HTML trang trí bằng icon (Đã xoá sạch các mặt người cam kỳ lạ)
welcome_html = f"""
<img class="scrapbook-item" 
     style="top: 120%; left: 5%; transform: rotate(5deg); width: 100px;" 
     src="{PIC1}">

<img class="scrapbook-item" 
     style="bottom: 100%; left: 28%; transform: rotate(15deg); width: 85px;" 
     src="{PIC2}">

<img class="scrapbook-item" 
     style="bottom: 50%; left: 70%; transform: rotate(5deg); width: 100px;" 
     src="{PIC3}">

<img class="scrapbook-item" 
     style="bottom: 100%; left: 47%; width: 100px;" 
     src="{PIC4}">
     
<img class="scrapbook-item" 
     style="top: 110%; right: 1%; width: 125px;" 
     src="{PIC5}">

<img class="scrapbook-item" 
     style="top: 650%; right: 18%; width: 65px;" 
     src="{PIC6}">

<div class="scrapbook-item" style="top: 700%; left: 23%; transform: rotate(-5deg);">🔑</div>

<div class="bottom-block">
    <div class="tagline">This is a Visual Question Answering system powered by AI. <br>Upload any image, ask a question, and get a detailed answer instantly.</div>
    <div class="site-title">
        <span class="pre">welcome to our</span>
        <span class="main">"our project name" </span>
    </div>
</div>
"""

# JS Text-to-Speech
js_tts = """
function() {
    let msg = new SpeechSynthesisUtterance("Welcome to Our ... . This is a Visual Question Answering system powered by AI. Upload any image, ask a question, and get a detailed answer instantly. Click Enter App to start the Visual Question Answering system.");
    msg.lang = 'en-US';
    msg.rate = 0.9;
    window.speechSynthesis.speak(msg);
}
"""

# ==========================================
# 5. KHỞI TẠO GRADIO APP
# ==========================================
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    
    # --- MÀN HÌNH 1: CHÀO MỪNG (SCRAPBOOK) ---
    with gr.Column(visible=True, elem_id="welcome-wrapper") as welcome_col:
        # Lớp HTML nền và icon
        gr.HTML(welcome_html, elem_id="html-layer")
        
        # Các nút bấm ở giữa
        btn_enter = gr.Button("Enter App →", elem_id="btn-enter")
        btn_speak = gr.Button("🔊 Listen to Intro", elem_id="btn-speak")
        
        btn_speak.click(fn=None, js=js_tts)

    # --- MÀN HÌNH 2: ỨNG DỤNG CHÍNH ---
    with gr.Column(visible=False, elem_id="main-app") as main_col:
        btn_back = gr.Button("← Back to Home", elem_id="btn-back")
        
        gr.HTML("<h1>Visual Question Answering</h1><p class='subtitle'>AI-Powered Image Analysis System</p>")
        
        with gr.Tab("Dataset Evaluation (Offline)"):
            with gr.Row():
                with gr.Column(scale=1):
                    dropdown = gr.Dropdown(choices=image_keys[:50], label="Select Image ID", value=image_keys[0] if image_keys else None)
                    img_off = gr.Image(type="pil", interactive=False, label="Input Image")
                with gr.Column(scale=2):
                    q_off = gr.Textbox(label="Question", lines=2)
                    ans_exp = gr.Textbox(label="Expert Answer (Ground Truth)", lines=5)
                    ans_gem = gr.Textbox(label="AI Model Answer", lines=5)
            dropdown.change(fn=evaluate_offline, inputs=dropdown, outputs=[img_off, q_off, ans_exp, ans_gem])

        with gr.Tab("Live Camera Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_live = gr.Image(type="pil", sources=["webcam", "upload"], label="Upload or Capture Image")
                    q_live = gr.Textbox(label="Your Question", value="Please describe what is in this image in detail.", lines=2)
                    btn_live = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    ans_live = gr.Textbox(label="AI Response", lines=15)
            btn_live.click(fn=live_demo, inputs=[img_live, q_live], outputs=ans_live)

    # Sự kiện chuyển màn hình
    btn_enter.click(fn=go_to_main, inputs=None, outputs=[welcome_col, main_col])
    btn_back.click(fn=go_to_welcome, inputs=None, outputs=[welcome_col, main_col])

# Sửa launch — truyền thư mục chứa ảnh vào allowed_paths
if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.abspath(".")])