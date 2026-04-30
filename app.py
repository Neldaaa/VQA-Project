import os
import gradio as gr
from google import genai
import json
import requests
import base64
from PIL import Image
import logging
from vqa_evssm_integration import preprocess_for_vqa
logger = logging.getLogger(__name__)

# ==========================================
# 1. EVSSM IMAGE PROCESSING IMPORT
# ==========================================
try:
    from vqa_evssm_integration import preprocess_for_vqa
    EVSSM_AVAILABLE = True
    print("✓ EVSSM integration loaded successfully")
except Exception as e:
    EVSSM_AVAILABLE = False
    print(f"⚠ EVSSM not available: {str(e)}. VQA will work without image preprocessing.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. API CONFIGURATION
# ==========================================
API_KEY = "API_KEY"  # Replace with your API key
client = genai.Client(api_key=API_KEY)

# ==========================================
# 3. DATASET LOADING (FALLBACK TO MOCK DATA FOR TESTING)
# ==========================================
try:
    with open("data/all.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    image_keys = list(dataset.keys())
except FileNotFoundError:
    print("⚠️ Could not find data/all.json. Using mock data for interface testing...")
    dataset = {
        "TEST_001": {
            "question": "Describe this image in detail.",
            "image_url": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=400",
            "long_answers": {
                "Expert": {"answer_paragraph": "This is a sample expert answer. Please load the actual dataset."},
                "Gemini": {"answer_paragraph": "This is a sample AI answer stored in the file."}
            }
        }
    }
    image_keys = list(dataset.keys())

# ==========================================
# 4. AI PROCESSING FUNCTIONS
# ==========================================
def evaluate_offline(image_id):
    """Evaluate using offline dataset"""
    data = dataset[image_id]
    try:
        image = Image.open(requests.get(data["image_url"], stream=True).raw).convert('RGB')
    except:
        image = None
    expert = data["long_answers"].get("Expert", {}).get("answer_paragraph", "")
    gemini = data["long_answers"].get("Gemini", {}).get("answer_paragraph", "")
    return image, data["question"], expert, gemini

def live_demo(image, question, shortQ):
    """
    Live VQA demo with optional EVSSM image preprocessing
    
    Args:
        image: PIL Image from user upload/webcam
        question: Detailed question
        shortQ: Short answer question
    
    Returns:
        tuple: (detailed_answer, short_answer)
    """
    if image is None:
        return "Please upload an image first.", ""
    
    # ============================================================
    # 🔥 NEW: APPLY EVSSM IMAGE PREPROCESSING (DEBLURRING)
    # ============================================================
    processed_image = image
    if EVSSM_AVAILABLE:
        try:
            logger.info("🔧 Applying EVSSM image preprocessing (deblurring)...")
            processed_image = preprocess_for_vqa(image)
            logger.info("✓ Image preprocessing completed successfully")
        except Exception as e:
            logger.warning(f"⚠ Image preprocessing failed, using original image: {str(e)}")
            processed_image = image
    else:
        logger.info("ℹ EVSSM not available, using original image for VQA")
    
    # ============================================================
    # PREPARE PROMPTS
    # ============================================================
    prompt = f"Question: {question}. Provide a detailed answer. If blurry or unclear, please still try your best to answer."
    short_prompt = f"Question: {shortQ}. Provide a short and simple answer in 1-2 sentences."

    # ============================================================
    # SEND TO GEMINI API WITH PROCESSED IMAGE
    # ============================================================
    try:
        logger.info("📤 Sending to Gemini API for VQA analysis...")
        
        # Generate detailed response
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[prompt, processed_image]
        )
        
        # Generate short response
        short_response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[short_prompt, processed_image]
        )
        
        logger.info("✓ Responses received from Gemini")
        return response.text, short_response.text
        
    except Exception as e:
        error_msg = f"API Error: {str(e)}"
        logger.error(error_msg)
        return error_msg, error_msg

# ==========================================
# 5. PAGE NAVIGATION FUNCTIONS
# ==========================================
def go_to_main():
    """Navigate from welcome to main app"""
    return gr.update(visible=False), gr.update(visible=True)

def go_to_welcome():
    """Navigate from main app to welcome"""
    return gr.update(visible=True), gr.update(visible=False)


def img_to_base64(path):
    """Read an image file and return a base64 string for embedding in HTML."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = path.split(".")[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return ""

PIC1 = img_to_base64("pic1.png")
PIC2 = img_to_base64("pic2.png")
PIC3 = img_to_base64("pic3.png")
PIC4 = img_to_base64("pic4.png")
PIC5 = img_to_base64("pic5.png")
PIC6 = img_to_base64("pic6.png")

# ==========================================
# 6. UI STYLING (CSS)
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Space+Mono:wght@400;700&display=swap');

/* GLOBAL BACKGROUND & FONT */
body, .gradio-container { 
    background-color: #F7F5F0 !important; 
    font-family: 'Space Mono', monospace !important; 
}

/* WELCOME SCREEN */
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

/* ENTER APP BUTTON */
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

/* LISTEN BUTTON */
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

/* FLOATING SCRAPBOOK ITEMS */
.scrapbook-item {
    position: absolute;
    font-size: 50px;
    transition: transform 0.5s;
    filter: drop-shadow(2px 5px 10px rgba(0,0,0,0.18));
}
.scrapbook-item:hover { transform: scale(1.15) rotate(5deg); z-index: 50; }

/* BOTTOM TEXT BLOCK */
.bottom-block {
    position: absolute;
    top: 250px;
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

/* MAIN APP */
#main-app { padding: 20px; }
h1 { font-family: 'DM Serif Display', serif !important; font-size: 40px !important; color: #1A1A1A !important; margin-bottom: 0px !important;}
p.subtitle { font-family: 'Space Mono', monospace; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0px;}

/* BACK BUTTON */
#btn-back { background: transparent !important; border: 2px solid #1A1A1A !important; border-radius: 30px !important; color: #1A1A1A !important; width: 180px !important; margin-bottom: 20px !important; }
#btn-back:hover { background: #1A1A1A !important; color: #FFF !important; }

/* INPUT FIELDS */
.gr-box, .gr-block, input, textarea, .gr-dropdown {
    background: #FFF !important;
    border: 2px solid #D6DCE4 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    color: #1A1A1A !important;
}

textarea, input::placeholder {
    color: #1A1A1A !important;
    -webkit-text-fill-color: #1A1A1A !important;
    opacity: 1 !important;
}

.block-label { color: #1A1A1A !important; font-weight: bold !important; }

/* TABS */
.tabs button { color: #6B7280 !important; font-weight: bold !important; }
.tabs button.selected { color: #1A1A1A !important; border-bottom: 2px solid #6594B1 !important; }
.tabs button:hover { background-color: #6594B1 !important; color: #1A1A1A !important; }
.tabs + div { border: 1px solid #6594B1 !important; background-color: #FFFFFF !important; border-radius: 0 0 8px 8px !important; }

button.primary { background: #1A1A1A !important; color: #FFF !important; border-radius: 30px !important; font-weight: bold !important; }

#welcome-wrapper.hide { display: none !important; }

/* TTS STATUS INDICATOR */
#tts-status {
    font-size: 12px;
    color: #6B7280;
    font-family: 'Space Mono', monospace;
    margin-top: 6px;
    min-height: 18px;
}
"""

welcome_html = f"""
<img class="scrapbook-item" style="top: 120%; left: 5%; width: 100px;" src="{PIC1}">
<img class="scrapbook-item" style="bottom: 100%; left: 28%; width: 85px;" src="{PIC2}">
<img class="scrapbook-item" style="bottom: 50%; left: 70%; width: 100px;" src="{PIC3}">
<img class="scrapbook-item" style="bottom: 100%; left: 47%; width: 140px;" src="{PIC4}">
<img class="scrapbook-item" style="top: 110%; right: 1%; width: 125px;" src="{PIC5}">
<img class="scrapbook-item" style="top: 650%; right: 18%; width: 65px;" src="{PIC6}">
<div class="scrapbook-item" style="top: 700%; left: 23%; transform: rotate(-5deg);">🔑</div>
<div class="bottom-block">
    <div class="tagline">This is a Visual Question Answering system powered by AI. <br>Upload any image, ask a question, and get a detailed answer instantly.</div>
    <div class="site-title">
        <span class="pre">welcome to our</span>
        <span class="main">"VQA Assistant"</span>
    </div>
</div>
"""

# JS: Manual intro TTS (Listen button)
js_tts = """
function() {
    let text = "Welcome to VQA Assistant. This is a Visual Question Answering system powered by AI. Upload any image, ask a question, and get a detailed answer instantly. If you want to enter the app, press the enter button.";
    let msg = new SpeechSynthesisUtterance(text);
    msg.lang = 'en-US';
    msg.rate = 0.9;
    let voices = window.speechSynthesis.getVoices();
    let femaleVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Zira') || v.name.includes('Samantha'));
    if (femaleVoice) msg.voice = femaleVoice;
    window.speechSynthesis.speak(msg);
}
"""

# JS: Auto-play TTS on first user interaction (page load)
onload_js = """
function() {
    window.welcomeSpeech = null;
    let hasSpoken = false;

    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = function() {
        window.speechSynthesis.getVoices();
    };

    function playAudioOnFirstTouch(e) {
        if (hasSpoken) return;
        let text = "Welcome to our project. This is a Visual Question Answering system powered by AI. Upload any image, ask a question, and get a detailed answer instantly. If you want to enter the app, press the enter button.";
        window.welcomeSpeech = new SpeechSynthesisUtterance(text);
        window.welcomeSpeech.lang = 'en-US';
        window.welcomeSpeech.rate = 0.9;
        let voices = window.speechSynthesis.getVoices();
        let femaleVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Zira') || v.name.includes('Samantha') || v.name.includes('Female'));
        if (femaleVoice) window.welcomeSpeech.voice = femaleVoice;
        window.speechSynthesis.speak(window.welcomeSpeech);
        hasSpoken = true;
        document.removeEventListener('click', playAudioOnFirstTouch);
        document.removeEventListener('keydown', playAudioOnFirstTouch);
    }

    document.addEventListener('click', playAudioOnFirstTouch);
    document.addEventListener('keydown', playAudioOnFirstTouch);

    // Spacebar shortcut to enter the app
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            let btnEnter = document.querySelector('#btn-enter');
            if (btnEnter) btnEnter.click();
        }
    });
}
"""

# JS: Auto-read AI response aloud after it is generated
js_auto_tts = """
function(response) {
    if (!response || response.trim() === "") return response;

    // Stop any currently playing speech
    window.speechSynthesis.cancel();

    let msg = new SpeechSynthesisUtterance(response);
    msg.lang = 'en-US';
    msg.rate = 0.9;

    // Try to select a female voice
    let voices = window.speechSynthesis.getVoices();
    let femaleVoice = voices.find(v =>
        v.name.includes('Google US English') ||
        v.name.includes('Zira') ||
        v.name.includes('Samantha')
    );
    if (femaleVoice) msg.voice = femaleVoice;

    // Update status indicator
    let statusEl = document.getElementById('tts-status');
    if (statusEl) statusEl.innerText = '🔊 Reading answer...';
    msg.onend = function() {
        if (statusEl) statusEl.innerText = '✅ Done reading.';
    };

    window.speechSynthesis.speak(msg);
    return response;
}
"""

# ==========================================
# 7. GRADIO APP INITIALIZATION
# ==========================================
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:

    # --- SCREEN 1: WELCOME PAGE ---
    with gr.Column(visible=True, elem_id="welcome-wrapper") as welcome_col:
        gr.HTML(welcome_html, elem_id="html-layer")
        btn_enter = gr.Button("Enter App →", elem_id="btn-enter")
        btn_speak = gr.Button("🔊 Listen to Intro", elem_id="btn-speak")
        btn_speak.click(fn=None, js=js_tts)

    # --- SCREEN 2: MAIN APPLICATION ---
    with gr.Column(visible=False, elem_id="main-app") as main_col:
        btn_back = gr.Button("← Back to Home", elem_id="btn-back")
        gr.HTML("<h1>Visual Question Answering</h1><p class='subtitle'>AI-Powered Image Analysis System</p>")
        
        # Display EVSSM status
        status_text = "✓ With EVSSM Image Enhancement" if EVSSM_AVAILABLE else "⚠ Without Image Enhancement"
        status_color = "#22c55e" if EVSSM_AVAILABLE else "#f97316"
        gr.HTML(f"<p style='color: {status_color}; font-size: 12px; margin-top: -10px;'>{status_text}</p>")

        with gr.Tab("Dataset Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    dropdown = gr.Dropdown(choices=image_keys[:50], label="Select Image ID", value=image_keys[0] if image_keys else None)
                    img_off = gr.Image(type="pil", interactive=False, label="Input Image")
                with gr.Column(scale=2):
                    q_off = gr.Textbox(label="Question", lines=2)
                    ans_exp = gr.Textbox(label="Expert Answer (Ground Truth)", lines=5)
                    ans_gem = gr.Textbox(label="AI Model Answer", lines=5)
            dropdown.change(fn=evaluate_offline, inputs=dropdown, outputs=[img_off, q_off, ans_exp, ans_gem])

        with gr.Tab("Live Camera"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_live = gr.Image(type="pil", sources=["webcam", "upload"], label="Upload or Capture Image")
                    q_live = gr.Textbox(label="Your Question", value="Please describe what is in this image in detail.", lines=2)
                    short_q_live = gr.Textbox(label="Short Answer Question", value="What is the main subject?", lines=1)
                    btn_live = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    ans_live = gr.Textbox(label="AI Response", lines=12)
                    # TTS status indicator
                    gr.HTML('<div id="tts-status"></div>')
                    short_ans_live = gr.Textbox(label="Short Answer", lines=4)

            # Click Analyze -> get AI response
            btn_live.click(
                fn=live_demo,
                inputs=[img_live, q_live, short_q_live],
                outputs=[ans_live, short_ans_live]
            )
            # Response updated -> auto-read aloud
            ans_live.change(
                fn=None,
                inputs=[ans_live],
                outputs=[ans_live],
                js=js_auto_tts
            )

    # Page navigation events
    btn_enter.click(fn=go_to_main, inputs=None, outputs=[welcome_col, main_col])
    btn_back.click(fn=go_to_welcome, inputs=None, outputs=[welcome_col, main_col], js=onload_js)
    demo.load(fn=None, js=onload_js)

if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.abspath(".")])
