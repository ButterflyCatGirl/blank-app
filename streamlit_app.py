import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Configuration & State Setup
# -----------------------------
st.set_page_config(page_title="AI Health Assistant", layout="wide", initial_sidebar_state="collapsed")

# Language state
if "language" not in st.session_state:
    st.session_state.language = "en"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "processor" not in st.session_state:
    st.session_state.processor = None
if "model" not in st.session_state:
    st.session_state.model = None

# Language toggle button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🇪🇬 العربية" if st.session_state.language == "en" else "🇺🇸 English"):
        st.session_state.language = "ar" if st.session_state.language == "en" else "en"
        st.rerun()

# Translations
translations = {
    "en": {
        "title": "🏥 AI Health Assistant",
        "subtitle": "Your multilingual medical image analysis companion",
        "welcome": "Hello! I'm your AI health assistant. Upload a medical image or ask a health question.",
        "upload_prompt": "Upload a medical image (X-ray, CT scan, MRI, etc.)",
        "text_input": "Ask your health question...",
        "analyzing": "🔬 Analyzing your medical image...",
        "disclaimer": "⚠️ Medical Disclaimer: This AI assistant provides general information only and should not replace professional medical advice. Always consult with qualified healthcare professionals for medical decisions.",
        "loading_model": "🤖 Loading AI model... This may take a moment.",
        "send": "Send",
        "clear": "Clear Chat"
    },
    "ar": {
        "title": "🏥 مساعد الصحة الذكي",
        "subtitle": "رفيقك متعدد اللغات لتحليل الصور الطبية",
        "welcome": "مرحباً! أنا مساعدك الصحي الذكي. ارفع صورة طبية أو اسأل سؤالاً صحياً.",
        "upload_prompt": "ارفع صورة طبية (أشعة سينية، أشعة مقطعية، رنين مغناطيسي، إلخ)",
        "text_input": "اسأل سؤالك الصحي...",
        "analyzing": "🔬 جاري تحليل صورتك الطبية...",
        "disclaimer": "⚠️ إخلاء مسؤولية طبية: يقدم هذا المساعد الذكي معلومات عامة فقط ولا يجب أن يحل محل المشورة الطبية المهنية. استشر دائماً أخصائيي الرعاية الصحية المؤهلين لاتخاذ القرارات الطبية.",
        "loading_model": "🤖 جاري تحميل النموذج الذكي... قد يستغرق هذا بعض الوقت.",
        "send": "إرسال",
        "clear": "مسح المحادثة"
    }
}

# -----------------------------
# Model Loader (Cached)
# -----------------------------
@st.cache_resource
def load_blip2_model():
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 # Use half-precision to reduce memory usage
        )
        return processor, model
    except Exception as e:
        st.error(f"Model load error: {str(e)}. Ensure internet access and sufficient memory.")
        return None, None

# Load model if not already loaded
if not st.session_state.model_loaded:
    with st.spinner(translations[st.session_state.language]["loading_model"]):
        st.session_state.processor, st.session_state.model = load_blip2_model()
        if st.session_state.processor and st.session_state.model:
            st.session_state.model_loaded = True
            st.success("✅ AI Model loaded successfully!")

# -----------------------------
# Medical Response Logic
# -----------------------------
def get_medical_response(query, image=None, lang="en"):
    if image:
        # Medical prompt engineering
        prompt = {
            "en": "This is a medical image. Describe findings, abnormalities, and potential diagnosis.",
            "ar": "هذه صورة طبية. صف النتائج والتشوهات والتشخيص المحتمل."
        }[lang]
        
        image = Image.open(image).convert("RGB")
        inputs = st.session_state.processor(image, prompt, return_tensors="pt").to(st.session_state.model.device)
        with torch.no_grad():
            generated_ids = st.session_state.model.generate(**inputs, max_new_tokens=200)
            response = st.session_state.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Add disclaimer
        response += "\n\n⚠️ هذه النتائج تُعد معلومات عامة فقط، ويجب استشارة طبيب مختص قبل اتخاذ أي قرار طبي."
        return response
    else:
        # Text-based response template
        return {
            "en": f"Your question: '{query}'\n\n⚠️ These results are for informational purposes only. Always consult a qualified healthcare provider.",
            "ar": f"سؤالك: '{query}'\n\n⚠️ هذه النتائج تُعد معلومات عامة فقط، ويجب استشارة طبيب مختص قبل اتخاذ أي قرار طبي."
        }[lang]

# -----------------------------
# UI Rendering
# -----------------------------
t = translations[st.session_state.language]
is_rtl = st.session_state.language == "ar"

# Title
if is_rtl:
    st.markdown(f'<h1 class="arabic-text">{t["title"]}</h1>', unsafe_allow_html=True)
else:
    st.title(t["title"])

# Disclaimer
st.markdown(f'<div style="background:#fff3cd;padding:10px;border-radius:5px;">{t["disclaimer"]}</div>', unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    cls = "user" if msg["role"] == "user" else "assistant"
    style = "margin-left:auto;background:#e3f2fd;" if cls == "user" else "background:#f5f5f5;"
    if is_rtl:
        style += "direction:rtl;text-align:right;"
    st.markdown(f'<div style="{style}padding:10px;margin:5px 0;border-radius:8px;">{msg["content"]}</div>', unsafe_allow_html=True)
    if "image" in msg:
        st.image(msg["image"], width=300)

# Input area
col1, col2 = st.columns([4, 1])
with col1:
    user_text = st.text_input(t["text_input"], key="input_text")
with col2:
    uploaded_file = st.file_uploader(t["upload_prompt"], type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if st.button(t["send"]) and (user_text or uploaded_file):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text if user_text else ("صورة طبية" if is_rtl else "Medical Image"),
        "image": uploaded_file
    })

    # Generate response
    with st.spinner(t["analyzing"]):
        response = get_medical_response(user_text, uploaded_file, st.session_state.language)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
    st.rerun()

# Clear chat
if st.button(t["clear"]):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f'<p style="text-align:center;">🏥 AI Health Assistant | Powered by BLIP2</p>', unsafe_allow_html=True)
