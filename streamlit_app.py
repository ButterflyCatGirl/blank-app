import streamlit as st
from PIL import Image
import torch
import io
import base64
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time
from medical_terms import get_medical_translation, get_medical_response_template

# Configure page
st.set_page_config(
    page_title="AI Health Assistant - مساعد الصحة الذكي",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Arabic support and medical theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', 'Noto Sans Arabic', sans-serif;
}

.arabic-text {
    font-family: 'Noto Sans Arabic', sans-serif;
    direction: rtl;
    text-align: right;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    max-width: 80%;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: auto;
    margin-right: 0;
}

.bot-message {
    background-color: #f5f5f5;
    margin-left: 0;
    margin-right: auto;
}

.medical-disclaimer {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "processor" not in st.session_state:
    st.session_state.processor = None
if "model" not in st.session_state:
    st.session_state.model = None

# Language toggle
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

t = translations[st.session_state.language]
is_rtl = st.session_state.language == "ar"

# Header
if is_rtl:
    st.markdown(f'<h1 class="arabic-text">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="arabic-text">{t["subtitle"]}</p>', unsafe_allow_html=True)
else:
    st.title(t["title"])
    st.markdown(t["subtitle"])

# Medical Disclaimer
disclaimer_class = "medical-disclaimer arabic-text" if is_rtl else "medical-disclaimer"
st.markdown(f'<div class="{disclaimer_class}">{t["disclaimer"]}</div>', unsafe_allow_html=True)

# Load BLIP2 model
@st.cache_resource
def load_blip2_model():
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model
if not st.session_state.model_loaded:
    with st.spinner(t["loading_model"]):
        processor, model = load_blip2_model()
        if processor and model:
            st.session_state.processor = processor
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ AI Model loaded successfully!")
        else:
            st.error("❌ Failed to load AI model")

# Chat interface
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    message_class = "chat-message user-message" if message["role"] == "user" else "chat-message bot-message"
    if message.get("is_arabic", False):
        message_class += " arabic-text"
    
    st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)
    
    if "image" in message:
        st.image(message["image"], width=300)

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    # Text input
    text_input = st.text_input(
        t["text_input"],
        key="text_input",
        label_visibility="collapsed"
    )

# Image upload
uploaded_file = st.file_uploader(
    t["upload_prompt"],
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=False
)

# Send button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    send_button = st.button(t["send"], use_container_width=True)

# Clear chat button
with col3:
    if st.button(t["clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Process input
if send_button and (text_input or uploaded_file):
    # Detect language
    def detect_language(text):
        arabic_pattern = r'[\u0600-\u06FF]'
        import re
        return "ar" if re.search(arabic_pattern, text) else "en"
    
    detected_lang = detect_language(text_input) if text_input else st.session_state.language
    is_arabic = detected_lang == "ar"
    
    # Add user message
    user_message = {
        "role": "user",
        "content": text_input if text_input else "صورة طبية مرفقة" if is_arabic else "Medical image uploaded",
        "is_arabic": is_arabic
    }
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        user_message["image"] = image
    
    st.session_state.messages.append(user_message)
    
    # Generate response
    with st.spinner(t["analyzing"]):
        try:
            if uploaded_file and st.session_state.model_loaded:
                # BLIP2 image analysis
                image = Image.open(uploaded_file).convert('RGB')
                
                # Medical context prompts
                medical_prompts = {
                    "en": "This is a medical image. Describe what you see in detail, including any abnormalities, anatomical structures, and potential medical findings. Be specific about medical terminology.",
                    "ar": "هذه صورة طبية. صف ما تراه بالتفصيل، بما في ذلك أي تشوهات أو هياكل تشريحية أو نتائج طبية محتملة. كن محدداً في المصطلحات الطبية."
                }
                
                prompt = medical_prompts[detected_lang]
                
                inputs = st.session_state.processor(image, prompt, return_tensors="pt")
                
                # Generate response
                with torch.no_grad():
                    generated_ids = st.session_state.model.generate(**inputs, max_length=200, num_beams=5)
                    generated_text = st.session_state.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Translate medical terms if Arabic
                if detected_lang == "ar":
                    response = get_medical_translation(generated_text, detected_lang)
                else:
                    response = generated_text
                
                # Add medical context
                response += "\n\n" + get_medical_response_template(detected_lang)
                
            else:
                # Text-only response
                response = get_medical_response_template(detected_lang, text_input)
            
            # Add bot response
            bot_message = {
                "role": "assistant",
                "content": response,
                "is_arabic": detected_lang == "ar"
            }
            st.session_state.messages.append(bot_message)
            
        except Exception as e:
            error_msg = f"خطأ في التحليل: {str(e)}" if is_arabic else f"Analysis error: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "is_arabic": is_arabic
            })
    
    st.rerun()

# Footer
st.markdown("---")
footer_text = "🏥 مساعد الصحة الذكي - تطوير بالذكاء الاصطناعي" if is_rtl else "🏥 AI Health Assistant - Powered by AI"
if is_rtl:
    st.markdown(f'<p class="arabic-text" style="text-align: center;">{footer_text}</p>', unsafe_allow_html=True)
else:
    st.markdown(f'<p style="text-align: center;">{footer_text}</p>', unsafe_allow_html=True)
