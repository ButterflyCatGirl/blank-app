import streamlit as st
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import io
import base64
import json
from medical_terms import EGYPTIAN_MEDICAL_TERMS

# Configure Streamlit page
st.set_page_config(
    page_title="AI Medical Assistant - مساعد طبي ذكي",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_model():
    """Load BLIP2 model with caching for better performance"""
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def detect_language(text):
    """Detect if text contains Arabic characters"""
    arabic_pattern = r'[\u0600-\u06FF]'
    import re
    return "ar" if re.search(arabic_pattern, text) else "en"

def translate_medical_term(term, target_lang="ar"):
    """Translate medical terms to Egyptian Arabic"""
    if target_lang == "ar" and term.lower() in EGYPTIAN_MEDICAL_TERMS:
        return EGYPTIAN_MEDICAL_TERMS[term.lower()]
    return term

def analyze_medical_image(image, question, language="en"):
    """Analyze medical image using BLIP2 model"""
    processor, model = load_model()
    
    if processor is None or model is None:
        return get_error_message(language)
    
    try:
        # Prepare the image and question for BLIP2
        if question:
            prompt = f"Question: {question} Answer:"
        else:
            prompt = "Describe this medical image in detail:"
        
        # Process image and text
        inputs = processor(image, prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        
        # Decode response
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Clean up the response
        if "Answer:" in generated_text:
            response = generated_text.split("Answer:")[-1].strip()
        else:
            response = generated_text.strip()
        
        # Add medical disclaimer
        disclaimer = get_medical_disclaimer(language)
        
        # Translate to Arabic if needed
        if language == "ar":
            response = translate_to_arabic(response)
        
        return f"{response}\n\n{disclaimer}"
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return get_error_message(language)

def translate_to_arabic(text):
    """Basic medical translation to Egyptian Arabic"""
    # Simple keyword-based translation for common medical terms
    translations = {
        "x-ray": "أشعة سينية",
        "ct scan": "أشعة مقطعية",
        "mri": "رنين مغناطيسي",
        "bone": "عظم",
        "fracture": "كسر",
        "normal": "طبيعي",
        "abnormal": "غير طبيعي",
        "chest": "صدر",
        "lung": "رئة",
        "heart": "قلب",
        "brain": "مخ",
        "spine": "عمود فقري"
    }
    
    result = text.lower()
    for eng, ar in translations.items():
        result = result.replace(eng, ar)
    
    return result

def get_medical_disclaimer(language="en"):
    """Get medical disclaimer in appropriate language"""
    disclaimers = {
        "en": "⚠️ Medical Disclaimer: This analysis is for educational purposes only and should not replace professional medical consultation. Always consult with qualified healthcare professionals for medical advice.",
        "ar": "⚠️ إخلاء مسؤولية طبية: هذا التحليل لأغراض تعليمية فقط ولا يجب أن يحل محل الاستشارة الطبية المهنية. استشر دائماً أطباء مؤهلين للحصول على المشورة الطبية."
    }
    return disclaimers.get(language, disclaimers["en"])

def get_error_message(language="en"):
    """Get error message in appropriate language"""
    errors = {
        "en": "Sorry, I'm having trouble analyzing the image. Please try again later or consult with a healthcare professional.",
        "ar": "عذراً، أواجه مشكلة في تحليل الصورة. يرجى المحاولة مرة أخرى لاحقاً أو استشارة أخصائي رعاية صحية."
    }
    return errors.get(language, errors["en"])

def process_text_query(question, language="en"):
    """Process text-only medical queries"""
    responses = {
        "en": [
            "Thank you for your health question. For personalized medical advice, I recommend consulting with a healthcare professional who can properly assess your specific situation.",
            "I understand your concern about your health. While I can provide general information, it's important to speak with a qualified medical practitioner for proper diagnosis and treatment.",
            "Your health question is important. Please consider scheduling an appointment with a healthcare provider who can give you personalized medical guidance."
        ],
        "ar": [
            "شكراً لك على سؤالك الصحي. للحصول على استشارة طبية شخصية، أنصح بالتشاور مع أخصائي رعاية صحية يمكنه تقييم حالتك الخاصة بشكل صحيح.",
            "أفهم قلقك بشأن صحتك. بينما يمكنني تقديم معلومات عامة، من المهم التحدث مع ممارس طبي مؤهل للحصول على التشخيص والعلاج المناسب.",
            "سؤالك الصحي مهم. يرجى النظر في تحديد موعد مع مقدم الرعاية الصحية الذي يمكنه إعطاؤك إرشادات طبية شخصية."
        ]
    }
    
    import random
    lang_responses = responses.get(language, responses["en"])
    response = random.choice(lang_responses)
    
    disclaimer = get_medical_disclaimer(language)
    return f"{response}\n\n{disclaimer}"

# API Endpoints for React Frontend
if st.query_params.get("api") == "chat":
    # Handle API requests from React frontend
    if st.query_params.get("method") == "POST":
        try:
            # Get request data
            request_data = st.query_params.get("data")
            if request_data:
                data = json.loads(request_data)
                message = data.get("message", "")
                language = detect_language(message)
                
                # Process image if present
                if "image" in data:
                    # Decode base64 image
                    image_data = base64.b64decode(data["image"])
                    image = Image.open(io.BytesIO(image_data))
                    response = analyze_medical_image(image, message, language)
                else:
                    response = process_text_query(message, language)
                
                # Return JSON response
                st.json({
                    "response": response,
                    "language": language,
                    "status": "success"
                })
        except Exception as e:
            st.json({
                "error": str(e),
                "status": "error"
            })
    st.stop()

# Main Streamlit Interface
def main():
    # Header with Egyptian theme
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #DC143C 0%, #000000 50%, #FFD700 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 2.5rem; margin: 0;'>🏥 AI Medical Assistant</h1>
        <h2 style='color: #FFD700; font-size: 1.8rem; margin: 0.5rem 0;'>مساعد طبي ذكي</h2>
        <p style='color: white; font-size: 1.1rem; margin: 0;'>Advanced Medical Image Analysis with Egyptian Arabic Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selection
    language = st.selectbox(
        "Select Language / اختر اللغة",
        options=["en", "ar"],
        format_func=lambda x: "English 🇺🇸" if x == "en" else "العربية 🇪🇬",
        index=0
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Medical Image / ارفع الصورة الطبية",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload medical images like X-rays, CT scans, MRI images, etc."
    )
    
    # Text input
    if language == "ar":
        question = st.text_area(
            "اكتب سؤالك الطبي هنا:",
            placeholder="مثال: ما هو الموضح في هذه الأشعة السينية؟",
            height=100
        )
    else:
        question = st.text_area(
            "Ask your medical question:",
            placeholder="Example: What does this X-ray show?",
            height=100
        )
    
    # Process button
    if st.button("🔍 Analyze / تحليل", type="primary"):
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Medical Image", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing medical image... / جاري تحليل الصورة الطبية..."):
                    response = analyze_medical_image(image, question, language)
                    st.markdown("### Analysis Result / نتيجة التحليل")
                    st.write(response)
        
        elif question:
            with st.spinner("Processing your question... / جاري معالجة سؤالك..."):
                response = process_text_query(question, language)
                st.markdown("### Response / الرد")
                st.write(response)
        else:
            if language == "ar":
                st.warning("يرجى رفع صورة طبية أو كتابة سؤال.")
            else:
                st.warning("Please upload a medical image or ask a question.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🏥 AI Medical Assistant | مساعد طبي ذكي<br>"
        "Powered by BLIP2 & Streamlit | مدعوم بتقنية BLIP2 و Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
