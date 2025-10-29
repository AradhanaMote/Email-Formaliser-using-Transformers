import streamlit as st
from transformers import pipeline

# Try a different, more reliable model
@st.cache_resource
def load_model():
    try:
        # Use a different paraphrasing model
        return pipeline(
            "text2text-generation", 
            model="humarin/chatgpt_paraphraser_on_T5_base",
            device=-1
        )
    except:
        try:
            # Fallback to general T5 model
            return pipeline("text2text-generation", model="t5-base")
        except:
            return None

paraphraser = load_model()

def formalize_email(text):
    if paraphraser is None:
        return rule_based_formalizer(text)
    
    # Try different prompts with the new model
    prompts = [
        f"paraphrase this in a formal business tone: {text}",
        f"rewrite this as a professional email: {text}",
        f"make this sound more formal and professional: {text}",
        f"convert this casual text to formal business language: {text}"
    ]
    
    for prompt in prompts:
        try:
            result = paraphraser(
                prompt,
                max_length=128,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=2.0,
                num_beams=5
            )
            generated_text = result[0]['generated_text']
            
            # Check if output is different
            if (generated_text.strip().lower() != text.strip().lower() and 
                len(generated_text) > len(text) * 0.5):
                return generated_text
        except Exception as e:
            continue
    
    # If model fails, use rule-based approach
    return rule_based_formalizer(text)

def rule_based_formalizer(text):
    """Fallback rule-based formalizer"""
    import re
    
    # Common informal to formal replacements
    replacements = {
        r'\bhey\b': 'Hello',
        r'\bhi\b': 'Hello',
        r'\bu\b': 'you',
        r'\bur\b': 'your',
        r'\basap\b': 'at your earliest convenience',
        r'\bpls\b': 'please',
        r'\bplease\b': 'kindly',
        r'\bthx\b': 'thank you',
        r'\bthanks\b': 'thank you',
        r'\bcan u\b': 'Could you',
        r'\bcould u\b': 'Could you',
        r'\bwanna\b': 'would like to',
        r'\bgonna\b': 'going to',
        r'\bgr8\b': 'great',
        r'\bbtw\b': 'additionally',
        r'\bimo\b': 'in my opinion',
        r'\brn\b': 'right now',
        r'\bomg\b': '',
        r'\blol\b': '',
        r'!\s*$': '.',  # Remove exclamation at end
    }
    
    formal_text = text.lower()
    
    # Apply replacements
    for informal, formal in replacements.items():
        formal_text = re.sub(informal, formal, formal_text)
    
    # Capitalize first letter and ensure proper punctuation
    formal_text = formal_text.capitalize()
    if not formal_text.endswith(('.', '!', '?')):
        formal_text += '.'
    
    # Remove multiple spaces
    formal_text = re.sub(r'\s+', ' ', formal_text).strip()
    
    return formal_text

# Streamlit interface
st.title("Email Formalizer using Transformers")
st.write("Enter casual/informal text below to get a formal version.")

user_input = st.text_area("Casual Input:", value="hey can u send me the report asap?")

if st.button("Formalize"):
    if user_input:
        with st.spinner("Formalizing your text..."):
            formal_text = formalize_email(user_input)
        
        st.subheader("Formalized Email:")
        st.write(formal_text)
        
        # Show which method was used
        if paraphraser is None:
            st.info("ℹ️ Using rule-based formalizer (AI model unavailable)")
        elif formal_text == rule_based_formalizer(user_input):
            st.info("ℹ️ Using rule-based formalizer (AI model didn't produce good results)")
        else:
            st.success("✅ Using AI-powered formalization")
            
    else:
        st.warning("Please enter some text to formalize.")

# Add some examples
st.subheader("Examples:")
examples = {
    "hey can u send me the report asap?": "Hello, could you please send me the report at your earliest convenience?",
    "im gonna need that file by tomorrow": "I will require that file by tomorrow.",
    "thx for the help!": "Thank you for your assistance.",
    "pls review the doc and lmk what u think": "Kindly review the document and let me know your thoughts."
}

for informal, formal in examples.items():
    with st.expander(f"'{informal}' → '{formal}'"):
        st.write(f"**Formal:** {formal}")