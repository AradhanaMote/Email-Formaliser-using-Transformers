import streamlit as st
from transformers import pipeline

# Load paraphrasing model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

paraphraser = load_model()

def formalize_email(text):
    prompt = f"paraphrase: {text} </s>"
    result = paraphraser(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
    return result[0]['generated_text']

# Streamlit interface
st.title("Email Formalizer using Transformers")
st.write("Enter casual/informal text below to get a formal version.")

user_input = st.text_area("Casual Input:")

if user_input:
    formal_text = formalize_email(user_input)
    st.subheader("Formalized Email:")
    st.write(formal_text)
