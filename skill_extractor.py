import streamlit as st
import pdfplumber
from huggingface_hub import snapshot_download
import spacy
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_model():
    model_path = snapshot_download("amjad-awad/skill-extractor", repo_type="model")
    return spacy.load(model_path)

@st.cache_resource
def load_vlm_model():
    return pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

vlm = load_vlm_model()
nlp = load_model()

# Convert PDF to text
def pdf_to_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def pdf_to_images(file):
    Images = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=150).original
            Images.append(img)
    return Images

st.title("📄 Resume Skill Extractor")
st.write("Upload a resume (PDF) and extract **skills** using a pre-trained Hugging Face model.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text_input = st.text_area("Or paste resume text here:", height=200)

resume_text = ""

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        resume_text = pdf_to_text(uploaded_file)
elif resume_text_input.strip():
    resume_text = resume_text_input

if resume_text.strip():
    st.subheader("📜 Resume Text Preview")
    st.text_area("Extracted Text", resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text, height=200)

    with st.spinner("Identifying skills using SLM (spaCy)..."):
        doc = nlp(resume_text)
        skills = [ent.text for ent in doc.ents if ent.label_.upper() == "SKILLS"]

    st.subheader("🛠 Extracted Skills (SLM)")
    if skills:
        unique_skills = sorted(set(skills))
        st.success(", ".join(unique_skills))
        st.download_button("⬇️ Download Skills", "\n".join(unique_skills), "skills.txt")
    else:
        st.warning("No skills detected by SLM.")
elif uploaded_file is not None:
    st.warning("No text found in PDF. Trying VLM (BLIP captioning)...")

    with st.spinner("Extracting text from resume images using VLM..."):
        images = pdf_to_images(uploaded_file)
        vlm_text =""
        for img in images:
            vlm_text += vlm(Image.fromarray(img))[0]["generated_text"] + "\n"
        
        doc = nlp(vlm_text)
        skills = [ent.text for ent in doc.ents if ent.label_.upper() == "SKILLS"]    
        if skills:
            unique_skills = sorted(set(skills))
            st.success(", ".join(unique_skills))
            st.download_button("⬇️ Download Skills", "\n".join(unique_skills), "skills.txt")
        else:
            st.warning("No skills detected by VLM either.")


# if uploaded_file is not None:
#     with st.spinner("Extracting text from resume..."):
#         resume_text = pdf_to_text(uploaded_file)

#     st.subheader("📜 Resume Text Preview")
#     st.text_area("Extracted Text", resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text, height=200)

#     with st.spinner("Identifying skills..."):
#         doc = nlp(resume_text)
#         skills = [ent.text for ent in doc.ents if ent.label_.upper() == "SKILLS"]

#     st.subheader("🛠 Extracted Skills")
#     if skills:
#         st.success(", ".join(set(skills)))
#     else:
#         st.warning("No skills were detected in this resume.")

# import streamlit as st
# import pdfplumber
# from transformers import pipeline

# # -------------------------------
# # Load LLM (small model for prompt engineering)
# # -------------------------------
# @st.cache_resource
# def load_llm():
#     return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

# llm = load_llm()

# # -------------------------------
# # PDF → Text
# # -------------------------------
# def pdf_to_text(file):
#     text = ""
#     with pdfplumber.open(file) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ""
#     return text

# # -------------------------------
# # Prompt for skill extraction
# # -------------------------------
# def extract_skills_with_prompt(resume_text):
#     prompt = f"""
#     You are a resume analyzer in a large IT company.
#     The purpose is to check whether the employee is worth hiring or not.
#     Extract only the professional and technical SKILLS from the following resume text.
#     Return them as a comma-separated list, nothing else.

#     Resume:
#     {resume_text}
#     """
#     response = llm(prompt, max_new_tokens=200, do_sample=False)
#     return response[0]["generated_text"].split("Resume:")[-1].strip()

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("📄 Resume Skill Extractor (Prompt Engineering)")
# st.write("Upload a resume (PDF) or paste text. Skills are extracted using prompt engineering with an LLM.")

# uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
# resume_text_input = st.text_area("Or paste resume text here:", height=200)

# resume_text = ""

# if uploaded_file is not None:
#     with st.spinner("Extracting text from PDF..."):
#         resume_text = pdf_to_text(uploaded_file)

# elif resume_text_input.strip():
#     resume_text = resume_text_input

# if resume_text:
#     st.subheader("📜 Resume Text Preview")
#     st.text_area("Extracted Text", resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text, height=200)

#     with st.spinner("Extracting skills with LLM..."):
#         skills = extract_skills_with_prompt(resume_text)

#     st.subheader("🛠 Extracted Skills")
#     if skills:
#         st.success(skills)
#         st.download_button("⬇️ Download Skills", skills, "skills.txt")
#     else:
#         st.warning("No skills detected.")
