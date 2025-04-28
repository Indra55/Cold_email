import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import PyPDF2
import io
import re
import spacy
from chain import Chain
from utils import clean_text
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Extract information from uploaded PDF resumes
def extract_resume_info(pdf_file):
    nlp = spacy.load("en_core_web_sm")
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    doc = nlp(text)
    info = {"name": "", "email": "", "skills": [], "experience": []}

    # Extract name from the contact section
    contact_section = text[:500]
    doc_contact = nlp(contact_section)
    for ent in doc_contact.ents:
        if ent.label_ == "PERSON":
            info["name"] = ent.text
            break
    
    # Extract email using regex
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        info["email"] = emails[0]
    
    # Identify skills
    common_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "HTML", "CSS", "SQL",
        "React", "Node.js", "Express", "Flask", "Flutter", "NumPy", "Pandas", 
        "TensorFlow", "PyTorch", "Matplotlib", "BeautifulSoup", "Scikit-learn",
        "TailwindCSS", "Streamlit", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
        "MongoDB", "PostgreSQL", "Firebase", "REST", "GraphQL", "CI/CD", "DevOps", "Machine Learning", "AI"
    ]
    text_lower = text.lower()
    info["skills"] = [skill for skill in common_skills if skill.lower() in text_lower]

    # Extract experience information
    experience_pattern = r"(\d+)\s*(year|yr|yrs)\s*(?:at|with)\s*([A-Za-z\s]+)"
    experience_matches = re.findall(experience_pattern, text, re.IGNORECASE)
    if experience_matches:
        for match in experience_matches:
            years, _, company = match
            info["experience"].append(f"{years} years at {company.strip()}")

    return info

# Main Streamlit app
def create_streamlit_app(llm, clean_text):
    # Page title and layout
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("📧 Cold Connect")
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stFileUploader>div>div>button {
            background-color: #008CBA;
            color: white;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Input fields
    st.header("Your Details")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email address")
            company = st.text_input("Company Name", placeholder="Enter the company name")
            experience = st.text_area("Work Experience", placeholder="e.g., '2 years at Google'")
        with col2:
            designation = st.text_input("Designation", placeholder="Your role (e.g., Developer)")
            resume_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
        job_url = st.text_input("Job URL", placeholder="Paste the job URL here")
        submit_button = st.form_submit_button("Generate Email")

    # Process resume and generate email
    if submit_button:
        if not (name and email and company and designation):
            st.error("Please fill in all required fields")
        else:
            if resume_file:
                with st.spinner("Processing your resume..."):
                    resume_info = extract_resume_info(resume_file)
                    st.success("Resume processed successfully!")
                    st.write(f"**Name:** {resume_info['name'] or 'N/A'}")
                    st.write(f"**Email:** {resume_info['email'] or 'N/A'}")
                    st.write("**Skills:** " + ", ".join(resume_info['skills']) if resume_info['skills'] else "None listed")
                    st.write("**Experience:** " + ", ".join(resume_info['experience']) if resume_info['experience'] else "None listed")

            with st.spinner("Generating your email..."):
                try:
                    loader = WebBaseLoader([job_url])
                    job_data = clean_text(loader.load().pop().page_content)
                    jobs = llm.extract_jobs(job_data)
                    for job in jobs:
                        user_info = {
                            "name": name,
                            "email": email,
                            "company": company,
                            "designation": designation,
                            "experience": experience.split(", ")
                        }
                        email_content = llm.write_mail(job, user_info)
                        st.subheader("Generated Email")
                        st.code(email_content, language="markdown")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Entry point
if __name__ == "__main__":
    from chain import Chain
    chain = Chain()
    create_streamlit_app(chain, clean_text)
