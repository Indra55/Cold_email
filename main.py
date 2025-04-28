import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import PyPDF2
import io
import re
from chain import Chain
from utils import clean_text

# --- Simple name extraction via regex/heuristics ---
def extract_name(text: str) -> str:
    # Look at the first few lines for a “Name-like” line:
    for line in text.splitlines()[:10]:
        # Remove extra spaces
        cand = line.strip()
        # If it’s at least two words, all capitalized (e.g. “John Doe”)
        parts = cand.split()
        if len(parts) >= 2 and all(p[0].isupper() and p[1:].islower() for p in parts if p.isalpha()):
            return cand
    return "Name not found"

def extract_resume_info(pdf_file) -> dict:
    # Read PDF text
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
    
    info = {"name": "", "email": "", "skills": [], "experience": []}
    
    # Name
    info["name"] = extract_name(text)
    
    # Email
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    emails = re.findall(email_pattern, text)
    info["email"] = emails[0] if emails else "Email not found"
    
    # Skills
    common_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "HTML", "CSS", "SQL",
        "React", "Node.js", "Express", "Flask", "Flutter", "NumPy", "Pandas",
        "TensorFlow", "PyTorch", "Matplotlib", "BeautifulSoup", "Scikit-learn",
        "TailwindCSS", "Streamlit", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
        "MongoDB", "PostgreSQL", "Firebase", "REST", "GraphQL", "CI/CD", "DevOps", "Machine Learning", "AI"
    ]
    lower = text.lower()
    info["skills"] = [s for s in common_skills if s.lower() in lower]
    
    # Experience (e.g. “2 years at Google”)
    exp_pattern = r"(\d+)\s*(?:year|yr|yrs)\s*(?:at|with)\s*([A-Za-z0-9 &]+)"
    matches = re.findall(exp_pattern, text, re.IGNORECASE)
    info["experience"] = [f"{yrs} years at {comp.strip()}" for yrs, comp in matches]
    
    return info

def create_streamlit_app(llm, clean_text_fn):
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

    if submit_button:
        if not (name and email and company and designation):
            st.error("Please fill in all required fields")
            return

        if resume_file:
            with st.spinner("Processing your resume..."):
                resume_info = extract_resume_info(resume_file)
            st.success("Resume processed!")

            st.write(f"**Name:** {resume_info['name']}")
            st.write(f"**Email:** {resume_info['email']}")
            st.write("**Skills:** " + (", ".join(resume_info['skills']) or "None listed"))
            st.write("**Experience:** " + (", ".join(resume_info['experience']) or "None listed"))

        with st.spinner("Generating your email..."):
            try:
                loader = WebBaseLoader([job_url])
                page = loader.load().pop()
                job_data = clean_text_fn(page.page_content)
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
                st.error(f"Error generating email: {e}")

if __name__ == "__main__":
    chain = Chain()
    create_streamlit_app(chain, clean_text)
