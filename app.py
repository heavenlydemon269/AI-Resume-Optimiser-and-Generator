import streamlit as st
import PyPDF2
import docx
import re
import collections
import io

def extract_text_from_pdf(pdf_content_bytes):
    """Extracts text from a PDF file's byte content."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content_bytes))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        return f"Error reading PDF file: {e}"
    return text

def extract_text_from_docx(docx_content_bytes):
    """Extracts text from a DOCX file's byte content."""
    text = ""
    try:
        doc = docx.Document(io.BytesIO(docx_content_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        return f"Error reading DOCX file: {e}"
    return text

def get_resume_text(filename, content_bytes):
    """Determines file type and extracts text accordingly."""
    if filename.lower().endswith('.pdf'):
        return extract_text_from_pdf(content_bytes)
    elif filename.lower().endswith('.docx'):
        return extract_text_from_docx(content_bytes)
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."

def extract_keywords(text):
    """Extracts keywords from text using simple NLP."""
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "responsibilities", "requirements", "experience", "skills", "duties",
        "qualifications", "required", "preferred", "plus", "degree", "etc"
    ])

    words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    most_common_keywords = collections.Counter(keywords).most_common(25)
    return [keyword for keyword, count in most_common_keywords]


st.set_page_config(layout="wide", page_title="AI Resume Optimiser")

st.title("🤖 AI Resume Optimiser")
st.markdown("""
Welcome! This tool helps you tailor your resume to a specific job description to improve your chances of passing through Applicant Tracking Systems (ATS) and catching a recruiter's eye.

**Instructions:**
1.  **Upload your resume** in PDF or DOCX format.
2.  **Paste the job description** into the text box.
3.  Click the **"Optimise Resume"** button to get your analysis and a tailored template.
""")

st.header("Step 1: Provide Your Details")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload Your Resume", type=['pdf', 'docx'], help="Please upload your resume in PDF or DOCX format.")

with col2:
    job_description = st.text_area("Paste the Job Description Here", height=300, placeholder="Paste the entire job description text here...")

if st.button("🚀 Optimise Resume", type="primary"):
    if uploaded_resume is not None and job_description:
        with st.spinner('Analysing your resume and the job description...'):
            resume_bytes = uploaded_resume.getvalue()
            resume_filename = uploaded_resume.name

            resume_text = get_resume_text(resume_filename, resume_bytes)

            if "Error" in resume_text or "Unsupported" in resume_text:
                st.error(f"Could not process resume: {resume_text}")
            else:
                jd_keywords = extract_keywords(job_description)

                found_keywords = []
                missing_keywords = []
                resume_text_lower = resume_text.lower()

                for keyword in jd_keywords:
                    if keyword in resume_text_lower:
                        found_keywords.append(keyword)
                    else:
                        missing_keywords.append(keyword)

                st.header("Step 2: Your Optimisation Report")
                st.success("Analysis complete! Here are your results.")

                report_col1, report_col2 = st.columns(2)

                with report_col1:
                    st.subheader("📊 Keyword Analysis")
                    st.markdown(f"**Top Keywords from Job Description:**")
                    st.info(f"{', '.join(jd_keywords[:15])}")

                    st.markdown(f"**✅ Keywords Found in Your Resume:**")
                    if found_keywords:
                        st.success(f"{', '.join(found_keywords)}")
                    else:
                        st.warning("None of the top keywords were found in your resume.")

                    st.markdown(f"**⚠️ Keywords to Add (if experienced):**")
                    if missing_keywords:
                        st.warning(f"{', '.join(missing_keywords)}")
                        st.markdown("_**Tip:** Weave these naturally into your summary, skills, and experience descriptions._")
                    else:
                        st.success("Great job! Your resume contains all the top keywords.")


                with report_col2:
                    st.subheader("📝 Formatting & ATS Tips")
                    st.markdown("""
                    * **Professional Summary:** Start with a 3-4 sentence summary tailored to the job.
                    * **Skills Section:** Use a clear, bulleted list of your skills. Group them by category.
                    * **Action Verbs & Metrics:** Begin experience bullet points with strong verbs (e.g., *Engineered, Managed, Led*). Quantify achievements with numbers (e.g., *"...increased revenue by 15%."*).
                    * **Simple Formatting:** Avoid tables, columns, and images which can confuse ATS software.
                    """)

                st.subheader("📄 Your Suggested Optimised Resume (Template)")
                st.markdown("---")
                st.markdown("_Use this template as a guide. Fill it with your specific achievements, making sure to incorporate the missing keywords where relevant._")

                optimised_resume = f"""
*[Your Name]*
[Your Phone Number] | [Your Email] | [Your LinkedIn Profile URL]

---

*Professional Summary*
A results-oriented **[Your Role, e.g., Software Engineer]** with X years of experience, specializing in **{missing_keywords[0] if missing_keywords else 'backend development'}**. Proven ability to leverage **{found_keywords[0] if found_keywords else 'Python'}** and **{found_keywords[1] if found_keywords and len(found_keywords)>1 else 'Java'}** to build scalable systems. Eager to apply my skills in **{missing_keywords[1] if missing_keywords and len(missing_keywords)>1 else 'cloud computing'}** to contribute to [Target Company Name].

---

*Skills*
* **Key Skills (from Job Description):** {', '.join(jd_keywords[:8])}
* **Programming & Languages:** [List languages, e.g., Python, Java, SQL]
* **Technologies & Frameworks:** [List tech, e.g., Docker, Kubernetes, Django, AWS, GCP]
* **Soft Skills:** [e.g., Agile Methodologies, Problem-Solving, Team Collaboration]

---

**Professional Experience**

**[Your Most Recent Job Title]** | [Company Name] | [City, State] | [Dates]
* Engineered a new feature using **{found_keywords[0] if found_keywords else 'Python'}**, which improved system performance by 20%.
* Collaborated in an Agile team to develop and deploy microservices on **{missing_keywords[0] if missing_keywords else 'AWS'}**, reducing latency by 150ms.
* Managed the full software development lifecycle (SDLC) for a critical customer-facing application, improving user retention by 10%.

**[Your Previous Job Title]** | [Company Name] | [City, State] | [Dates]
* [Start with an action verb. Weave in a keyword. Add a number.]
* [Action Verb + Keyword + Result]

---

**Education**

**[Your Degree]** | [University Name] | [City, State] | [Year of Graduation]
"""
                st.text_area("Copy this template", optimised_resume, height=500)

    else:
        st.error("Please make sure you have uploaded a resume AND pasted the job description before optimising.")
