import streamlit as st
import os
import io
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- API KEY CONFIGURATION ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    keys_loaded = True
except (KeyError, FileNotFoundError):
    keys_loaded = False

# --- SHARED COMPONENTS INITIALIZATION ---
# Initialize LLM and Tools only once
@st.cache_resource
def get_shared_components():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.5, # Slightly lower temperature for more predictable output
    )
    search_tool = TavilySearchResults(k=5)
    return llm, search_tool

if keys_loaded:
    llm, search_tool = get_shared_components()

# --- HELPER FUNCTIONS ---

def parse_resume(file):
    """Parses uploaded resume file (PDF or DOCX) and returns its text content."""
    file_extension = os.path.splitext(file.name)[1]
    text = ""
    try:
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_extension == ".docx":
            doc = Document(io.BytesIO(file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return None # Unsupported file type
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None
    return text

def calculate_match_score(resume_text, job_description):
    """Calculates a match score between resume and job description using TF-IDF."""
    if not resume_text or not job_description:
        return 0
    text = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return int(cosine_sim[0][0] * 100)


# --- AGENT & FEATURE FUNCTIONS ---

def run_research_agent(company_name, job_role):
    """
    Initializes and runs the LangChain agent to research a company and job role.
    (This is your original function, slightly adapted to use the shared llm and tool)
    """
    tools = [search_tool]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )

    input_prompt = f"""
    Research the company '{company_name}' and the specific job role of '{job_role}'.

    Your final answer MUST be a comprehensive summary structured into two clear sections using Markdown:

    ### *Company Overview*
    * *Domain/Industry*: What is the company's primary domain or industry?
    * *Size*: What is its approximate size (e.g., number of employees)?
    * *Recent News*: Find and summarize one or two recent, significant news articles about the company.

    ### *Role-Specific Requirements*
    * *Common Skills*: What are the most commonly required skills for an '{job_role}' at this company or in the industry?
    * *Experience Level*: What is the typical level of experience (e.g., years, degrees) needed?
    * *Salary Range*: What is the estimated salary range for this role? If a specific range for the company isn't available, provide a general industry estimate.
    """

    response = agent_executor.invoke({"input": input_prompt})
    return response['output']

def run_job_matcher(resume_text):
    """Finds suitable job postings based on resume text."""
    query_prompt = f"""
    Based on the following resume text, generate a concise and effective search query to find suitable job postings on the internet.
    The query should include a likely job title and key skills.
    Example: "Data Scientist jobs with Python, SQL, and Machine Learning"

    Resume Text:
    ---
    {resume_text[:2000]}
    ---
    Search Query:
    """
    query_response = llm.invoke(query_prompt)
    search_query = query_response.content.strip()

    st.info(f"**Searching for jobs with query:** `{search_query}`")

    # Use Tavily to find job postings
    search_results = search_tool.invoke(f"{search_query} job description")

    matched_jobs = []
    for result in search_results:
        title = result.get('title', 'No Title')
        url = result.get('url', '#')
        description = result.get('content', '')

        if description:
            score = calculate_match_score(resume_text, description)
            # The 'if score > 40' condition has been removed.
            # Every result will now be added to the list.
            matched_jobs.append({
                "title": title,
                "url": url,
                "description": description,
                "score": score
            })

    # Sort jobs by score in descending order
    matched_jobs.sort(key=lambda x: x['score'], reverse=True)
    return matched_jobs[:5] # Return the top 5 results found


def run_resume_tailor(resume_text, job_description):
    """Uses an LLM to tailor a resume for a specific job description."""
    prompt = f"""
    You are an expert career coach and professional resume writer. Your task is to rewrite the provided resume to be perfectly tailored for the given job description.

    Follow these instructions carefully:
    1.  **Analyze Both Documents:** Thoroughly read and understand the user's original resume and the target job description.
    2.  **Incorporate Keywords:** Identify the key skills, technologies, and qualifications mentioned in the job description. Integrate them naturally into the resume's summary, skills, and experience sections.
    3.  **Align with Job Duties:** Rephrase the experience bullet points to highlight accomplishments and responsibilities that are most relevant to the new role. Use action verbs from the job description where appropriate.
    4.  **DO NOT FABRICATE:** You must not invent or exaggerate the user's experience. Work ONLY with the information provided in the original resume. Rephrase and reframe, but do not lie.
    5.  **Maintain Professional Tone:** The output must be professional, clear, and concise.
    6.  **Output Format:** Produce the final, tailored resume in well-structured Markdown format.

    Here is the user's original resume:
    ---
    {resume_text}
    ---

    Here is the target job description:
    ---
    {job_description}
    ---

    Now, please provide the new, tailored resume.
    """
    response = llm.invoke(prompt)
    return response.content

# --- STREAMLIT UI ---
st.title("🤖 AI Career Assistant")
st.markdown("Your all-in-one tool to research companies, match your resume to jobs, and tailor your application.")

if not keys_loaded:
    st.error("API keys not found. Please add your GOOGLE_API_KEY and TAVILY_API_KEY to your Streamlit secrets.")
    st.markdown(
        "Create a file named `.streamlit/secrets.toml` in your project directory and add your keys like this:\n"
        "```toml\n"
        "GOOGLE_API_KEY = \"your_google_api_key_here\"\n"
        "TAVILY_API_KEY = \"your_tavily_api_key_here\"\n"
        "```"
    )
else:
    # Initialize session state for resume text
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""

    tab1, tab2, tab3 = st.tabs(["Company Research", "Resume Analysis & Job Matching", "Resume Tailoring"])

    # --- TAB 1: COMPANY RESEARCH ---
    with tab1:
        st.header("🔍 Research a Company and Role")
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Enter Company Name:", placeholder="e.g., Google, Microsoft", key="company")
        with col2:
            job_role = st.text_input("Enter Job Role:", placeholder="e.g., Software Engineer", key="role")

        if st.button("Start Research", type="primary", key="research_button"):
            if company_name and job_role:
                with st.spinner(f"Researching {company_name} for the role of {job_role}..."):
                    try:
                        result = run_research_agent(company_name, job_role)
                        st.markdown("---")
                        st.subheader("Research Summary")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter both a company name and a job role.")

# --- TAB 2: RESUME ANALYSIS & JOB MATCHING ---
    with tab2:
        st.header("📄 Analyze Your Resume and Find Matching Jobs")
        uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_uploader_tab2")

        if uploaded_file:
            st.session_state.resume_text = parse_resume(uploaded_file)
            if st.session_state.resume_text:
                st.success("Resume parsed successfully!")

        if st.button("Find Matching Jobs", type="primary", key="matcher_button", disabled=not st.session_state.resume_text):
            with st.spinner("Analyzing resume and searching for the best job matches..."):
                try:
                    matched_jobs = run_job_matcher(st.session_state.resume_text)
                    st.markdown("---")
                    st.subheader("Top Job Matches")

                    if not matched_jobs:
                        st.warning("The web search did not return any job postings for your profile. You might try uploading a more detailed resume or checking your search query.")
                    else:
                        st.markdown("Here are the top matches found for your profile. This can help you understand what the job market looks like, even if the match percentage is low.")
                        for job in matched_jobs:
                            with st.expander(f"**{job['title']}** - Match Score: {job['score']}%"):
                                st.markdown(f"**URL:** [Link]({job['url']})")
                                st.markdown("**Job Description Snippet:**")
                                st.write(job['description'][:500] + "...")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # --- TAB 3: RESUME TAILORING ---
    with tab3:
        st.header("✨ Tailor Your Resume for a Specific Job")
        st.markdown("Upload your resume and paste a job description below to get a tailored version.")

        col1_tailor, col2_tailor = st.columns(2)
        with col1_tailor:
            uploaded_file_tailor = st.file_uploader("1. Upload your Resume", type=["pdf", "docx"], key="resume_uploader_tab3")
            if uploaded_file_tailor:
                st.session_state.resume_text = parse_resume(uploaded_file_tailor)
                if st.session_state.resume_text:
                    st.success("Resume parsed successfully!")

        with col2_tailor:
            job_description_text = st.text_area("2. Paste Job Description Here", height=300, key="jd_text")

        if st.button("Tailor My Resume", type="primary", key="tailor_button", disabled=not (st.session_state.resume_text and job_description_text)):
            with st.spinner("Your personal AI career coach is crafting the perfect resume..."):
                try:
                    # Calculate score before tailoring
                    original_score = calculate_match_score(st.session_state.resume_text, job_description_text)

                    # Run the tailor function
                    tailored_resume = run_resume_tailor(st.session_state.resume_text, job_description_text)

                    # Calculate score after tailoring
                    new_score = calculate_match_score(tailored_resume, job_description_text)

                    st.markdown("---")
                    st.subheader("Your Tailored Resume")

                    score_col1, score_col2 = st.columns(2)
                    with score_col1:
                        st.metric(label="Original Match Score", value=f"{original_score}%")
                    with score_col2:
                        st.metric(label="New Tailored Score", value=f"{new_score}%", delta=f"{new_score - original_score}%")

                    st.markdown(tailored_resume)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
