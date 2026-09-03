import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

MODEL = "gemini-3.5-flash"
MAX_FILE_SIZE_MB = 10


def get_api_key():
    """Read the Gemini API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def extract_pdf_text(file_bytes):
    """Extract text from a text-based PDF."""
    reader = PdfReader(file_bytes)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def extract_docx_text(file_bytes):
    """Extract text from DOCX paragraphs and tables."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        document = Document(temp_path)
        parts = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    parts.append(row_text)

        return "\n".join(parts).strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def extract_resume_text(uploaded_file):
    """Extract readable text from PDF, DOCX, or TXT."""
    file_bytes = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_bytes)

    if extension == ".docx":
        return extract_docx_text(file_bytes)

    if extension == ".txt":
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type.")


def analyze_resume(resume_text, job_description):
    """Send the resume to Gemini and return structured analysis."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to "
            "Streamlit Secrets or your environment variables."
        )

    client = genai.Client(api_key=api_key)

    response_schema = {
        "type": "object",
        "properties": {
            "ats_score": {
                "type": "integer",
                "description": "Estimated ATS compatibility score from 0 to 100.",
            },
            "summary": {
                "type": "string",
                "description": "Short overall assessment of the resume.",
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "formatting_issues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "action_plan": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "ats_score",
            "summary",
            "strengths",
            "missing_keywords",
            "formatting_issues",
            "improvements",
            "action_plan",
        ],
    }

    job_text = job_description.strip() or "No job description was provided."

    # Keep the prompt reasonably sized.
    resume_text = resume_text[:50000]
    job_text = job_text[:30000]

    prompt = f"""
You are an expert resume reviewer and ATS optimization specialist.

Analyze the resume and estimate how well it would perform in an
Applicant Tracking System (ATS).

IMPORTANT:
- This is an estimated ATS compatibility score, not a score from a
  specific ATS vendor.
- Score from 0 to 100.
- Do not invent experience, skills, education, certifications,
  employers, dates, achievements, or keywords as facts.
- If a job description is provided, prioritize alignment with it.
- If no job description is provided, evaluate general ATS readiness.
- Give practical recommendations the candidate can actually apply.

SCORING GUIDANCE:
0-25   Very weak
26-50  Needs major improvement
51-70  Fair
71-85  Strong
86-100 Excellent

Evaluate:
1. Keyword and job-description alignment
2. Standard resume section headings
3. Relevant skills
4. Measurable achievements
5. Clear job titles
6. Clear employment dates
7. ATS-safe formatting
8. Readability
9. Action verbs
10. Relevance and clarity of experience

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    result = json.loads(response.text)

    score = int(result.get("ats_score", 0))
    result["ats_score"] = max(0, min(100, score))

    return result


def display_list(items, empty_message):
    """Display a list of analysis items."""
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.success(empty_message)


def main():
    st.title("📄 Resume ATS Analyzer")
    st.write(
        "Upload your resume to get an estimated ATS score, "
        "keyword gaps, formatting issues, and actionable improvements."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.write(f"**AI model:** `{MODEL}`")
        st.write("**Supported:** PDF, DOCX, TXT")
        st.write(f"**Maximum file size:** {MAX_FILE_SIZE_MB} MB")
        st.divider()
        st.info(
            "For the most useful ATS analysis, paste the target "
            "job description as well."
        )
        st.warning(
            "Never put your Gemini API key directly in app.py "
            "or commit it to GitHub."
        )

    uploaded_file = st.file_uploader(
        "📤 Upload your resume",
        type=["pdf", "docx", "txt"],
        help="Upload a text-based PDF, DOCX, or TXT resume.",
    )

    job_description = st.text_area(
        "💼 Target job description (optional)",
        height=220,
        placeholder=(
            "Paste the job description here. The analyzer will "
            "compare your resume against the requirements."
        ),
    )

    if not uploaded_file:
        st.info("Upload a resume above to start the analysis.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.caption(
        f"Selected file: **{uploaded_file.name}** · "
        f"{file_size_mb:.2f} MB"
    )

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"File is too large. Please upload a file smaller than "
            f"{MAX_FILE_SIZE_MB} MB."
        )
        return

    if st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_resume_text(uploaded_file)

            if not resume_text:
                st.error(
                    "No readable text was found. If this is a scanned "
                    "PDF, upload a text-based PDF or DOCX version."
                )
                return

            with st.spinner("Gemini is analyzing your resume..."):
                result = analyze_resume(
                    resume_text=resume_text,
                    job_description=job_description,
                )

            st.session_state["analysis"] = result

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check your Gemini API key, internet connection, "
                "file format, and Streamlit logs."
            )

    result = st.session_state.get("analysis")

    if not result:
        return

    st.divider()

    score_col, summary_col = st.columns([1, 2])

    with score_col:
        score = result["ats_score"]
        st.metric("Estimated ATS Score", f"{score}/100")
        st.progress(score / 100)

        if score >= 86:
            st.success("Excellent ATS readiness")
        elif score >= 71:
            st.success("Strong ATS readiness")
        elif score >= 51:
            st.warning("Fair ATS readiness")
        else:
            st.error("Needs significant improvement")

    with summary_col:
        st.subheader("📋 Overall Assessment")
        st.write(result["summary"])

    st.subheader("💪 Strengths")
    display_list(
        result.get("strengths", []),
        "No major strengths were identified.",
    )

    st.subheader("🔑 Missing / Weak Keywords")
    display_list(
        result.get("missing_keywords", []),
        "No major keyword gaps were identified.",
    )

    st.subheader("⚠️ Formatting Issues")
    display_list(
        result.get("formatting_issues", []),
        "No major formatting issues were identified.",
    )

    st.subheader("🛠️ Recommended Improvements")
    improvements = result.get("improvements", [])

    if improvements:
        for index, item in enumerate(improvements, 1):
            st.write(f"**{index}.** {item}")
    else:
        st.success("No additional improvements were identified.")

    st.subheader("🚀 Action Plan")
    action_plan = result.get("action_plan", [])

    if action_plan:
        for index, item in enumerate(action_plan, 1):
            st.write(f"**{index}.** {item}")
    else:
        st.success("No additional action items were identified.")

    analysis_json = json.dumps(result, indent=2, ensure_ascii=False)

    st.download_button(
        "⬇️ Download Analysis as JSON",
        data=analysis_json,
        file_name="resume_ats_analysis.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
