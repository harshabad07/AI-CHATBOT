"""
Study & Exam Assistant
College Assignment and Notes System using Groq (OpenAI-compatible API) with Streamlit.
Features: Conversational Chatbot, Practice Exam Generator, Flashcards,
          Step-by-Step Solver, and Study Roadmap.
"""

import os
import json
import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

ACADEMIC_LEVELS = [
    "High School",
    "College / University",
    "Graduate / Post-Graduate",
    "Professional / Competitive Exams",
]

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_api_key():
    """Read the API key from environment. Never print the key."""
    return (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("XAI_API_KEY", "").strip()
    )


def get_llm_client():
    """Create an OpenAI-compatible client pointed at Groq."""
    api_key = get_api_key()
    if not api_key or api_key == "your_xai_api_key_here":
        raise ValueError(
            "Missing API key. Put your Groq key in the .env file as GROQ_API_KEY."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def call_llm(messages, model, temperature=0.7, max_tokens=1500):
    """Send messages to the LLM and return the generated text."""
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as error:
        raise RuntimeError(f"API error: {error}") from error

    if not response or not getattr(response, "choices", None):
        raise RuntimeError("Invalid API response: no choices returned.")

    message = response.choices[0].message
    text = getattr(message, "content", None)
    if (not text or not str(text).strip()) and getattr(message, "reasoning", None):
        text = message.reasoning
    if not text or not str(text).strip():
        raise RuntimeError("Invalid API response: empty content.")

    return str(text).strip()


# ---------------------------------------------------------------------------
# System prompts for each feature
# ---------------------------------------------------------------------------

def system_prompt_chatbot(level, notes_context=""):
    """System prompt for the conversational study chatbot."""
    notes_section = ""
    if notes_context:
        notes_section = f"""
REFERENCE NOTES / SYLLABUS (provided by the student):
---
{notes_context}
---
Use the above notes as primary context when answering. Refer to them when relevant.
"""
    return f"""You are a friendly, knowledgeable educational AI assistant called "Study & Exam Assistant".

TARGET LEVEL: {level}

{notes_section}

RULES:
- Help students understand concepts, prepare for exams, solve practice problems, create flashcards, and build study roadmaps.
- Adjust explanation depth to the student's academic level.
- Use clear, step-by-step explanations.
- When asked about a concept, provide examples.
- Be encouraging and supportive.
- If you don't know something, say so honestly.
- Keep responses focused and well-structured using markdown formatting.
"""


def system_prompt_exam(level, notes_context=""):
    """System prompt for the practice exam generator."""
    notes_section = ""
    if notes_context:
        notes_section = f"\nUse the following reference notes/syllabus to generate questions:\n---\n{notes_context}\n---\n"
    return f"""You are an expert exam question creator for {level} students.
{notes_section}
TASK: Generate a practice exam based on the subject and topic the student specifies.

OUTPUT FORMAT:
## Practice Exam: [Topic]

**Instructions:** Answer all questions. Time suggested: X minutes.

### Multiple Choice Questions
(Generate 5 MCQs with 4 options each. Mark the correct answer.)

### Short Answer Questions
(Generate 3 short-answer questions.)

### Long Answer / Essay Question
(Generate 1 detailed question.)

---
### Answer Key
(Provide all answers with brief explanations.)
"""


def system_prompt_flashcards(level, notes_context=""):
    """System prompt for flashcard generation."""
    notes_section = ""
    if notes_context:
        notes_section = f"\nUse the following reference notes/syllabus to create flashcards:\n---\n{notes_context}\n---\n"
    return f"""You are a flashcard creator for {level} students.
{notes_section}
TASK: Create study flashcards for the topic the student specifies.

OUTPUT FORMAT:
Create 10 flashcards. For each flashcard use this exact format:

**Card 1**
- **Front:** [Question or term]
- **Back:** [Answer or definition]

Make the flashcards progressively harder. Cover key concepts, definitions, formulas, and applications.
"""


def system_prompt_solver(level, notes_context=""):
    """System prompt for the step-by-step solver."""
    notes_section = ""
    if notes_context:
        notes_section = f"\nReference notes for context:\n---\n{notes_context}\n---\n"
    return f"""You are a patient tutor who solves problems step-by-step for {level} students.
{notes_section}
RULES:
- Break every solution into clear, numbered steps.
- Explain the reasoning behind each step.
- Use proper mathematical notation (when applicable).
- At the end, provide a summary of the approach used.
- Highlight common mistakes students make on this type of problem.

OUTPUT FORMAT:
## Problem
(Restate the problem)

## Solution

**Step 1:** ...
**Step 2:** ...
(continue as needed)

## Summary
(Brief recap of the method)

## Common Mistakes to Avoid
- ...
"""


def system_prompt_roadmap(level, notes_context=""):
    """System prompt for study roadmap generation."""
    notes_section = ""
    if notes_context:
        notes_section = f"\nUse the following syllabus/notes to create a targeted roadmap:\n---\n{notes_context}\n---\n"
    return f"""You are an expert academic planner creating study roadmaps for {level} students.
{notes_section}
TASK: Create a structured study roadmap for the subject/exam the student specifies.

OUTPUT FORMAT:
## 📅 Study Roadmap: [Subject/Exam]

### Overview
(Brief description of what this roadmap covers)

### Prerequisites
(What the student should know before starting)

### Week-by-Week Plan

**Week 1: [Topic]**
- 📖 Topics to cover: ...
- 🎯 Learning goals: ...
- 📝 Practice tasks: ...
- ⏱️ Estimated time: ...

(Continue for 4-8 weeks depending on scope)

### Recommended Resources
- 📚 Books: ...
- 🌐 Online: ...
- 🎥 Videos: ...

### Exam Day Tips
- ...
"""


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Study & Exam Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    /* Main header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #fafbfc;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem;
        color: #1a1a2e;
    }

    /* Groq active badge */
    .groq-badge {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }

    /* Upload section styling */
    .upload-section {
        background: #f0f4ff;
        border: 1px dashed #93c5fd;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_notes" not in st.session_state:
    st.session_state.uploaded_notes = ""

if "groq_active" not in st.session_state:
    st.session_state.groq_active = False

# Check if API key is available
try:
    get_llm_client()
    st.session_state.groq_active = True
except ValueError:
    st.session_state.groq_active = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📋 Study Preferences")

    academic_level = st.selectbox(
        "Target Academic Level",
        ACADEMIC_LEVELS,
        index=1,
        help="This adjusts the complexity of explanations and generated content.",
    )

    selected_model = st.selectbox(
        "Select Groq Model",
        AVAILABLE_MODELS,
        index=0,
        help="Choose which Groq-hosted model to use for generation.",
    )

    if st.session_state.groq_active:
        st.markdown('<div class="groq-badge">⚡ Groq AI Active</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ API key missing. Add GROQ_API_KEY to your .env file.")

    st.divider()

    # Reference notes upload
    st.markdown("### 📑 Reference Notes / Syllabus")
    st.caption("Upload Study Notes (.txt, .md)")

    uploaded_file = st.file_uploader(
        "Upload",
        type=["txt", "md"],
        label_visibility="visible",
        help="Upload your study notes or syllabus. The AI will use them as context.",
    )
    st.caption("200MB per file • TXT, MD")

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            # Truncate very long notes to avoid exceeding token limits
            if len(content) > 8000:
                content = content[:8000] + "\n\n[... Notes truncated for context limit ...]"
            st.session_state.uploaded_notes = content
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Study & Exam Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Your educational AI assistant for understanding concepts, exam preparation, "
    "practice questions, and structured problem-solving."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_exam, tab_flash, tab_solver, tab_roadmap = st.tabs([
    "💬 Conversational Chatbot",
    "📝 Practice Exam Generator",
    "🃏 Flashcards",
    "📐 Step-by-Step Solver",
    "🗓️ Study Roadmap",
])

notes_ctx = st.session_state.uploaded_notes


# ---- Tab 1: Conversational Chatbot ----------------------------------------

with tab_chat:
    st.markdown(
        "### 💬 Study Chatbot (Conversational Memory Active) 🔗"
    )
    st.caption(
        "Ask questions, request follow-ups, or explore concepts step-by-step. "
        "The assistant remembers your previous questions in this session."
    )

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Clear Chat / New Topic", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("📄 Export Full Chat Transcript (.md)", use_container_width=True):
            if st.session_state.chat_history:
                transcript = "# Study & Exam Assistant — Chat Transcript\n\n"
                transcript += f"*Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"
                for msg in st.session_state.chat_history:
                    role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                    transcript += f"**{role}:**\n{msg['content']}\n\n---\n\n"
                st.download_button(
                    "⬇️ Download Transcript",
                    data=transcript,
                    file_name="chat_transcript.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("No chat history to export yet.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask about any subject or concept..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build messages for the API
        sys_prompt = system_prompt_chatbot(academic_level, notes_ctx)
        api_messages = [{"role": "system", "content": sys_prompt}]
        # Keep last 20 messages for context window
        recent = st.session_state.chat_history[-20:]
        api_messages.extend(recent)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = call_llm(api_messages, model=selected_model)
                    st.markdown(reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))


# ---- Tab 2: Practice Exam Generator ---------------------------------------

with tab_exam:
    st.markdown("### 📝 Practice Exam Generator")
    st.caption("Enter a subject and topic to generate a full practice exam with answer key.")

    exam_subject = st.text_input(
        "Subject / Topic",
        placeholder="e.g., Organic Chemistry — Alkenes and Alkynes",
        key="exam_subject",
    )

    exam_difficulty = st.select_slider(
        "Difficulty",
        options=["Easy", "Medium", "Hard", "Mixed"],
        value="Medium",
        key="exam_difficulty",
    )

    if st.button("🎯 Generate Practice Exam", use_container_width=True):
        if not exam_subject.strip():
            st.error("Please enter a subject or topic.")
        else:
            sys_prompt = system_prompt_exam(academic_level, notes_ctx)
            user_msg = (
                f"Generate a {exam_difficulty.lower()} difficulty practice exam "
                f"on: {exam_subject.strip()}"
            )
            api_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ]
            with st.spinner("Generating your practice exam..."):
                try:
                    result = call_llm(api_messages, model=selected_model, max_tokens=2500)
                    st.markdown(result)
                    st.download_button(
                        "⬇️ Download Exam (.md)",
                        data=result,
                        file_name="practice_exam.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))


# ---- Tab 3: Flashcards ----------------------------------------------------

with tab_flash:
    st.markdown("### 🃏 Flashcard Generator")
    st.caption("Generate study flashcards for any topic. Great for quick revision!")

    flash_topic = st.text_input(
        "Topic for flashcards",
        placeholder="e.g., Python Data Structures, Cell Biology, Calculus Limits",
        key="flash_topic",
    )

    num_cards = st.slider("Number of flashcards", 5, 20, 10, key="num_cards")

    if st.button("🃏 Generate Flashcards", use_container_width=True):
        if not flash_topic.strip():
            st.error("Please enter a topic.")
        else:
            sys_prompt = system_prompt_flashcards(academic_level, notes_ctx)
            user_msg = f"Create {num_cards} flashcards on: {flash_topic.strip()}"
            api_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ]
            with st.spinner("Creating flashcards..."):
                try:
                    result = call_llm(api_messages, model=selected_model, max_tokens=2500)
                    st.markdown(result)
                    st.download_button(
                        "⬇️ Download Flashcards (.md)",
                        data=result,
                        file_name="flashcards.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))


# ---- Tab 4: Step-by-Step Solver -------------------------------------------

with tab_solver:
    st.markdown("### 📐 Step-by-Step Problem Solver")
    st.caption("Paste a problem and get a detailed, step-by-step solution with explanations.")

    problem_text = st.text_area(
        "Enter your problem",
        placeholder=(
            "e.g., Solve the integral of x²·sin(x) dx\n"
            "or: Explain the mechanism of SN2 reaction with an example"
        ),
        height=150,
        key="problem_text",
    )

    if st.button("🔍 Solve Step-by-Step", use_container_width=True):
        if not problem_text.strip():
            st.error("Please enter a problem to solve.")
        else:
            sys_prompt = system_prompt_solver(academic_level, notes_ctx)
            api_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": problem_text.strip()},
            ]
            with st.spinner("Solving step by step..."):
                try:
                    result = call_llm(api_messages, model=selected_model, max_tokens=2500)
                    st.markdown(result)
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))


# ---- Tab 5: Study Roadmap -------------------------------------------------

with tab_roadmap:
    st.markdown("### 🗓️ Study Roadmap Generator")
    st.caption("Get a personalised week-by-week study plan for any subject or exam.")

    roadmap_subject = st.text_input(
        "Subject or Exam",
        placeholder="e.g., Data Structures & Algorithms, GRE Preparation, Organic Chemistry",
        key="roadmap_subject",
    )

    roadmap_weeks = st.slider(
        "Roadmap duration (weeks)", 2, 12, 6, key="roadmap_weeks"
    )

    if st.button("📅 Generate Study Roadmap", use_container_width=True):
        if not roadmap_subject.strip():
            st.error("Please enter a subject or exam name.")
        else:
            sys_prompt = system_prompt_roadmap(academic_level, notes_ctx)
            user_msg = (
                f"Create a {roadmap_weeks}-week study roadmap for: "
                f"{roadmap_subject.strip()}"
            )
            api_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ]
            with st.spinner("Building your study roadmap..."):
                try:
                    result = call_llm(api_messages, model=selected_model, max_tokens=3000)
                    st.markdown(result)
                    st.download_button(
                        "⬇️ Download Roadmap (.md)",
                        data=result,
                        file_name="study_roadmap.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))
