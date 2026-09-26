"""
AI-Powered Content Creation and Analysis System
College assignment app using Grok (xAI API) with Streamlit.
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# Groq API (keys start with gsk_). OpenAI-compatible chat endpoint.
# Groq retired llama-3.3-70b-versatile. This key can use:
# openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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


def build_content_prompt(topic, content_type):
    """Structured prompt: Role, Context, Task, Constraints, Output Format."""
    return f"""ROLE:
You are an expert content creator.

CONTEXT:
The user wants content about the given topic.
Topic: {topic}
Content type: {content_type}

TASK:
Create the requested type of content.

CONSTRAINTS:
- Use simple language.
- Make the content informative and engaging.
- Avoid unnecessary repetition.
- Keep the content original.

OUTPUT FORMAT:
Title:
Content:
"""


def build_podcast_prompt(topic):
    """Structured prompt for podcast planning."""
    return f"""ROLE:
You are an experienced podcast producer.

CONTEXT:
The podcast topic is: {topic}

TASK:
Create a podcast plan.

CONSTRAINTS:
- Make it informative and engaging.
- Suggest an appropriate guest type.
- Create exactly 5 relevant interview questions.

OUTPUT FORMAT:

Podcast Title:
Podcast Description:
Guest Type:

Interview Questions:
1.
2.
3.
4.
5.
"""


def call_llm(prompt, temperature=0.7, top_p=1.0):
    """Send a prompt to the LLM and return the generated text."""
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=900,
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


def analyze_sentiment(text):
    """Classify text as Positive, Negative, or Neutral using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound


def extract_keywords(text, max_keywords=8):
    """Extract top keywords with TF-IDF (local, no Grok)."""
    sentences = re.split(r"[.!?]\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Use sentences as documents so IDF has more than one sample when possible.
    documents = sentences if len(sentences) > 1 else sentences * 2

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50,
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return []

    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)

    keywords = []
    for term, _score in ranked:
        if term not in keywords:
            keywords.append(term)
        if len(keywords) >= max_keywords:
            break
    return keywords


def require_topic(topic):
    if not topic.strip():
        st.error("Please enter a topic first.")
        return False
    return True


st.set_page_config(page_title="AI Content Creation and Analysis", layout="centered")
st.title("AI-Powered Content Creation and Analysis System")
st.caption("Type a prompt, generate content, plan a podcast, analyze text, and compare LLM parameters.")

st.subheader("Your Prompt")
user_prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Write 5 simple study tips for college students.",
    height=140,
    key="user_prompt_box",
)

if st.button("Send Prompt"):
    if not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        try:
            with st.spinner("Getting a response..."):
                reply = call_llm(user_prompt.strip())
            st.session_state["prompt_reply"] = reply
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))

if st.session_state.get("prompt_reply"):
    st.markdown("**AI Response**")
    st.write(st.session_state["prompt_reply"])

st.divider()
st.subheader("Assignment Tools")
topic = st.text_input("Topic:", placeholder="Artificial Intelligence in Education")
content_type = st.selectbox("Content Type:", ["Story", "Poem", "Social Media Post"])

st.subheader("Structured Prompt (Prompt Engineering)")
content_prompt_preview = build_content_prompt(
    topic.strip() if topic.strip() else "{topic}",
    content_type,
)
st.code(content_prompt_preview, language="text")

if st.button("Generate Content"):
    if require_topic(topic):
        try:
            with st.spinner("Generating content with Grok..."):
                result = call_llm(build_content_prompt(topic.strip(), content_type))
            st.session_state["generated_content"] = result
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))

st.divider()
st.subheader("Generated Content")
if st.session_state.get("generated_content"):
    st.write(st.session_state["generated_content"])
else:
    st.info("Click **Generate Content** to see the Grok output here.")

st.divider()
st.subheader("Podcast Planner")
st.code(build_podcast_prompt(topic.strip() if topic.strip() else "{topic}"), language="text")

if st.button("Generate Podcast Plan"):
    if require_topic(topic):
        try:
            with st.spinner("Creating podcast plan with Grok..."):
                plan = call_llm(build_podcast_prompt(topic.strip()), temperature=0.7, top_p=1.0)
            st.session_state["podcast_plan"] = plan
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))

if st.session_state.get("podcast_plan"):
    st.write(st.session_state["podcast_plan"])

st.divider()
st.subheader("Text Analysis")
analysis_text = st.text_area(
    "Enter text for analysis",
    placeholder="Enter text here................",
    height=150,
)

if st.button("Analyze Text"):
    if not analysis_text.strip():
        st.error("Please enter text for analysis.")
    else:
        sentiment_label, compound = analyze_sentiment(analysis_text)
        keywords = extract_keywords(analysis_text)

        st.session_state["sentiment"] = sentiment_label
        st.session_state["compound"] = compound
        st.session_state["keywords"] = keywords

if st.session_state.get("sentiment"):
    st.write(f"**Sentiment:** {st.session_state['sentiment']}")
    st.caption(f"VADER compound score: {st.session_state['compound']:.3f}")
    st.write("**Keywords:**")
    keyword_list = st.session_state.get("keywords") or []
    if keyword_list:
        st.write("\n".join(keyword_list))
    else:
        st.write("No keywords could be extracted. Try a longer English paragraph.")

st.divider()
st.subheader("Parameter Experimentation")
st.markdown(
    """
- Lower temperature generally produces more predictable responses.
- Higher temperature generally allows more variation.
- Top-P controls the range of tokens considered during generation.

These settings change *style and variety*. One setting is not universally better.
"""
)

if st.button("Compare Temperature"):
    if require_topic(topic):
        prompt = build_content_prompt(topic.strip(), content_type)
        temps = [0.2, 0.7, 1.0]
        outputs = []
        try:
            with st.spinner("Generating three outputs with different temperature values..."):
                for temp in temps:
                    text = call_llm(prompt, temperature=temp, top_p=1.0)
                    outputs.append((temp, text))
            st.session_state["temp_outputs"] = outputs
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))

if st.session_state.get("temp_outputs"):
    st.markdown("**Temperature experiment** (Top-P kept constant at 1.0)")
    for temp, text in st.session_state["temp_outputs"]:
        st.markdown(f"**Temperature = {temp}**")
        st.write(text)

if st.button("Compare Top-P"):
    if require_topic(topic):
        prompt = build_content_prompt(topic.strip(), content_type)
        top_p_values = [0.5, 0.8, 1.0]
        outputs = []
        try:
            with st.spinner("Generating three outputs with different Top-P values..."):
                for top_p in top_p_values:
                    text = call_llm(prompt, temperature=0.7, top_p=top_p)
                    outputs.append((top_p, text))
            st.session_state["top_p_outputs"] = outputs
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))

if st.session_state.get("top_p_outputs"):
    st.markdown("**Top-P experiment** (Temperature kept constant at 0.7)")
    for top_p, text in st.session_state["top_p_outputs"]:
        st.markdown(f"**Top-P = {top_p}**")
        st.write(text)
