import os
import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk
import random
from typing import Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Package guards and imports
try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Install `apify-client` (`pip install apify-client`).")
    st.stop()

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google.cloud import aiplatform
    from google.api_core.client_options import ClientOptions
except ImportError:
    aiplatform = None

try:
    from textblob import TextBlob
    nltk.download("punkt")
except ModuleNotFoundError:
    st.error("Install `textblob` and `nltk`.")
    st.stop()


# --- Streamlit page setup ---
st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export with Vertex AI")


# --- Config / keyword lists ---
ONCOLOGY_TERMS = [
    "oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"
]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]


# --- Utility: retry decorator (kept as in your code) ---
def retry_with_backoff(func=None, *, max_retries=3, base_delay=2):
    def decorator(f):
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            while attempt < max_retries:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.info(f"Retry {attempt} for {f.__name__} after exception: {e}, sleeping {delay:.1f}s")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)


# --- Caching resource clients for reuse ---
@st.cache_resource
def get_apify_client(api_token: str) -> ApifyClient:
    return ApifyClient(api_token)


@st.cache_resource
def get_openai_client(api_key: str):
    if not openai:
        raise RuntimeError("OpenAI Python SDK is not installed.")
    return openai.OpenAI(api_key=api_key)


@st.cache_resource
def get_vertex_client(project_id: str, location: str):
    if not aiplatform:
        raise RuntimeError("Google Cloud AI Platform SDK is not installed.")
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    client_options = ClientOptions(api_endpoint=api_endpoint)
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    logger.info(f"Initialized Vertex AI client: {project_id} @ {location}")
    return client


# --- API call wrappers with retry ---
@retry_with_backoff
def call_openai(prompt: str, client, model: str, temperature: float, max_tokens: int) -> str:
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


@retry_with_backoff
def call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
) -> str:
    if not genai:
        raise RuntimeError("Google Gemini Python SDK not installed.")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    params = dict(prompt=prompt)
    if max_tokens > 0:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if reasoning_effort and reasoning_effort != "None":
        params["reasoning_effort"] = reasoning_effort.lower()
    if reasoning_summary and reasoning_summary != "None":
        params["reasoning_summary"] = reasoning_summary.lower()
    response = model_obj.generate_content(**params)
    return getattr(response, "text", str(response)).strip()


@retry_with_backoff
def call_vertex_ai(
    prompt: str,
    client,
    project_id: str,
    location: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"
    instances = [{"content": prompt}]
    parameters = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens if max_tokens > 0 else 1024,
    }
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    if response.predictions and isinstance(response.predictions[0], dict):
        return response.predictions[0].get("content", "")
    return str(response.predictions)


def get_llm_response(
    prompt: str,
    provider: str,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    vertex_client: Optional[any] = None,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_temperature: float = 0.6,
    openai_max_tokens: int = 512,
    gemini_model: Optional[str] = None,
    gemini_temperature: float = 0.6,
    gemini_max_tokens: int = 512,
    gemini_reasoning_effort: Optional[str] = None,
    gemini_reasoning_summary: Optional[str] = None,
    vertex_model: Optional[str] = None,
    vertex_temperature: float = 0.6,
    vertex_max_tokens: int = 512,
) -> str:
    try:
        if provider == "OpenAI GPT":
            client = get_openai_client(openai_api_key)
            return call_openai(prompt, client, openai_model, openai_temperature, openai_max_tokens)
        elif provider == "Google Gemini":
            return call_gemini(prompt, gemini_api_key, gemini_model, gemini_temperature, gemini_max_tokens,
                               gemini_reasoning_effort, gemini_reasoning_summary)
        elif provider == "Google Vertex AI":
            if not vertex_client or not vertex_project or not vertex_location or not vertex_model:
                raise ValueError("Vertex AI client, project, location, and model must be provided for Vertex AI.")
            return call_vertex_ai(prompt, vertex_client, vertex_project, vertex_location,
                                  vertex_model, vertex_temperature, vertex_max_tokens)
        else:
            return f"Unknown LLM provider: {provider}"
    except Exception as e:
        logger.error(f"LLM call ({provider}) error: {e}")
        return f"{provider} call failed: {e}"


# --- TikTok Apify Scraper function with caching ---
@st.cache_data(show_spinner=False, persist="disk")
def run_apify_scraper_batched(api_key: str, query: str, target_total: int, batch_size: int) -> list:
    MAX_WAIT_SECONDS = 180  # 3 minutes batch timeout
    client = get_apify_client(api_key)
    results = []
    offset = 0
    failures = 0
    pbar = st.progress(0.0, text="Scraping TikTok...")
    while len(results) < target_total and failures < 5:
        st.info(f"Launching batch {1 + offset // batch_size}: {len(results)} of {target_total}")
        try:
            run = client.actor("clockworks~tiktok-scraper").call(
                run_input={
                    "searchQueries": [query],
                    "resultsPerPage": batch_size,
                    "searchType": "keyword",
                    "pageNumber": offset // batch_size,
                }
            )
            run_id = run.get("id", None) or run.get("data", {}).get("id")

            batch_start = time.time()
            dataset_id = None
            while run_id and (time.time() - batch_start) < MAX_WAIT_SECONDS:
                run_status = client.actor("clockworks~tiktok-scraper").run(run_id)
                if run_status.get("status") == "SUCCEEDED":
                    dataset_id = run_status.get("defaultDatasetId")
                    break
                time.sleep(5)

            if not dataset_id:
                failures += 1
                st.error("Batch run timed out or failed to return dataset, retrying...")
                continue

            batch_items = list(client.dataset(dataset_id).iterate_items())
            if not batch_items:
                failures += 1
                st.error("No posts found for batch, retrying...")
                continue

            for item in batch_items:
                if item not in results:
                    results.append(item)

            offset += batch_size
            pbar.progress(min(1.0, len(results) / float(target_total)))
            if len(batch_items) < batch_size:
                break  # No more data
        except Exception as e:
            st.error(f"Scraping batch failed with error: {e}")
            failures += 1
    pbar.progress(1.0)
    return results[:target_total]


def fetch_tiktok_transcripts_apify(api_token: str, video_urls: list) -> Dict[str, str]:
    try:
        client = get_apify_client(api_token)
        run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input={"videos": video_urls})
        transcripts = {}
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            video_id = item.get("id")
            transcript = item.get("transcript", "")
            transcripts[video_id] = transcript
        return transcripts
    except Exception as e:
        st.error(f"Fetching transcripts failed: {e}")
        return {}


def classify_kol_dol(score: float) -> str:
    if score >= 8:
        return "KOL"
    if score >= 5:
        return "DOL"
    return "Not Suitable"


def classify_sentiment(score: float) -> str:
    if score > 0.15:
        return "Positive"
    if score < -0.15:
        return "Negative"
    return "Neutral"


def generate_rationale(
    text: str, transcript: str, author: str, score: float, sentiment: str, mode: str
) -> str:
    all_text = f"{text or ''} {transcript or ''}".lower()
    tags = {
        "onco": any(t in all_text for t in ONCOLOGY_TERMS),
        "gi": any(t in all_text for t in GI_TERMS),
        "res": any(t in all_text for t in RESEARCH_TERMS),
        "brand": any(t in all_text for t in BRAND_TERMS),
    }
    name = author or "This creator"
    rationale = ""
    if "Doctor" in mode:
        if score >= 8:
            rationale = f"{name} is highly influential,"
        elif score >= 5:
            rationale = f"{name} has moderate relevance,"
        else:
            rationale = f"{name} does not actively discuss core campaign topics,"
        if tags["onco"]:
            rationale += " frequently engaging in oncology content"
        if tags["gi"]:
            rationale += ", particularly in GI-focused diseases"
        if tags["res"]:
            rationale += " and demonstrating strong research credibility"
        if tags["brand"]:
            rationale += ", mentioning monoclonal therapies or campaign drugs specifically"
        if transcript and "not found" not in transcript.lower():
            rationale += f'. Transcript: "{transcript[:90].strip()}..."'
        else:
            rationale += f". {transcript}"
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if transcript and "not found" not in transcript.lower():
            rationale += f' Transcript: "{transcript[:90].strip()}..."'
        else:
            rationale += f". {transcript}"
    return rationale


def process_posts(
    posts: list,
    transcript_map: Dict[str, str],
    fetch_time: Optional[datetime] = None,
    last_fetch_time: Optional[datetime] = None,
) -> pd.DataFrame:
    results = []
    for post in posts:
        try:
            author = post.get("authorMeta", {}).get("name", "")
            text = post.get("text", "")
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"
            transcript = transcript_map.get(post_id, "Transcript not found")
            ts = datetime.fromtimestamp(post.get("createTime", time.time()))
            body = f"{text} {transcript}"
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, transcript, author, dol_score, sentiment, run_mode)
            is_new = "🟢 New" if last_fetch_time is None or ts > last_fetch_time else "Old"
            results.append(
                {
                    "Author": author,
                    "Text": text.strip(),
                    "Transcript": transcript or "Transcript not found",
                    "Likes": post.get("diggCount", 0),
                    "Views": post.get("playCount", 0),
                    "Comments": post.get("commentCount", 0),
                    "Shares": post.get("shareCount", 0),
                    "Timestamp": ts,
                    "Post URL": url,
                    "DOL Score": dol_score,
                    "Sentiment Score": sentiment_score,
                    "KOL/DOL Status": f"{'🌟' if kol_dol_label == 'KOL' else '👍' if kol_dol_label == 'DOL' else '❌'} {kol_dol_label}",
                    "Brand Sentiment Label": sentiment,
                    "LLM DOL Score Rationale": rationale,
                    "Data Fetched At": fetch_time,
                    "Is New": is_new,
                }
            )
        except Exception as e:
            st.warning(f"Skipped post due to error: {e}")
    return pd.DataFrame(results)


# --- Streamlit UI & main app logic ---
def main():
    global run_mode  # make run_mode available in process_posts and rationale generation
    # Sidebar inputs
    apify_api_key = st.sidebar.text_input(
        "Apify API Token", value=os.getenv("APIFY_API_TOKEN", ""), type="password"
    )
    llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini", "Google Vertex AI"])

    # LLM config depends on provider
    openai_api_key = openai_model = gemini_api_key = gemini_model = gemini_reasoning_effort = gemini_reasoning_summary = None
    vertex_api_key = vertex_model = vertex_project = vertex_location = None
    temperature = 0.6
    max_tokens = 512

    if llm_provider == "OpenAI GPT":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        openai_model = st.sidebar.selectbox(
            "OpenAI Model",
            ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini-2025-04-16"],
        )
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens", 0, 4096, 512)
    elif llm_provider == "Google Gemini":
        gemini_api_key = st.sidebar.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
        )
        gemini_model = st.sidebar.selectbox(
            "Gemini AI Model",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        )
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens", 0, 4096, 512)
        gemini_reasoning_effort = st.sidebar.selectbox(
            "Reasoning Effort", ["None", "Low", "Medium", "High"], index=0
        )
        gemini_reasoning_summary = st.sidebar.selectbox(
            "Reasoning Summary", ["None", "Concise", "Detailed", "Auto"], index=0
        )
    else:
        vertex_api_key = st.sidebar.text_input("Vertex AI API Key (optional)", type="password")
        vertex_project = st.sidebar.text_input("Vertex AI Project ID", os.getenv("VERTEX_PROJECT_ID", ""))
        vertex_location = st.sidebar.text_input("Vertex AI Location", os.getenv("VERTEX_LOCATION", "us-central1"))
        vertex_model = st.sidebar.selectbox("Vertex AI Model", ["text-bison@001", "code-bison@001"])
        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.sidebar.number_input("Max Completion Tokens", 0, 4096, 512)

    st.sidebar.header("Scrape Controls")
    query = st.sidebar.text_input("TikTok Search Term", "doctor")
    target_total = st.sidebar.number_input(
        "Total TikTok Videos", min_value=10, value=200, step=10
    )
    batch_size = st.sidebar.number_input(
        "Batch Size per Run", min_value=10, max_value=200, value=20
    )
    run_mode = st.sidebar.radio(
        "Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"]
    )

    # Initialize session state variables
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = None
    if "tiktok_df" not in st.session_state:
        st.session_state.tiktok_df = pd.DataFrame()
    if "llm_notes_text" not in st.session_state:
        st.session_state.llm_notes_text = ""
    if "llm_score_result" not in st.session_state:
        st.session_state.llm_score_result = ""

    if st.button("Go 🚀", use_container_width=True):
        if not apify_api_key:
            st.error("Apify API Token is required.")
            return
        st.session_state.last_fetch_time = datetime.now()
        posts = run_apify_scraper_batched(apify_api_key, query, target_total, batch_size)
        if not posts:
            st.warning("No TikTok posts found.")
            return
        video_urls = [
            f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}'
            for p in posts
        ]
        transcripts = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
        df = process_posts(posts, transcripts, fetch_time=st.session_state.last_fetch_time, last_fetch_time=None)
        st.session_state.tiktok_df = df
        st.success(f"Fetched and processed {len(df)} TikTok posts.")

    df = st.session_state.tiktok_df
    if not df.empty:
        st.metric("TikTok Posts", len(df))
        st.subheader("📋 TikTok Analysis Results")

        tiktok_cols = [
            "Author", "Text", "Transcript", "Likes", "Views", "Comments", "Shares",
            "DOL Score", "Sentiment Score", "Post URL", "KOL/DOL Status", "Brand Sentiment Label",
            "LLM DOL Score Rationale", "Timestamp", "Data Fetched At", "Is New",
        ]
        display_option = st.radio(
            "Choose display columns:", ["All columns", "Only main info", "Just DOL / Sentiment"]
        )
        if display_option == "All columns":
            columns = tiktok_cols
        elif display_option == "Only main info":
            columns = [
                "Author", "Text", "Likes", "Views", "Comments", "Shares", "DOL Score", "Timestamp", "Is New"
            ]
        else:
            columns = [
                "Author", "KOL/DOL Status", "DOL Score", "Sentiment Score", "Brand Sentiment Label", "Is New"
            ]

        dol_min, dol_max = st.slider("Select DOL Score Range", 1, 10, (1, 10))
        filtered_df = df[(df["DOL Score"] >= dol_min) & (df["DOL Score"] <= dol_max)]
        st.dataframe(filtered_df[columns], use_container_width=True)

        st.download_button(
            "Download TikTok CSV",
            filtered_df[columns].to_csv(index=False),
            file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )

        if st.checkbox("Show Raw TikTok Data"):
            st.subheader("Raw TikTok Data")
            st.dataframe(df, use_container_width=True)

        st.subheader("📝 LLM Notes & Suitability Scoring")
        default_template = """Summary:
Relevance:
Strengths:
Weaknesses:
Red Flags:
Brand Mentions:
Research Notes:
"""
        note_template = st.text_area("Customize LLM Notes Template", value=default_template, height=150)

        if st.button("Generate LLM Vetting Notes"):
            st.session_state.llm_score_result = ""
            with st.spinner("Calling LLM to generate notes..."):
                vertex_client = None
                if llm_provider == "Google Vertex AI":
                    try:
                        vertex_client = get_vertex_client(vertex_project, vertex_location)
                    except Exception as e:
                        st.error(f"Vertex AI client init error: {e}")
                        vertex_client = None

                notes_text = generate_llm_notes(
                    filtered_df,
                    note_template,
                    provider=llm_provider,
                    openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                    gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                    openai_model=openai_model if llm_provider == "OpenAI GPT" else None,
                    openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                    openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                    gemini_model=gemini_model if llm_provider == "Google Gemini" else None,
                    gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                    gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                    gemini_reasoning_effort=gemini_reasoning_effort if llm_provider == "Google Gemini" else None,
                    gemini_reasoning_summary=gemini_reasoning_summary if llm_provider == "Google Gemini" else None,
                    vertex_client=vertex_client,
                    vertex_project=vertex_project if llm_provider == "Google Vertex AI" else None,
                    vertex_location=vertex_location if llm_provider == "Google Vertex AI" else None,
                    vertex_model=vertex_model if llm_provider == "Google Vertex AI" else None,
                    vertex_temperature=temperature if llm_provider == "Google Vertex AI" else 0.6,
                    vertex_max_tokens=max_tokens if llm_provider == "Google Vertex AI" else 512,
                )
                st.session_state.llm_notes_text = notes_text

        if st.session_state.llm_notes_text:
            st.markdown("#### LLM Vetting Notes")
            st.markdown(st.session_state.llm_notes_text)
            st.download_button(
                "Download LLM Vetting Notes",
                st.session_state.llm_notes_text,
                file_name="llm_vetting_notes.txt",
                mime="text/plain",
            )

            if st.button("Generate LLM Score & Rationale"):
                with st.spinner("Calling LLM for scoring..."):
                    vertex_client = None
                    if llm_provider == "Google Vertex AI":
                        try:
                            vertex_client = get_vertex_client(vertex_project, vertex_location)
                        except Exception as e:
                            st.error(f"Vertex AI client init error: {e}")
                            vertex_client = None

                    score_result = generate_llm_score(
                        st.session_state.llm_notes_text,
                        provider=llm_provider,
                        openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                        gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                        openai_model=openai_model if llm_provider == "OpenAI GPT" else None,
                        openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                        openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                        gemini_model=gemini_model if llm_provider == "Google Gemini" else None,
                        gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                        gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                        gemini_reasoning_effort=gemini_reasoning_effort if llm_provider == "Google Gemini" else None,
                        gemini_reasoning_summary=gemini_reasoning_summary if llm_provider == "Google Gemini" else None,
                        vertex_client=vertex_client,
                        vertex_project=vertex_project if llm_provider == "Google Vertex AI" else None,
                        vertex_location=vertex_location if llm_provider == "Google Vertex AI" else None,
                        vertex_model=vertex_model if llm_provider == "Google Vertex AI" else None,
                        vertex_temperature=temperature if llm_provider == "Google Vertex AI" else 0.6,
                        vertex_max_tokens=max_tokens if llm_provider == "Google Vertex AI" else 512,
                    )
                    st.session_state.llm_score_result = score_result

        if st.session_state.llm_score_result:
            st.markdown("#### LLM DOL/KOL Score & Rationale")
            st.code(st.session_state.llm_score_result, language="yaml")


# Helper wrappers for LLM notes & score generation
def generate_llm_notes(*args, **kwargs):
    return get_llm_response(*args, **kwargs)


def generate_llm_score(*args, **kwargs):
    return get_llm_response(*args, **kwargs)


if __name__ == "__main__":
    main()
