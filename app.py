import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk
import random
import re

# Install and import rapidfuzz for fuzzy string matching
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Install `rapidfuzz` (`pip install rapidfuzz`).")
    st.stop()

# Setup & Package Guards
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
    from textblob import TextBlob
    nltk.download("punkt")
except ModuleNotFoundError:
    st.error("Install `textblob` and `nltk`.")
    st.stop()

st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")

# --- Sidebar Params ---
apify_api_key = st.sidebar.text_input("Apify API Token", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])

# Advanced Options Expander
with st.sidebar.expander("Advanced Options: Model and Generation Settings", expanded=True):
    if llm_provider == "OpenAI GPT":
        model = st.selectbox(
            "AI Model",
            ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini-2025-04-16"],
            help="Choose the OpenAI model for generation. gpt-4o-mini is an optimized variant.",
        )
        temperature = st.slider(
            "Temperature (Optional)",
            0.0,
            2.0,
            0.6,
            help="Controls randomness. Lower values produce more deterministic output.",
        )
        max_tokens = st.number_input(
            "Max Completion Tokens (Optional)",
            min_value=0,
            max_value=4096,
            value=512,
            help="Maximum tokens in the response. 0 means no limit.",
        )
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.info("Max tokens limits generation length. Use higher for detailed responses.")
    else:  # Google Gemini
        model = st.selectbox(
            "AI Model",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="Select Google Gemini model variant.",
        )
        temperature = st.slider(
            "Temperature (Optional)",
            0.0,
            2.0,
            0.6,
            help="Controls randomness of the output.",
        )
        max_tokens = st.number_input(
            "Max Completion Tokens (Optional)",
            min_value=0,
            max_value=4096,
            value=512,
            help="Max tokens limit; 0 means no limit.",
        )
        reasoning_effort = st.selectbox(
            "Reasoning Effort",
            ["None", "Low", "Medium", "High"],
            index=0,
            help="Set amount of reasoning effort: None=off, Low to High increase complexity.",
        )
        reasoning_summary = st.selectbox(
            "Reasoning Summary",
            ["None", "Concise", "Detailed", "Auto"],
            index=0,
            help="Whether to include reasoning summary. None=no summary, Concise=brief, Detailed=full, Auto=auto decision.",
        )
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        st.info("Reasoning settings affect output depth and length.")
        
st.sidebar.header("Scrape Controls")

# Multi-line input for batch doctor names
st.sidebar.subheader("Batch Doctor Vetting")
doctor_names_text = st.sidebar.text_area(
    "Enter Doctor Full Names (one per line)",
    height=150,
    help="Enter full names (First Last) for batch vetting with fuzzy matching"
)
doctor_names = [doc.strip() for doc in doctor_names_text.splitlines() if doc.strip()]

query = st.sidebar.text_input("TikTok Search Term (used if no doctors input)", "doctor")
target_total = st.sidebar.number_input(
    "Total TikTok Videos per Doctor", min_value=10, value=100, step=10
)
batch_size = st.sidebar.number_input(
    "Batch Size per Run", min_value=10, max_value=200, value=20
)
run_mode = st.sidebar.radio(
    "Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"]
)

ONCOLOGY_TERMS = [
    "oncology",
    "cancer",
    "monoclonal",
    "checkpoint",
    "immunotherapy",
]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = [
    "biomarker",
    "clinical trial",
    "abstract",
    "network",
    "congress",
]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

# ---- State initialization ----
for key, default in [
    ("top_kols", []),
    ("analysis_count", 0),
    ("feedback_logs", []),
    ("last_fetch_time", None),
    ("llm_notes_text", ""),
    ("llm_score_result", ""),
    ("tiktok_df", pd.DataFrame()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def normalize_name(name: str) -> str:
    # Lowercase, remove punctuation, normalize spaces for fuzzy matching
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name

def is_name_match(input_name: str, candidate_name: str, threshold: int = 85) -> bool:
    """
    Return True if candidate_name matches input_name approximately at or above threshold %
    Uses token sort ratio to ignore word order differences.
    """
    input_name_norm = normalize_name(input_name)
    candidate_name_norm = normalize_name(candidate_name)
    score = fuzz.token_sort_ratio(input_name_norm, candidate_name_norm)
    return score >= threshold

def filter_posts_by_doctor(posts, doctor_full_name, threshold=85):
    """
    Filter TikTok posts where author's name fuzzy matches the doctor full name.
    """
    filtered = []
    for post in posts:
        author = post.get("authorMeta", {}).get("name", "")
        if is_name_match(doctor_full_name, author, threshold):
            filtered.append(post)
    return filtered

def classify_kol_dol(score):
    return "KOL" if score >= 8 else "DOL" if score >= 5 else "Not Suitable"

def classify_sentiment(score):
    return "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"

def generate_rationale(text, transcript, author, score, sentiment, mode):
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
        if transcript and "not found" not in transcript:
            rationale += f'. Transcript: “{transcript[:90].strip()}...”'
        else:
            rationale += f". {transcript}"
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if transcript and "not found" not in transcript:
            rationale += f' Transcript: “{transcript[:90].strip()}...”'
        else:
            rationale += f". {transcript}"
    return rationale

def retry_with_backoff(func, max_retries=3, base_delay=2):
    def wrapper(*args, **kwargs):
        attempt = 0
        last_exception = None
        while attempt < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                attempt += 1
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(delay)
        raise last_exception
    return wrapper

@retry_with_backoff
def call_openai(prompt, openai_api_key, model, temperature, max_tokens):
    if not openai:
        return "OpenAI SDK not installed."
    if not openai_api_key:
        return "No OpenAI key."
    client = openai.OpenAI(api_key=openai_api_key)
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
    prompt,
    gemini_api_key,
    model,
    temperature,
    max_tokens,
    reasoning_effort=None,
    reasoning_summary=None,
):
    if not genai:
        return "Gemini SDK not installed."
    if not gemini_api_key:
        return "No Gemini key."
    genai.configure(api_key=gemini_api_key)
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

def get_llm_response(
    prompt,
    provider,
    openai_api_key=None,
    gemini_api_key=None,
    openai_model=None,
    openai_temperature=0.6,
    openai_max_tokens=512,
    gemini_model=None,
    gemini_temperature=0.6,
    gemini_max_tokens=512,
    gemini_reasoning_effort=None,
    gemini_reasoning_summary=None,
):
    try:
        if provider == "OpenAI GPT":
            return call_openai(
                prompt, openai_api_key, openai_model, openai_temperature, openai_max_tokens
            )
        elif provider == "Google Gemini":
            return call_gemini(
                prompt,
                gemini_api_key,
                gemini_model,
                gemini_temperature,
                gemini_max_tokens,
                gemini_reasoning_effort,
                gemini_reasoning_summary,
            )
        else:
            return "Unknown provider"
    except Exception as e:
        return f"{provider} call failed: {e}"

def generate_llm_notes(
    posts_df,
    note_template,
    provider,
    openai_api_key=None,
    gemini_api_key=None,
    openai_model=None,
    openai_temperature=0.6,
    openai_max_tokens=512,
    gemini_model=None,
    gemini_temperature=0.6,
    gemini_max_tokens=512,
    gemini_reasoning_effort=None,
    gemini_reasoning_summary=None,
):
    posts_texts = "\n\n".join(
        [
            f"{i+1}. Author: {row['Author']}\nContent: {row['Text']}\nTranscript: {row['Transcript']}"
            for i, row in posts_df.iterrows()
        ]
    )
    prompt = f"""Using the following social posts, generate notes for KOL/DOL vetting in this structure:
{note_template}
Social posts:
{posts_texts}
Return in markdown, each section with a title."""
    return get_llm_response(prompt, provider, openai_api_key, gemini_api_key,
                            openai_model, openai_temperature, openai_max_tokens,
                            gemini_model, gemini_temperature, gemini_max_tokens,
                            gemini_reasoning_effort, gemini_reasoning_summary)

def generate_llm_score(
    notes,
    provider,
    openai_api_key=None,
    gemini_api_key=None,
    openai_model=None,
    openai_temperature=0.6,
    openai_max_tokens=512,
    gemini_model=None,
    gemini_temperature=0.6,
    gemini_max_tokens=512,
    gemini_reasoning_effort=None,
    gemini_reasoning_summary=None,
):
    prompt = f"""You are a medical affairs expert. Based on these vetting notes and evidence, assign a DOL suitability score (1=poor, 10=ideal) for pharma and give a rationale.
Notes: {notes}
Respond in YAML:
score: <1-10>
rationale: <short explanation>
"""
    return get_llm_response(prompt, provider, openai_api_key, gemini_api_key,
                            openai_model, openai_temperature, openai_max_tokens,
                            gemini_model, gemini_temperature, gemini_max_tokens,
                            gemini_reasoning_effort, gemini_reasoning_summary)

def fetch_tiktok_transcripts_apify(api_token, video_urls):
    client = ApifyClient(api_token)
    run_input = {"videos": video_urls}
    run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input=run_input)
    transcripts = {}
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        video_id = item.get("id")
        transcript = item.get("transcript", "")
        transcripts[video_id] = transcript
    return transcripts

@st.cache_data(show_spinner=False, persist="disk")
def run_apify_scraper_batched(api_key, query, target_total, batch_size):
    MAX_WAIT_SECONDS = 180  # 3 minutes batch hard timeout
    result = []
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    offset = 0
    failures = 0
    pbar = st.progress(0.0, text="Scraping TikTok...")
    try:
        while len(result) < target_total and failures < 5:
            st.info(f"Launching batch {1+offset//batch_size}: {len(result)} of {target_total}")
            start = requests.post(
                run_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "searchQueries": [query],
                    "resultsPerPage": batch_size,
                    "searchType": "keyword",
                    "pageNumber": offset // batch_size,
                },
            ).json()
            run_id = start.get("data", {}).get("id")
            batch_start = time.time()
            while run_id and (time.time() - batch_start) < MAX_WAIT_SECONDS:
                resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}).json()
                if resp.get("data", {}).get("status") == "SUCCEEDED":
                    dataset_id = resp["data"].get("defaultDatasetId")
                    break
                time.sleep(5)
            else:
                failures += 1
                st.error("Batch run timed out, retrying.")
                continue
            data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            batch_posts = requests.get(data_url).json()
            if not batch_posts:
                failures += 1
                st.error("No posts after batch, retrying.")
                continue
            for p in batch_posts:
                if p not in result:
                    result.append(p)
            offset += batch_size
            pbar.progress(min(1.0, len(result) / float(target_total)))
            if len(batch_posts) < batch_size:
                break  # exhausted
    except Exception as e:
        st.error(f"Apify error: {e}")
        return []
    pbar.progress(1.0)
    return result[:target_total]

def process_posts(posts, transcript_map, fetch_time=None, last_fetch_time=None):
    results = []
    for i, post in enumerate(posts):
        try:
            author = post.get("authorMeta", {}).get("name", "")
            text = post.get("text", "")
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"
            tscript = transcript_map.get(post_id, "Transcript not found")
            ts = datetime.fromtimestamp(post.get("createTime", time.time()))
            body = f"{text} {tscript}"
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, tscript, author, dol_score, sentiment, run_mode)
            is_new = "🟢 New" if last_fetch_time is None or ts > last_fetch_time else "Old"
            results.append(
                {
                    "Author": author,
                    "Text": text.strip(),
                    "Transcript": tscript or "Transcript not found",
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
            st.warning(f"⛔ Skipped 1 post: {e}")
            continue
    return pd.DataFrame(results)

# --- MAIN APP FLOW ---
if st.button("Go 🚀", use_container_width=True) and apify_api_key:
    st.session_state["analysis_count"] += 1
    fetch_time = datetime.now()
    last_fetch_time = st.session_state["last_fetch_time"]

    all_results = []

    if doctor_names:
        for doc_name in doctor_names:
            st.info(f"Searching posts for doctor: {doc_name}")
            raw_posts = run_apify_scraper_batched(apify_api_key, doc_name, int(target_total), int(batch_size))
            if not raw_posts:
                st.warning(f"No TikTok posts found for {doc_name}.")
                continue

            # Debugging: Show author similarity scores in expander
            similarity_data = []
            for p in raw_posts:
                author_name = p.get("authorMeta", {}).get("name", "")
                score = fuzz.token_sort_ratio(normalize_name(doc_name), normalize_name(author_name))
                similarity_data.append({"Author": author_name, "Similarity Score": score})
            with st.expander(f"Show Fuzzy Match Scores for '{doc_name}'"):
                sim_df = pd.DataFrame(similarity_data).sort_values(by="Similarity Score", ascending=False)
                st.dataframe(sim_df)

            matched_posts = filter_posts_by_doctor(raw_posts, doc_name, threshold=85)
            if not matched_posts:
                st.warning(f"No TikTok posts matched fuzzy author name for {doc_name}.")
                continue

            video_urls = [
                f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}'
                for p in matched_posts
            ]
            transcript_map = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
            df = process_posts(matched_posts, transcript_map=transcript_map, fetch_time=fetch_time, last_fetch_time=last_fetch_time)
            df["Doctor Name"] = doc_name
            all_results.append(df)
    else:
        tiktok_data = run_apify_scraper_batched(apify_api_key, query, int(target_total), int(batch_size))
        if not tiktok_data:
            st.warning("No TikTok posts found.")
        else:
            video_urls = [
                f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}'
                for p in tiktok_data
            ]
            transcript_map = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
            df = process_posts(tiktok_data, transcript_map=transcript_map, fetch_time=fetch_time, last_fetch_time=last_fetch_time)
            all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        st.session_state["last_fetch_time"] = fetch_time
        st.session_state["tiktok_df"] = combined_df
    else:
        st.session_state["tiktok_df"] = pd.DataFrame()

df = st.session_state.get("tiktok_df", pd.DataFrame())

if not df.empty:
    st.metric("TikTok Posts Analyzed", len(df))
    st.subheader("📋 TikTok Analysis Results")

    if doctor_names:
        unique_doctors = df["Doctor Name"].unique().tolist()
        selected_doctor = st.selectbox("Filter by Doctor", ["All"] + unique_doctors)
        if selected_doctor != "All":
            df = df[df["Doctor Name"] == selected_doctor]

    tiktok_cols = [
        "Doctor Name",
        "Author",
        "Text",
        "Transcript",
        "Likes",
        "Views",
        "Comments",
        "Shares",
        "DOL Score",
        "Sentiment Score",
        "Post URL",
        "KOL/DOL Status",
        "Brand Sentiment Label",
        "LLM DOL Score Rationale",
        "Timestamp",
        "Data Fetched At",
        "Is New",
    ]

    display_option = st.radio(
        "Choose display columns:", ["All columns", "Only main info", "Just DOL / Sentiment"]
    )
    if display_option == "All columns":
        columns = tiktok_cols
    elif display_option == "Only main info":
        columns = [
            "Doctor Name",
            "Author",
            "Text",
            "Likes",
            "Views",
            "Comments",
            "Shares",
            "DOL Score",
            "Timestamp",
            "Is New",
        ]
    else:
        columns = [
            "Doctor Name",
            "Author",
            "KOL/DOL Status",
            "DOL Score",
            "Sentiment Score",
            "Brand Sentiment Label",
            "Is New",
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

    # Add popup info for DOL/KOL score explanation
    if st.button("What does the DOL/KOL score mean?"):
        st.info(
            """The DOL (Digital Opinion Leader) score is a numerical rating (1–10) that evaluates the relevance and influence of a TikTok content creator in the context of healthcare vetting.
- **Higher scores (8-10)** indicate top-tier key opinion leaders (KOLs) with strong expertise and engagement.
- **Mid-range scores (5-7)** suggest moderate influence or emerging leaders (DOLs).
- **Lower scores (1-4)** represent creators less suitable for campaigns.
The score is generated based on sentiment, content relevance, and other factors analyzed by AI."""
        )

    # --- LLM Notes and Scoring (STATEFUL) ---
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
        with st.spinner("Calling LLM to generate notes..."):
            notes_text = generate_llm_notes(
                filtered_df,
                note_template,
                provider=llm_provider,
                openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                openai_model=model if llm_provider == "OpenAI GPT" else None,
                openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                gemini_model=model if llm_provider == "Google Gemini" else None,
                gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                gemini_reasoning_effort=reasoning_effort if llm_provider == "Google Gemini" else None,
                gemini_reasoning_summary=reasoning_summary if llm_provider == "Google Gemini" else None,
            )
        st.session_state["llm_notes_text"] = notes_text
        st.session_state["llm_score_result"] = ""
    if st.session_state["llm_notes_text"]:
        st.markdown("#### LLM Vetting Notes")
        st.markdown(st.session_state["llm_notes_text"])
        st.download_button(
            label="Download LLM Vetting Notes",
            data=st.session_state["llm_notes_text"],
            file_name="llm_vetting_notes.txt",
            mime="text/plain",
        )
        if st.button("Generate LLM Score & Rationale"):
            with st.spinner("Calling LLM for scoring..."):
                score_result = generate_llm_score(
                    st.session_state["llm_notes_text"],
                    provider=llm_provider,
                    openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                    gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None,
                    openai_model=model if llm_provider == "OpenAI GPT" else None,
                    openai_temperature=temperature if llm_provider == "OpenAI GPT" else 0.6,
                    openai_max_tokens=max_tokens if llm_provider == "OpenAI GPT" else 512,
                    gemini_model=model if llm_provider == "Google Gemini" else None,
                    gemini_temperature=temperature if llm_provider == "Google Gemini" else 0.6,
                    gemini_max_tokens=max_tokens if llm_provider == "Google Gemini" else 512,
                    gemini_reasoning_effort=reasoning_effort if llm_provider == "Google Gemini" else None,
                    gemini_reasoning_summary=reasoning_summary if llm_provider == "Google Gemini" else None,
                )
            st.session_state["llm_score_result"] = score_result
    if st.session_state["llm_score_result"]:
        st.markdown("#### LLM DOL/KOL Score & Rationale")
        st.code(st.session_state["llm_score_result"], language="yaml")




