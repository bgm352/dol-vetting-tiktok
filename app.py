import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk
import random
import re

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Install `rapidfuzz` by running: pip install rapidfuzz")
    st.stop()

try:
    import jellyfish
except ModuleNotFoundError:
    st.error("Install `jellyfish` by running: pip install jellyfish")
    st.stop()

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Install `apify-client` by running: pip install apify-client")
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
    st.error("Install `textblob` and `nltk` by running: pip install textblob nltk")
    st.stop()

st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")

apify_api_key = st.sidebar.text_input("Apify API Token", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])
with st.sidebar.expander("Advanced Options: Model and Generation Settings", expanded=True):
    if llm_provider == "OpenAI GPT":
        model = st.selectbox(
            "AI Model",
            ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini-2025-04-16"],
            help="Choose the OpenAI model for generation. gpt-4o-mini is an optimized variant.",
        )
        temperature = st.slider(
            "Temperature (Optional)", 0.0, 2.0, 0.6, help="Controls randomness. Lower values produce more deterministic output."
        )
        max_tokens = st.number_input(
            "Max Completion Tokens (Optional)", min_value=0, max_value=4096, value=512, help="Maximum tokens in the response. 0 means no limit."
        )
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.info("Max tokens limits generation length. Use higher for detailed responses.")
    else:
        model = st.selectbox(
            "AI Model",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="Select Google Gemini model variant.",
        )
        temperature = st.slider(
            "Temperature (Optional)", 0.0, 2.0, 0.6, help="Controls randomness of the output."
        )
        max_tokens = st.number_input(
            "Max Completion Tokens (Optional)", min_value=0, max_value=4096, value=512, help="Max tokens limit; 0 means no limit."
        )
        reasoning_effort = st.selectbox(
            "Reasoning Effort", ["None", "Low", "Medium", "High"], index=0, help="Set amount of reasoning effort."
        )
        reasoning_summary = st.selectbox(
            "Reasoning Summary", ["None", "Concise", "Detailed", "Auto"], index=0, help="Whether to include reasoning summary."
        )
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        st.info("Reasoning settings affect output depth and length.")

st.sidebar.header("Scrape Controls")
st.sidebar.subheader("Batch Doctor Vetting")
doctor_names_text = st.sidebar.text_area(
    "Enter Doctor Full Names (one per line)", height=150, help="Enter full names (First Last) for batch vetting with fuzzy + phonetic matching"
)
doctor_names = [doc.strip() for doc in doctor_names_text.splitlines() if doc.strip()]
query = st.sidebar.text_input("TikTok Search Term (used if no doctors input)", "doctor")
target_total = st.sidebar.number_input("Total TikTok Videos per Doctor", min_value=10, value=100, step=10)
batch_size = st.sidebar.number_input("Batch Size per Run", min_value=10, max_value=200, value=20)
run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

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
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def phonetic_match(name1: str, name2: str) -> bool:
    return jellyfish.metaphone(name1) == jellyfish.metaphone(name2)

def is_name_match(input_name: str, candidate_name: str, thresh_token=85, thresh_partial=75) -> bool:
    input_norm = normalize_name(input_name)
    candidate_norm = normalize_name(candidate_name)
    token_score = fuzz.token_sort_ratio(input_norm, candidate_norm)
    partial_score = fuzz.partial_ratio(input_norm, candidate_norm)
    phonetic = phonetic_match(input_norm, candidate_norm)
    return (token_score >= thresh_token or partial_score >= thresh_partial or phonetic)

def filter_posts_by_doctor(posts, doctor_full_name):
    filtered = []
    for post in posts:
        author_name = post.get("authorMeta", {}).get("name", "")
        if is_name_match(doctor_full_name, author_name):
            filtered.append(post)
    return filtered

def classify_kol_dol(score):
    if score >= 8:
        return "KOL"
    elif score >= 5:
        return "DOL"
    else:
        return "Not Suitable"

def classify_sentiment(score):
    if score > 0.15:
        return "Positive"
    elif score < -0.15:
        return "Negative"
    else:
        return "Neutral"

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
    MAX_WAIT_SECONDS = 180
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
                break
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
            results.append({
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
            })
        except Exception as e:
            st.warning(f"⛔ Skipped 1 post: {e}")
            continue
    return pd.DataFrame(results)

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
            similarity_data = []
            for p in raw_posts:
                author_name = p.get("authorMeta", {}).get("name", "")
                token_score = fuzz.token_sort_ratio(normalize_name(doc_name), normalize_name(author_name))
                partial_score = fuzz.partial_ratio(normalize_name(doc_name), normalize_name(author_name))
                phonetics_match = phonetic_match(normalize_name(doc_name), normalize_name(author_name))
                similarity_data.append({
                    "Author": author_name,
                    "Token Sort Score": token_score,
                    "Partial Ratio": partial_score,
                    "Phonetic Match": phonetics_match,
                    "Considered Match": token_score >= 85 or partial_score >= 75 or phonetics_match
                })
            with st.expander(f"Fuzzy & Phonetic Match Scores for '{doc_name}'"):
                sim_df = pd.DataFrame(similarity_data).sort_values(by=["Considered Match", "Token Sort Score", "Partial Ratio"], ascending=[False, False, False])
                st.dataframe(sim_df)
            matched_posts = filter_posts_by_doctor(raw_posts, doc_name)
            if not matched_posts:
                st.warning(f"No TikTok posts matched fuzzy or phonetic author name for {doc_name}.")
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
    if st.button("What does the DOL/KOL score mean?"):
        st.info(
            """The DOL (Digital Opinion Leader) score is a numerical rating (1–10) that evaluates the relevance and influence of a TikTok content creator in the context of healthcare vetting.
- **Higher scores (8-10)** indicate top-tier key opinion leaders (KOLs) with strong expertise and engagement.
- **Mid-range scores (5-7)** suggest moderate influence or emerging leaders (DOLs).
- **Lower scores (1-4)** represent creators less suitable for campaigns.
The score is generated based on sentiment, content relevance, and other factors analyzed by AI."""
        )
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




