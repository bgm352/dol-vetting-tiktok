import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import random
import re
# Dependency imports with clear user errors if missing
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Missing package: rapidfuzz. Please install with 'pip install rapidfuzz'")
    st.stop()
try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Missing package: apify-client. Please install with 'pip install apify-client'")
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
except ModuleNotFoundError:
    st.error("Missing package: textblob. Please install with 'pip install textblob'")
    st.stop()
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")
# Sidebar setup
apify_api_key = st.sidebar.text_input("Apify API Token", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])
with st.sidebar.expander("Advanced Options", expanded=True):
    if llm_provider == "OpenAI GPT":
        model = st.selectbox("AI Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini-2025-04-16"])
        temperature = st.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.number_input("Max Completion Tokens", min_value=0, max_value=4096, value=512)
        openai_api_key = st.text_input("OpenAI API Key", type="password")
    else:
        model = st.selectbox("AI Model", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"])
        temperature = st.slider("Temperature", 0.0, 2.0, 0.6)
        max_tokens = st.number_input("Max Completion Tokens", min_value=0, max_value=4096, value=512)
        reasoning_effort = st.selectbox("Reasoning Effort", ["None", "Low", "Medium", "High"], index=0)
        reasoning_summary = st.selectbox("Reasoning Summary", ["None", "Concise", "Detailed", "Auto"], index=0)
        gemini_api_key = st.text_input("Gemini API Key", type="password")
st.sidebar.header("Scrape Controls")
search_queries_text = st.sidebar.text_area(
    "Enter Search Queries (Doctor name, Specialty, Location) one per line",
    height=200,
    help="Example: Pashtoon Kasi Medical Oncology New York NY"
)
search_queries = [q.strip() for q in search_queries_text.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Total TikTok Videos per Query", min_value=10, value=100, step=10)
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
# Your helper functions here (normalize_name, classify_kol_dol, etc.) unchanged

@st.cache_data(show_spinner=False, persist="disk")
def run_apify_scraper_batched(api_key, query, target_total, batch_size):
    MAX_WAIT_SECONDS = 300  # Increased timeout from 180 to 300 seconds
    result = []
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    offset = 0
    failures = 0
    pbar = st.progress(0.0, text="Scraping TikTok...")
    try:
        while len(result) < target_total and failures < 5:
            st.info(f"Launching batch {1+offset//batch_size}: {len(result)} of {target_total}")
            start_resp = requests.post(run_url, headers={"Authorization": f"Bearer {api_key}"}, json={
                "searchQueries": [query],
                "resultsPerPage": batch_size,
                "searchType": "keyword",
                "pageNumber": offset // batch_size,
            })
            if start_resp.status_code != 200:
                failures += 1
                st.error(f"Batch request failed: {start_resp.status_code} {start_resp.text}")
                time.sleep(5)
                continue
            start = start_resp.json()
            run_id = start.get("data", {}).get("id")
            batch_start = time.time()
            while run_id and (time.time() - batch_start) < MAX_WAIT_SECONDS:
                resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code != 200:
                    failures += 1
                    st.error(f"Failed to get batch status: {resp.status_code}")
                    time.sleep(5)
                    break
                resp_json = resp.json()
                if resp_json.get("data", {}).get("status") == "SUCCEEDED":
                    dataset_id = resp_json["data"].get("defaultDatasetId")
                    break
                time.sleep(5)
            else:
                failures += 1
                st.error("Batch run timed out, retrying.")
                time.sleep(10)  # Backoff delay before retry
                continue
            data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            batch_posts_resp = requests.get(data_url)
            if batch_posts_resp.status_code != 200:
                failures += 1
                st.error(f"Failed to fetch batch posts: {batch_posts_resp.status_code}")
                time.sleep(5)
                continue
            batch_posts = batch_posts_resp.json()
            if not batch_posts:
                failures += 1
                st.error("No posts returned in batch, retrying.")
                time.sleep(10)
                continue
            for p in batch_posts:
                if p not in result:
                    result.append(p)
            offset += batch_size
            pbar.progress(min(1.0, len(result) / float(target_total)))
            if len(batch_posts) < batch_size:
                break
            time.sleep(3)  # Pause to avoid throttling
    except Exception as e:
        st.error(f"Apify error: {e}")
        return []
    pbar.progress(1.0)
    return result[:target_total]

# Keep rest of your code unchanged here...

