import streamlit as st
import requests
import pandas as pd
import time
import json
import threading

# Dependency check imports
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Missing package: rapidfuzz. Install with 'pip install rapidfuzz'")
    st.stop()

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Missing package: apify-client. Install with 'pip install apify-client'")
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
    st.error("Missing package: textblob. Install with 'pip install textblob'")
    st.stop()

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")

# ---------- Helper Functions ----------

@st.cache_data(show_spinner=True, persist="disk")
def run_apify_scraper_batched(api_key, query, target_total, batch_size):
    MAX_WAIT_SECONDS = 300
    result = []
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    offset = 0
    failures = 0
    progress_bar = st.progress(0.0, text="Scraping TikTok...")

    while len(result) < target_total and failures < 5:
        st.info(f"Launching batch {1 + offset // batch_size}: {len(result)} of {target_total}")
        try:
            response = requests.post(run_url, headers={"Authorization": f"Bearer {api_key}"},
                                     json={"searchQueries": [query], "resultsPerPage": batch_size,
                                           "searchType": "keyword", "pageNumber": offset // batch_size},
                                     timeout=60)
            if response.status_code not in [200, 201]:
                failures += 1
                st.error(f"Batch request failed: {response.status_code} {response.text}")
                time.sleep(5)
                continue

            data = response.json()
            run_id = data.get("data", {}).get("id")
            if not run_id:
                failures += 1
                st.error("No run ID returned from Apify.")
                continue

            time.sleep(3)  # Allow Apify to initialize
            start_time = time.time()
            dataset_id = None

            while time.time() - start_time < MAX_WAIT_SECONDS:
                status_resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
                if status_resp.status_code != 200:
                    failures += 1
                    st.error(f"Failed to get batch status: {status_resp.status_code}")
                    time.sleep(5)
                    break

                status_json = status_resp.json()
                status = status_json.get("data", {}).get("status")

                if status == "SUCCEEDED":
                    dataset_id = status_json["data"].get("defaultDatasetId")
                    break
                elif status in ["READY", "RUNNING"]:
                    time.sleep(5)
                else:
                    st.warning(f"Unexpected status: {status}")
                    time.sleep(5)

            if not dataset_id:
                failures += 1
                st.error("No dataset ID returned; skipping batch.")
                continue

            data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            posts_resp = requests.get(data_url, timeout=30)
            if posts_resp.status_code != 200:
                failures += 1
                st.error(f"Failed to fetch batch posts: {posts_resp.status_code}")
                time.sleep(5)
                continue

            posts = posts_resp.json()
            if not posts:
                st.warning("No posts returned in this batch; retrying...")
                time.sleep(10)
                continue

            for post in posts:
                if post not in result:
                    result.append(post)

            offset += batch_size
            progress_bar.progress(min(1.0, len(result) / target_total))
            time.sleep(1)

        except requests.RequestException as e:
            failures += 1
            st.error(f"API error: {e}")
            time.sleep(5)
            continue

    progress_bar.progress(1.0)
    return result[:target_total]

def analyze_and_score(tiktok_data, model="gpt-4", api_provider="OpenAI", temperature=0.3):
    prompt = f"Analyze these TikTok profiles and return a JSON array with: 'profile', 'influence_score' (0-1), 'key_opinion_leader' (true/false):\n{tiktok_data}"
    try:
        if api_provider == "OpenAI":
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
            )
            result_text = response.choices[0].message["content"]
        else:
            # Add Gemini API integration here; placeholder for now
            result_text = "[]"

        scores = json.loads(result_text)
        df = pd.DataFrame(scores)
        expected_cols = {"profile", "influence_score", "key_opinion_leader"}
        if not expected_cols.issubset(df.columns):
            raise ValueError("Missing expected fields in AI response")

        return df

    except json.JSONDecodeError:
        st.error("Could not parse AI model response; please verify formatting.")
    except Exception as e:
        st.error(f"Error during AI vetting: {e}")

    return pd.DataFrame(columns=["profile", "influence_score", "key_opinion_leader"])

def run_async(func, *args):
    result_container = {}

    def wrapper():
        result_container["result"] = func(*args)

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread, result_container

# ---------- UI Components and Workflow ----------

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
search_queries_text = st.sidebar.text_area("Enter Search Queries (Doctor name, Specialty, Location) one per line",
                                           height=200,
                                           help="Example: Pashtoon Kasi Medical Oncology New York NY")
search_queries = [q.strip() for q in search_queries_text.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Total TikTok Videos per Query", min_value=10, value=100, step=10)
batch_size = st.sidebar.number_input("Batch Size per Run", min_value=10, max_value=200, value=20)
run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

if "tiktok_df" not in st.session_state:
    st.session_state["tiktok_df"] = pd.DataFrame()

# Launch Scraper Button & Async Handling
def launch_scraper():
    if not apify_api_key:
        st.warning("Please enter your Apify API token.")
        return
    if not search_queries:
        st.warning("Please enter at least one search query.")
        return

    thread, container = run_async(run_apify_scraper_batched, apify_api_key, search_queries[0], target_total, batch_size)
    with st.spinner("Scraping TikTok... please wait"):
        thread.join()
    results = container.get("result", [])
    if results:
        df = pd.DataFrame(results)
        st.session_state["tiktok_df"] = df
        st.success(f"Scraping complete! {len(results)} posts retrieved.")
        st.dataframe(df)
    else:
        st.error("No results returned from scraper.")

if st.button("🚀 Launch Scraper"):
    launch_scraper()

# Analyze & Score Influence Button & Async Handling
def launch_ai_vetting():
    if "tiktok_df" not in st.session_state or st.session_state["tiktok_df"].empty:
        st.warning("No TikTok data to analyze.")
        return

    api_provider = "OpenAI" if llm_provider == "OpenAI GPT" else "Gemini"
    key = openai_api_key if llm_provider == "OpenAI GPT" else gemini_api_key
    if not key:
        st.warning(f"Enter your {llm_provider} API key first.")
        return

    if llm_provider == "OpenAI GPT":
        openai.api_key = key
    else:
        # Gemini setup placeholder
        pass

    data_to_score = st.session_state["tiktok_df"].to_dict("records")

    thread, container = run_async(analyze_and_score, data_to_score, model=model, api_provider=api_provider, temperature=temperature)
    with st.spinner("Analyzing TikTok data and scoring influence..."):
        thread.join()

    scores_df = container.get("result", pd.DataFrame())
    if not scores_df.empty:
        st.session_state["llm_score_result"] = scores_df
        st.success("AI vetting complete!")
        st.dataframe(scores_df)
        st.download_button("Download Scores CSV", data=scores_df.to_csv(index=False), file_name="influence_scores.csv")
    else:
        st.error("AI vetting returned no results.")

if st.button("Analyze & Score Influence"):
    launch_ai_vetting()



