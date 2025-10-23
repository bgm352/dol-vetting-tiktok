import streamlit as st
import requests
import pandas as pd
import time
import json
import threading

# Dependency checks -------------------------------------------------
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Missing package: rapidfuzz. Run 'pip install rapidfuzz'")
    st.stop()

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Missing package: apify-client. Run 'pip install apify-client'")
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
    st.error("Missing package: textblob. Run 'pip install textblob'")
    st.stop()

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ------------------------------------------------------------------
# App configuration
# ------------------------------------------------------------------
st.set_page_config("TikTok Vetting Tool — GPT‑5 + Gemini", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool — GPT‑5 & Gemini 3.0 Pro Ready")

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
@st.cache_data(show_spinner=True, persist="disk")
def run_apify_scraper(api_key, query, total, batch_size):
    """Collect TikTok posts using the Apify TikTok Scraper."""
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    results, offset, attempts = [], 0, 0
    st.info(f"🕵️ Running scraper for: {query}")
    bar = st.progress(0.0, text="Collecting TikTok data...")

    while len(results) < total and attempts < 5:
        try:
            response = requests.post(
                run_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "searchQueries": [query],
                    "resultsPerPage": batch_size,
                    "searchType": "keyword",
                    "pageNumber": offset // batch_size,
                },
                timeout=60,
            )
            if response.status_code not in [200, 201]:
                raise ValueError(response.text)

            run_id = response.json().get("data", {}).get("id")
            if not run_id:
                raise RuntimeError("No run ID returned by Apify.")
            time.sleep(3)

            # Poll Apify run
            dataset_id = None
            for _ in range(60):
                status_check = requests.get(
                    f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}
                ).json()
                status = status_check.get("data", {}).get("status")
                if status == "SUCCEEDED":
                    dataset_id = status_check["data"]["defaultDatasetId"]
                    break
                time.sleep(5)

            if not dataset_id:
                raise TimeoutError("Apify run timed out.")

            dataset = requests.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            ).json()
            results.extend(dataset)
            offset += batch_size
            bar.progress(min(1.0, len(results) / total))
        except Exception as e:
            attempts += 1
            st.warning(f"Retry {attempts}/5 after error: {e}")
            time.sleep(5)

    bar.progress(1.0)
    return results[:total]


def analyze_llm(tiktok_data, model: str, provider: str, api_key: str, temperature=0.4):
    """Send TikTok data to LLM (OpenAI GPT or Google Gemini)."""
    prompt = (
        "Analyze each TikTok profile below and respond strictly in JSON array form "
        "as [{'profile': str, 'influence_score': float, 'key_opinion_leader': bool}].\n\n"
        f"Profiles:\n{json.dumps(tiktok_data[:5], indent=2)}"
    )
    try:
        if provider == "OpenAI":
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
            )
            output = resp.choices[0].message["content"]
        else:
            if not genai:
                raise RuntimeError("Gemini SDK not installed.")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            output = model_obj.generate_content(prompt).text

        data_json = json.loads(output)
        df = pd.DataFrame(data_json)
        expected = {"profile", "influence_score", "key_opinion_leader"}
        if not expected.issubset(df.columns):
            raise ValueError("Missing expected columns in model output.")
        return df
    except json.JSONDecodeError:
        st.error("Model output was not valid JSON.")
    except Exception as e:
        st.error(f"Problem during AI analysis: {e}")
    return pd.DataFrame(columns=["profile", "influence_score", "key_opinion_leader"])


def threaded_run(func, *args):
    """Run function in a separate thread."""
    result = {}

    def wrapper():
        try:
            result["data"] = func(*args)
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=wrapper)
    t.start()
    t.join()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("data")

# ------------------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------------------
st.sidebar.header("Configuration")

apify_key = st.sidebar.text_input("Apify API Token", type="password")
provider = st.sidebar.selectbox("Choose Provider", ["OpenAI GPT", "Google Gemini"])

if provider == "OpenAI GPT":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox(
        "OpenAI Model",
        [
            "gpt-5-pro",
            "gpt-5-instant",
            "gpt-o3",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        index=0,
    )
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    model = st.sidebar.selectbox(
        "Gemini Model",
        [
            "gemini-3.0-pro",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemma-3",
        ],
        index=0,
    )

temp = st.sidebar.slider("Temperature", 0.0, 2.0, 0.4)
st.sidebar.divider()
st.sidebar.header("Scraper")
queries_txt = st.sidebar.text_area("TikTok Search Queries", help="One per line.")
queries = [q.strip() for q in queries_txt.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Posts per Query", 10, 500, 100)
batch_size = st.sidebar.number_input("Batch Size", 10, 200, 20)

if "tiktok_df" not in st.session_state:
    st.session_state.tiktok_df = pd.DataFrame()

# ------------------------------------------------------------------
# Launch Scraper
# ------------------------------------------------------------------
def start_scraper():
    if not apify_key:
        st.warning("Please provide a valid Apify token.")
        return
    if not queries:
        st.warning("Enter search queries.")
        return

    all_data = []
    for q in queries:
        data = threaded_run(run_apify_scraper, apify_key, q, target_total, batch_size)
        all_data.extend(data)
    df = pd.DataFrame(all_data)
    st.session_state.tiktok_df = df
    if df.empty:
        st.error("Scraper returned no data.")
    else:
        st.success(f"Scraped {len(df)} posts.")
        st.dataframe(df)

# ------------------------------------------------------------------
# Launch Vetting
# ------------------------------------------------------------------
def start_vetting():
    df = st.session_state.get("tiktok_df", pd.DataFrame())
    if df.empty:
        st.warning("No TikTok data available.")
        return

    if provider == "OpenAI GPT" and not openai_key:
        st.warning("Enter your OpenAI API key.")
        return
    if provider == "Google Gemini" and not gemini_key:
        st.warning("Enter your Gemini API key.")
        return

    st.spinner("Running AI vetting... please wait.")
    key = openai_key if provider == "OpenAI GPT" else gemini_key
    df_results = threaded_run(analyze_llm, df.to_dict("records"), model, provider.split()[0], key, temp)

    if df_results.empty:
        st.error("AI vetting produced no output.")
    else:
        st.session_state.vetting_df = df_results
        st.success("✅ Vetting complete.")
        st.dataframe(df_results)
        st.download_button(
            "Download Results CSV",
            data=df_results.to_csv(index=False),
            file_name="vetting_results.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------
# Buttons
# ------------------------------------------------------------------
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Run TikTok Scraper"):
        start_scraper()
with col2:
    if st.button("🤖 Run Vetting with AI"):
        start_vetting()



