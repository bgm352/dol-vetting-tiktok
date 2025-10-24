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

st.set_page_config("TikTok Vetting Tool — GPT‑5 + Gemini", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool — GPT‑5 & Gemini 3.0 Pro Ready")

# Utility functions -------------------------------------------------
@st.cache_data(show_spinner=True, persist="disk")
def run_apify_scraper(api_key, hashtag, total, batch_size):
    """Collect TikTok posts by hashtag using updated Apify TikTok Hashtag Scraper."""
    run_url = "https://api.apify.com/v2/acts/epctex~tiktok-hashtag-scraper/runs"
    results, offset, attempts = [], 0, 0
    st.info(f"🕵️ Running scraper for hashtag: #{hashtag}")
    bar = st.progress(0.0, text="Collecting TikTok hashtag posts...")

    while len(results) < total and attempts < 5:
        try:
            response = requests.post(
                run_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "hashtags": [hashtag.replace("#","")],
                    "resultsPerPage": batch_size,
                    "proxyCountryCode": "US",
                    "maxItems": total,
                    "pageNumber": offset // batch_size
                },
                timeout=60,
            )
            if response.status_code not in [200, 201]:
                raise ValueError(f"HTTP {response.status_code}: {response.text}")

            run_id = response.json().get("data", {}).get("id")
            if not run_id:
                raise RuntimeError("No run ID returned by Apify.")

            time.sleep(10)  # Delay for dataset readiness

            # Poll run status until succeeded or timeout
            dataset_id = None
            for _ in range(60):
                status_resp = requests.get(
                    f"{run_url}/{run_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                status_json = status_resp.json()
                status = status_json.get("data", {}).get("status")
                if status == "SUCCEEDED":
                    dataset_id = status_json["data"]["defaultDatasetId"]
                    break
                time.sleep(5)

            if not dataset_id:
                raise TimeoutError("Apify run timed out.")

            dataset_resp = requests.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            batch_items = dataset_resp.json()
            if not batch_items:
                st.warning("No items returned this batch; retrying.")
                time.sleep(10)
                continue

            for item in batch_items:
                if item not in results:
                    results.append(item)

            offset += batch_size
            bar.progress(min(1.0, len(results) / total))
        except Exception as e:
            attempts += 1
            st.warning(f"Attempt {attempts}/5 failed: {e}")
            time.sleep(5)

    bar.progress(1.0)
    return results[:total]

def analyze_llm(tiktok_data, model: str, provider: str, api_key: str, temperature=0.4):
    """Send TikTok data to OpenAI or Google Gemini for vetting."""
    prompt = (
        "Analyze each TikTok profile below and respond only in JSON array format.\n"
        "Each element: {\"profile\": string, \"influence_score\": float 0-1, \"key_opinion_leader\": boolean}\n\n"
        f"{json.dumps(tiktok_data[:5], indent=2)}"
    )

    try:
        if provider == "OpenAI":
            openai.api_key = api_key
            # Use Responses API for GPT-5 models else ChatCompletion
            if model.startswith("gpt-5"):
                response = openai.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_output_tokens=2000,
                    response_format={"type":"json_object"}
                )
                output = response.output_text
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1500,
                )
                output = response.choices[0].message["content"]
        else:
            if not genai:
                raise RuntimeError("Google Gemini SDK is not installed.")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            output = model_obj.generate_content(prompt).text

        data = json.loads(output)
        df = pd.DataFrame(data)
        expected_cols = {"profile", "influence_score", "key_opinion_leader"}
        if not expected_cols.issubset(df.columns):
            raise ValueError("Missing expected fields in AI response.")
        return df
    except json.JSONDecodeError:
        st.error("AI response was not valid JSON.")
    except Exception as e:
        st.error(f"AI vetting error: {e}")
    return pd.DataFrame(columns=["profile", "influence_score", "key_opinion_leader"])

def threaded_run(func, *args):
    """Execute a blocking function in a dedicated thread."""
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

# Sidebar configuration
st.sidebar.header("Configuration")

apify_key = st.sidebar.text_input("Apify API Token", type="password")
provider = st.sidebar.selectbox("Provider", ["OpenAI GPT", "Google Gemini"])

if provider == "OpenAI GPT":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("OpenAI Model", [
        "gpt-5-pro",
        "gpt-5-instant",
        "gpt-o3",
        "gpt-4-turbo",
        "gpt-3.5-turbo"])
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    model = st.sidebar.selectbox("Gemini Model", [
        "gemini-3.0-pro",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-3"])

temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.4)

st.sidebar.header("Scraping Configuration")
query_text = st.sidebar.text_area("TikTok Hashtags (one per line, without # symbol)", help="Example: oncology")
queries = [q.strip() for q in query_text.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Posts per Hashtag", 10, 500, 100)
batch_size = st.sidebar.number_input("Batch Size", 10, 200, 20)

if "tiktok_df" not in st.session_state:
    st.session_state.tiktok_df = pd.DataFrame()

# Scraper launch
def launch_scraper():
    if not apify_key:
        st.warning("Please provide Apify API token.")
        return
    if not queries:
        st.warning("Enter at least one hashtag.")
        return

    all_items = []
    for ht in queries:
        items = threaded_run(run_apify_scraper, apify_key, ht, target_total, batch_size)
        all_items.extend(items)
    df = pd.DataFrame(all_items)
    st.session_state.tiktok_df = df
    if df.empty:
        st.error("Scraper returned no data.")
    else:
        st.success(f"Scraped {len(df)} posts.")
        st.dataframe(df)

# AI vetting launch
def launch_vetting():
    df = st.session_state.get("tiktok_df", pd.DataFrame())
    if df.empty:
        st.warning("No TikTok data available. Please run the scraper first.")
        return

    if provider == "OpenAI GPT" and not openai_key:
        st.warning("Enter your OpenAI API key.")
        return
    if provider == "Google Gemini" and not gemini_key:
        st.warning("Enter your Gemini API key.")
        return

    key = openai_key if provider == "OpenAI GPT" else gemini_key

    with st.spinner("Running AI vetting..."):
        results = threaded_run(analyze_llm, df.to_dict("records"), model, provider.split()[0], key, temperature)
        if results.empty:
            st.error("AI vetting produced no output.")
        else:
            st.session_state.vetting_df = results
            st.success("✅ Vetting complete.")
            st.dataframe(results)
            st.download_button("Download vetting results CSV", results.to_csv(index=False), "vetting_results.csv", "text/csv")

# UI Buttons
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Run TikTok Scraper"):
        launch_scraper()
with col2:
    if st.button("🤖 Run AI Vetting"):
        launch_vetting()


