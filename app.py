import streamlit as st
import requests
import pandas as pd
import json
import time
import threading

# Dependency Checks
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    st.error("Please install 'rapidfuzz'.")
    st.stop()

try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Please install 'apify-client'.")
    st.stop()

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

st.set_page_config(page_title="TikTok Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok Vetting & AI Scoring Tool")

@st.cache_data(show_spinner=True, persist="disk")
def run_apify_scraper(api_key, query, target_total, batch_size):
    run_url = "https://api.apify.com/v2/acts/epctex~tiktok-search-scraper/runs"
    results = []
    offset = 0
    attempts = 0
    progress_bar = st.progress(0.0, text="Starting scraping...")

    while len(results) < target_total and attempts < 5:
        try:
            response = requests.post(
                run_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"searchTerms": [query],
                      "resultsPerPage": batch_size,
                      "proxyCountryCode": "US",
                      "maxItems": target_total,
                      "pageNumber": offset // batch_size,
                      "searchType": "keyword"},
                timeout=60
            )
            response.raise_for_status()

            run_id = response.json().get("data", {}).get("id")
            if not run_id:
                raise RuntimeError("No run ID received")

            time.sleep(10)  # wait for dataset readiness

            dataset_id = None
            for _ in range(60):
                status_resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"})
                status_resp.raise_for_status()
                status_data = status_resp.json().get("data", {})
                status = status_data.get("status")
                if status == "SUCCEEDED":
                    dataset_id = status_data.get("defaultDatasetId")
                    break
                time.sleep(5)

            if not dataset_id:
                raise TimeoutError("Scraper actor timed out")

            dataset_resp = requests.get(f"https://api.apify.com/v2/datasets/{dataset_id}/items", headers={"Authorization": f"Bearer {api_key}"})
            dataset_resp.raise_for_status()
            batch_posts = dataset_resp.json()

            for post in batch_posts:
                if post not in results:
                    results.append(post)

            offset += batch_size
            progress_bar.progress(min(1.0, len(results) / target_total))

        except Exception as e:
            attempts += 1
            st.warning(f"Scraper attempt {attempts} failed: {e}")
            time.sleep(5)

    progress_bar.progress(1.0)
    return results[:target_total]

def analyze_llm(tiktok_data, model, provider, api_key, temperature=0.4):
    prompt = ("Analyze the TikTok profiles below and return a JSON list of "
              "objects with fields: 'profile', 'influence_score' (0-1), and "
              "'key_opinion_leader' (true/false).\n" + json.dumps(tiktok_data[:5], indent=2))
    try:
        if provider == "OpenAI":
            openai.api_key = api_key
            if model.startswith("gpt-5"):
                response = openai.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_output_tokens=2000,
                    response_format={"type": "json_object"}
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
                raise RuntimeError("Gemini SDK not installed")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            output = model_obj.generate_content(prompt).text

        data = json.loads(output)
        df = pd.DataFrame(data)
        expected_cols = {"profile", "influence_score", "key_opinion_leader"}
        if not expected_cols.issubset(df.columns):
            raise ValueError("Missing fields in AI response")
        return df

    except json.JSONDecodeError:
        st.error("AI response was not valid JSON.")
    except Exception as e:
        st.error(f"AI vetting failed: {e}")
    return pd.DataFrame(columns=["profile", "influence_score", "key_opinion_leader"])

def run_in_thread(func, *args):
    result = {}

    def runner():
        try:
            result["data"] = func(*args)
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("data")

# Sidebar configuration
st.sidebar.header("Configuration")
apify_key = st.sidebar.text_input("Apify API Token", type="password")
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])

if provider == "OpenAI GPT":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("OpenAI Model", [
        "gpt-5-pro", "gpt-5-instant", "gpt-o3", "gpt-4-turbo", "gpt-3.5-turbo"])
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    model = st.sidebar.selectbox("Gemini Model", [
        "gemini-3.0-pro", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemma-3"])

temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.4)
st.sidebar.header("TikTok Search Settings")
search_input = st.sidebar.text_area("Enter TikTok Search Keywords (one per line)", help="Keyword or phrase to search")
queries = [q.strip() for q in search_input.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Posts per Query", 10, 500, 100)
batch_size = st.sidebar.number_input("Batch Size per Run", 10, 200, 20)

if "tiktok_df" not in st.session_state:
    st.session_state.tiktok_df = pd.DataFrame()

# Scraper trigger
def launch_scraper():
    if not apify_key:
        st.warning("Please enter your Apify API token.")
        return
    if not queries:
        st.warning("Please enter at least one search keyword.")
        return

    all_results = []
    for query in queries:
        posts = run_in_thread(run_apify_scraper, apify_key, query, target_total, batch_size)
        all_results.extend(posts)
    df = pd.DataFrame(all_results)
    st.session_state.tiktok_df = df
    if df.empty:
        st.error("Scraper returned no data. Try different keywords.")
    else:
        st.success(f"Scraped {len(df)} TikTok posts.")
        st.dataframe(df)

# AI vetting trigger
def launch_ai_vetting():
    df = st.session_state.get("tiktok_df", pd.DataFrame())
    if df.empty:
        st.warning("No TikTok data available. Please run the scraper first.")
        return

    key = openai_key if provider == "OpenAI GPT" else gemini_key
    if not key:
        st.warning(f"Please enter your API key for {provider}.")
        return

    with st.spinner("Running AI vetting..."):
        vetting_results = run_in_thread(analyze_llm, df.to_dict("records"), model, provider.split()[0], key, temperature)
        if vetting_results.empty:
            st.error("AI vetting produced no output.")
        else:
            st.session_state.vetting_df = vetting_results
            st.success("AI vetting complete.")
            st.dataframe(vetting_results)
            st.download_button("Download Vetting Results CSV", vetting_results.to_csv(index=False), "vetting_results.csv")

# Interface Buttons
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Run TikTok Scraper"):
        launch_scraper()
with col2:
    if st.button("🤖 Run AI Vetting"):
        launch_ai_vetting()

