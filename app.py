import streamlit as st
import requests
import pandas as pd
import json
import time
import threading
from typing import List, Dict

# Dependency validation ----------------------------------------------------
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

# Page setup ---------------------------------------------------------------
st.set_page_config(
    page_title="TikTok DOL/KOL Vetting Tool",
    layout="wide",
    page_icon="🩺"
)
st.title("🩺 TikTok DOL/KOL Vetting Tool — Batch Scraping, AI Vetting, and Export")

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

@st.cache_data(show_spinner=True, persist="disk")
def run_apify_scraper_batched(api_key: str, query: str, target_total: int, batch_size: int) -> List[Dict]:
    """
    Fetch TikTok posts via Apify TikTok Scraper Actor with robust retry and timeout logic.
    """
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    MAX_WAIT_SECONDS = 300
    RETRY_LIMIT = 3
    backoff = 5
    all_results = []
    offset = 0

    progress = st.progress(0.0, text=f"Initializing scraping for query: {query}")

    try:
        while len(all_results) < target_total:
            for attempt in range(1, RETRY_LIMIT + 1):
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
                        raise requests.HTTPError(f"{response.status_code}: {response.text}")

                    run_id = response.json().get("data", {}).get("id")
                    if not run_id:
                        raise ValueError("Apify did not return a valid run ID.")
                    time.sleep(3)

                    # Poll for status
                    start_time = time.time()
                    dataset_id = None
                    while time.time() - start_time < MAX_WAIT_SECONDS:
                        status_resp = requests.get(
                            f"{run_url}/{run_id}",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=30
                        )
                        data = status_resp.json().get("data", {})
                        if data.get("status") == "SUCCEEDED":
                            dataset_id = data.get("defaultDatasetId")
                            break
                        time.sleep(5)

                    if not dataset_id:
                        raise TimeoutError("Actor run did not finish within allowed time.")

                    # Retrieve dataset
                    dataset_resp = requests.get(
                        f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=True",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=60,
                    )
                    batch_items = dataset_resp.json()
                    if not isinstance(batch_items, list):
                        raise ValueError("Dataset response was not a list.")

                    all_results.extend(batch_items)
                    offset += batch_size
                    progress.progress(min(1.0, len(all_results) / float(target_total)))
                    break
                except Exception as e:
                    if attempt == RETRY_LIMIT:
                        st.error(f"Failed batch for '{query}': {e}")
                    else:
                        time.sleep(backoff * attempt)
            if len(all_results) >= target_total:
                break
        progress.progress(1.0)
    except Exception as e:
        st.error(f"Scraper critical error: {e}")

    return all_results[:target_total]


def analyze_and_score(tiktok_data: List[Dict], model="gpt-4", api_provider="OpenAI", temperature=0.3) -> pd.DataFrame:
    """
    Send TikTok post data to OpenAI or Gemini to assess influence and identify KOLs/DOLs.
    """
    prompt = (
        "Analyze the given TikTok profiles and output a JSON array with fields: "
        "'profile', 'influence_score' (0-1), and 'key_opinion_leader' (true/false). "
        "Be concise, structured, and strictly JSON:\n\n"
        f"{json.dumps(tiktok_data[:5], indent=2)}"
    )

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
            # TODO: Implement Gemini API (placeholder)
            result_text = "[]"

        scores_json = json.loads(result_text)
        df = pd.DataFrame(scores_json)
        if not {"profile", "influence_score", "key_opinion_leader"}.issubset(df.columns):
            raise ValueError("Missing fields in model output.")
        return df
    except json.JSONDecodeError:
        st.error("Model returned invalid JSON. Check prompt or retry.")
    except Exception as e:
        st.error(f"AI vetting error: {str(e)}")
    return pd.DataFrame(columns=["profile", "influence_score", "key_opinion_leader"])


def run_thread(func, *args):
    """Execute a blocking job in a background thread safely."""
    result = {}

    def target():
        try:
            result["data"] = func(*args)
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=target)
    t.start()
    t.join()

    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("data")


# -------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------
st.sidebar.header("Configuration")
apify_api_key = st.sidebar.text_input("Apify API Key", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])

if llm_provider == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
    model = st.sidebar.selectbox("Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
else:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gemini-2.5-pro", "gemini-2.5-flash"])

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4)
st.sidebar.markdown("---")

st.sidebar.header("Scrape Controls")
queries_text = st.sidebar.text_area("TikTok Search Queries (one per line)", help="Example: oncology New York")
queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
target_total = st.sidebar.number_input("Posts per query", min_value=10, value=100, step=10)
batch_size = st.sidebar.number_input("Batch size", min_value=10, max_value=200, value=20)

if "tiktok_data" not in st.session_state:
    st.session_state["tiktok_data"] = pd.DataFrame()

# -------------------------------------------------------------------------
# Scraping and Vetting Workflow
# -------------------------------------------------------------------------
def start_scraping():
    if not apify_api_key:
        st.warning("Enter your Apify API key.")
        return
    if not queries:
        st.warning("Enter at least one TikTok search query.")
        return

    all_results = []
    for query in queries:
        with st.spinner(f"Scraping TikTok posts for '{query}'..."):
            data = run_thread(run_apify_scraper_batched, apify_api_key, query, target_total, batch_size)
            all_results.extend(data or [])
    df = pd.DataFrame(all_results)
    st.session_state["tiktok_data"] = df
    if not df.empty:
        st.success(f"Retrieved {len(df)} posts across {len(queries)} queries.")
        st.dataframe(df)
    else:
        st.error("No results. Please check your Apify key or query settings.")


def start_vetting():
    df = st.session_state.get("tiktok_data", pd.DataFrame())
    if df.empty:
        st.warning("Please run the scraper first.")
        return

    provider = "OpenAI" if llm_provider == "OpenAI GPT" else "Gemini"
    key_valid = openai_api_key if provider == "OpenAI" else gemini_api_key
    if not key_valid:
        st.warning(f"Please provide a valid API key for {provider}.")
        return

    with st.spinner("Analyzing TikTok profiles..."):
        try:
            result_df = run_thread(analyze_and_score, df.to_dict("records"), model, provider, temperature)
            if not result_df.empty:
                st.session_state["vetting_results"] = result_df
                st.success("AI vetting complete!")
                st.dataframe(result_df)
                st.download_button("Download Results CSV", result_df.to_csv(index=False), "vetting_results.csv")
            else:
                st.error("AI did not return valid scores.")
        except Exception as e:
            st.error(f"Vetting failed: {e}")


# -------------------------------------------------------------------------
# Buttons
# -------------------------------------------------------------------------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Run TikTok Scraper"):
        start_scraping()
with col2:
    if st.button("🤖 Run AI Vetting"):
        start_vetting()




