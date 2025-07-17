import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk

# --- NLP setup
try:
    from textblob import TextBlob
    nltk.download("punkt")
except ModuleNotFoundError:
    st.error("Install textblob and nltk in requirements.txt!")
    raise

st.set_page_config(page_title="TikTok/Threads Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok & Threads DOL/KOL Vetting Tool")

# --- API KEY ---
apify_api_key = st.sidebar.text_input("Apify API Token", type="password", help="Get this from https://console.apify.com/account#/integrations")

# --- PLATFORM MODE ---
mode = st.sidebar.radio("Scoring Strategy", [
    "By Doctor (DOL Vetting)",
    "By Brand (Brand Sentiment)"
], help="Choose by Doctor (KOL) or Brand (sentiment) context.")

query = st.sidebar.text_input(
    "TikTok Search Term",
    value="doctor" if "Doctor" in mode else "brandA",
    help="Keyword for TikTok search."
)
max_results = st.sidebar.slider("Max TikTok Posts", 5, 50, 10, help="# TikTok posts to analyze.")

# --- THREADS UX ---
st.sidebar.header("Meta Threads Scraper")
threads_enabled = st.sidebar.checkbox("Include Meta Threads Data", value=False)
threads_username = st.sidebar.text_input("Threads Username (optional)", value="", help="Public Threads username.")


with st.sidebar.expander("❓ FAQ & Help"):
    st.markdown("""
    - **How does scoring work?**  
      Uses sentiment + keyword analysis.
    - **Privacy:**  
      Your queries are not stored unless part of admin analytics.
    """)

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]  # Extend as needed

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
        rationale += f'. Transcript snippet: “{transcript[:90].strip()}...”' if transcript else ". No transcript available."
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        rationale += f' Transcript: “{transcript[:90].strip()}...”' if transcript else ". No transcript available."
    return rationale

# ------------------ APIFY TRANSCRIPT+ANALYSIS ACTOR INTEGRATION ------------------

def run_apify_tiktok_transcriber(api_key, query, max_results):
    """
    Calls Apify Actor emQXBCL3xePZYgJyn for TikTok captions/transcripts + text analysis.
    Returns: list of dict, each containing author, transcript, text, and stats.
    """
    ACTOR_ID = "emQXBCL3xePZYgJyn"
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs?token={api_key}"
    payload = {
        "searchQuery": query,
        "maxResults": max_results,
        "platform": "tiktok"
    }
    # Start the Actor run
    resp = requests.post(url, json=payload)
    if resp.status_code not in (200, 201):
        st.error("Failed to trigger Apify TikTok Transcriber actor.")
        return []
    run_data = resp.json()
    run_id = run_data["data"]["id"]
    # Poll for finish
    for _ in range(90):
        r = requests.get(
            f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs/{run_id}?token={api_key}")
        data = r.json()["data"]
        if data["status"] == "SUCCEEDED":
            dataset_id = data.get("defaultDatasetId")
            if dataset_id:
                break
        elif data["status"] == "FAILED":
            st.error("Apify TikTok Transcriber actor failed.")
            return []
        time.sleep(4)
    else:
        st.error("Apify Transcriber run timed out.")
        return []
    # Get dataset
    items_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={api_key}"
    items = requests.get(items_url).json()
    return items

def process_tiktok_apify_results(posts):
    results = []
    for post in posts:
        try:
            author = post.get("author", "")
            text = post.get("caption", "")
            transcript = post.get("transcript", "")
            post_id = post.get("id", "") or post.get("postId", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}" if author and post_id else ""
            ts = datetime.fromtimestamp(post.get("timestamp", time.time()))
            body = f"{text} {transcript}"
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, transcript, author, dol_score, sentiment, mode)
            results.append({
                "Author": author,
                "Text": text,
                "Transcript": transcript or "No transcript available.",
                "Likes": post.get("likes", 0),
                "Views": post.get("views", 0),
                "Comments": post.get("comments", 0),
                "Shares": post.get("shares", 0),
                "Timestamp": ts,
                "Post URL": url,
                "DOL Score": dol_score,
                "Sentiment Score": sentiment_score,
                "KOL/DOL Status": f"{'🌟' if kol_dol_label == 'KOL' else '👍' if kol_dol_label == 'DOL' else '❌'} {kol_dol_label}",
                "Brand Sentiment Label": sentiment,
                "LLM DOL Score Rationale": rationale
            })
        except Exception as e:
            st.warning(f"Skipped 1 post due to error: {e}")
            continue
    return pd.DataFrame(results)

# ---- THREADS SCRAPER (unchanged) ----
def run_threads_scraper(apify_token, username, max_results=10):
    if not username:
        return []
    url = "https://api.apify.com/v2/acts/curious_coder~threads-scraper/runs"
    input_data = {"usernames": [username], "resultsPerPage": max_results}
    headers = {"Authorization": f"Bearer {apify_token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=input_data)
    if resp.status_code not in (200, 201):
        st.warning("Failed to initiate Threads scraping. Check API token/quota.")
        return []
    run_id = resp.json()["data"]["id"]
    for _ in range(60):
        run_status_url = f"https://api.apify.com/v2/acts/curious_coder~threads-scraper/runs/{run_id}"
        r = requests.get(run_status_url, headers=headers)
        data = r.json().get("data", {})
        status = data.get("status")
        dataset_id = data.get("defaultDatasetId") or data.get("output", {}).get("defaultDatasetId")
        if status == "SUCCEEDED" and dataset_id:
            break
        elif status == "FAILED":
            st.warning("Threads scraping job failed on Apify.")
            return []
        time.sleep(5)
    if not dataset_id:
        st.warning("Could not find dataset for scraped Threads posts.")
        return []
    data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    return requests.get(data_url, headers=headers).json()

def process_threads_data(raw_posts):
    rows = []
    for item in raw_posts:
        try:
            user = item.get("user", {})
            caption = item.get("caption", {}).get("text", "")
            dt = datetime.fromtimestamp(item.get("taken_at", 0))
            rows.append({
                "Threads Username": user.get("username"),
                "Caption": caption,
                "Likes": item.get("like_count", 0),
                "Replies": item.get("reply_count", 0),
                "Timestamp": dt,
                "URL": f"https://www.threads.net/@{user.get('username')}/post/{item.get('code', '')}"
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

if "top_kols" not in st.session_state:
    st.session_state["top_kols"] = []
if "analysis_count" not in st.session_state:
    st.session_state["analysis_count"] = 0
if "feedback_logs" not in st.session_state:
    st.session_state["feedback_logs"] = []

if apify_api_key:
    if st.button("🚀 Run Analysis"):
        st.session_state["analysis_count"] += 1

        # Run TikTok Apify Actor (with built-in transcript extraction)
        tiktok_data = run_apify_tiktok_transcriber(apify_api_key, query, max_results)
        if not tiktok_data:
            st.warning("No TikTok posts found.")
        else:
            st.success(f"✅ {len(tiktok_data)} TikTok posts scraped and transcribed via Apify.")
            df = process_tiktok_apify_results(tiktok_data)
            st.metric("TikTok Posts", len(df))
            if not df.empty:
                kols = df[df["KOL/DOL Status"].str.contains("KOL|DOL", regex=True)]["Author"].unique().tolist()
                st.session_state["top_kols"].extend(x for x in kols if x not in st.session_state["top_kols"])
            st.subheader("📋 TikTok Analysis Results")
            tiktok_cols = [
                "Author", "Text", "Transcript", "Likes", "Views", "Comments", "Shares",
                "DOL Score", "Sentiment Score", "Post URL", "KOL/DOL Status",
                "Brand Sentiment Label", "LLM DOL Score Rationale"
            ]
            st.dataframe(df[tiktok_cols], use_container_width=True)
            st.download_button("Download TikTok CSV", df.to_csv(index=False),
                              file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

        if threads_enabled and threads_username.strip():
            st.info("Scraping Threads data...")
            raw_threads = run_threads_scraper(apify_api_key, threads_username.strip(), max_results)
            if raw_threads:
                threads_df = process_threads_data(raw_threads)
                st.subheader("🧵 Meta Threads Results")
                st.dataframe(threads_df, use_container_width=True)
                st.download_button("Download Threads CSV", threads_df.to_csv(index=False),
                                  file_name="threads_posts.csv")
            else:
                st.info("No Threads data found or available.")

        if st.session_state["analysis_count"] >= 5:
            st.sidebar.success("🎉 Power User: 5+ Analyses!")
            st.balloons()

        if st.session_state["top_kols"]:
            st.sidebar.markdown("👩‍⚕️ **Top DOL/KOLs analyzed (this session):**")
            for author in st.session_state["top_kols"]:
                st.sidebar.write(f"- {author}")

        feedback = st.radio("Did you find this analysis useful?", ["👍 Yes", "👎 No"], horizontal=True)
        if feedback:
            st.session_state["feedback_logs"].append({
                "time": datetime.now().isoformat(),
                "feedback": feedback,
                "last_query": query,
            })
            st.info(f"Thanks for your feedback: {feedback}")

else:
    st.info("🔑 Please provide your Apify API key to begin.")

if st.sidebar.checkbox("Show Analytics (Admin only)"):
    st.subheader("🔎 Usage Analytics")
    st.json({
        "analyses_this_session": st.session_state["analysis_count"],
        "feedback_this_session": st.session_state["feedback_logs"],
        "unique_DOL/KOLs": st.session_state["top_kols"]
    })

# requirements.txt:
# streamlit
# requests
# pandas
# textblob
# nltk


