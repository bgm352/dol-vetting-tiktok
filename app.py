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
st.title("🩺 TikTok & Threads Vetting Tool (DOL/KOL, Brand, Transcript, Social)")

# --- SECRETS/API KEYS ---
SUPADATA_API_KEY = st.secrets.get("SUPADATA_API_KEY") or st.sidebar.text_input("Supadata Transcript API Key", type="password")
apify_api_key = st.sidebar.text_input("Apify API Token", type="password")

# --- PLATFORM MODE ---
mode = st.sidebar.radio("Scoring Strategy", ["By Doctor (DOL Vetting)", "By Brand (Brand Sentiment)"])
query = st.sidebar.text_input("TikTok Search Term", value="doctor" if "Doctor" in mode else "brandA")
max_results = st.sidebar.slider("Max TikTok Posts", 5, 50, 10)

# --- THREADS UX ---
st.sidebar.header("Meta Threads Scraper")
threads_enabled = st.sidebar.checkbox("Include Meta Threads Data", value=False)
threads_username = st.sidebar.text_input("Threads Username (optional)", value="")

# --- KEYWORDS for rationale
ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

# --------------- HELPERS -------------------

def fetch_transcript(url, api_key):
    """Supadata AI transcript with 'mode=generate', handles 'not found' gracefully"""
    try:
        resp = requests.get("https://api.supadata.ai/v1/transcript",
                            params={"url": url, "mode": "generate"},
                            headers={"x-api-key": api_key}, timeout=30)
        data = resp.json() if resp.ok else {}
        t = data.get("transcript", "")
        if isinstance(t, list):
            return " ".join(x.get("text","") for x in t)
        return t or "Transcript not found"
    except Exception:
        return "Transcript not found"

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
            rationale += f". Transcript snippet: “{transcript[:90].strip()}...”"
        elif transcript and "not found" in transcript:
            rationale += ". No transcript available for in-depth spoken content review."
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if transcript and "not found" not in transcript:
            rationale += f" Transcript: “{transcript[:90].strip()}...”"
        elif transcript and "not found" in transcript:
            rationale += ". No transcript available for sentiment of spoken content."
    return rationale

@st.cache_data(show_spinner=False)
def run_apify_scraper(api_key, query, max_results):
    """Get TikTok posts from Apify scraper"""
    try:
        run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
        start = requests.post(run_url, headers={"Authorization": f"Bearer {api_key}"}, json={
            "searchQueries": [query], "resultsPerPage": max_results, "searchType": "keyword"
        }).json()
        run_id = start.get("data", {}).get("id")
        dataset_id = start.get("data", {}).get("defaultDatasetId")
        for _ in range(60):  # up to 5 min
            resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}).json()
            if resp["data"]["status"] == "SUCCEEDED":
                break
            time.sleep(5)
        data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        return requests.get(data_url).json()
    except Exception:
        st.error("💥 Failed to fetch TikTok data from Apify.")
        return []

def process_posts(posts, supa_key):
    results = []
    for i, post in enumerate(posts):
        try:
            author = post.get("authorMeta", {}).get("name", "")
            text = post.get("text", "")
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"
            tscript = fetch_transcript(url, supa_key)
            ts = datetime.fromtimestamp(post.get("createTime", time.time()))
            body = f"{text} {tscript}"
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, tscript, author, dol_score, sentiment, mode)
            results.append({
                "Author": author,
                "Text": text.strip(),
                "Transcript": tscript if tscript else "Transcript not found",
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
                "LLM DOL Score Rationale": rationale
            })
        except Exception as e:
            st.warning(f"⛔ Skipped 1 post due to error: {e}")
            continue
    return pd.DataFrame(results)

# --- Meta Threads integration ---
def run_threads_scraper(apify_token, username, max_results=10):
    """
    Run Apify Threads Scraper for a Threads username, returns post list.
    """
    if not username:
        return []
    url = "https://api.apify.com/v2/acts/curious_coder~threads-scraper/runs"
    input_data = {"usernames": [username], "resultsPerPage": max_results}
    headers = {"Authorization": f"Bearer {apify_token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json=input_data)
        if resp.status_code != 201:
            return []
        run_id = resp.json()["data"]["id"]
        status_url = f"https://api.apify.com/v2/acts/curious_coder~threads-scraper/runs/{run_id}"
        for _ in range(60):
            r = requests.get(status_url, headers=headers)
            stat = r.json()["data"]["status"]
            if stat == "SUCCEEDED":
                break
            elif stat == "FAILED":
                return []
            time.sleep(5)
        dataset_id = resp.json()["data"]["defaultDatasetId"]
        data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        return requests.get(data_url, headers=headers).json()
    except Exception as e:
        st.warning(f"Threads scraper failed: {e}")
        return []

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

# ============= MAIN APP FLOW =============
if apify_api_key and SUPADATA_API_KEY:

    if st.button("🚀 Run Analysis"):
        # TikTok
        tiktok_data = run_apify_scraper(apify_api_key, query, max_results)
        if not tiktok_data:
            st.warning("No TikTok posts found.")
        else:
            st.success(f"✅ {len(tiktok_data)} TikTok posts scraped.")
            df = process_posts(tiktok_data, SUPADATA_API_KEY)
            st.metric("TikTok Posts", len(df))
            st.subheader("📋 TikTok Analysis Results")
            tiktok_cols = ["Author", "Text", "Transcript", "Likes", "Views", "Comments", "Shares", "DOL Score", "Sentiment Score", "Post URL", "KOL/DOL Status", "Brand Sentiment Label", "LLM DOL Score Rationale"]
            st.dataframe(df[tiktok_cols], use_container_width=True)
            st.download_button("Download TikTok CSV", df.to_csv(index=False), file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
       
        # Threads
        if threads_enabled and threads_username.strip():
            st.info("Scraping Threads data...")
            raw_threads = run_threads_scraper(apify_api_key, threads_username.strip(), max_results)
            if raw_threads:
                threads_df = process_threads_data(raw_threads)
                st.subheader("🧵 Meta Threads Results")
                st.dataframe(threads_df, use_container_width=True)
                st.download_button("Download Threads CSV", threads_df.to_csv(index=False), file_name="threads_posts.csv")
            else:
                st.info("No Threads data found or available.")

else:
    st.info("🔑 Please provide both Apify and Supadata API keys to begin.")

# ------- 📈 Retention/UX/gamification (future roadmap hooks) ----------- 
# TODO: Personalize dashboard with "your top DOL/KOLs analyzed"
# TODO: Add "Loved this feature? 👍👎" after each analysis for feedback loop
# TODO: Track user sessions and badge repeat users for gamification
# TODO: Provide self-service FAQ, onboarding video and contextual tooltips per table field
# TODO: Modularize scoring/feature expansion for custom field plug-ins (future-proofing)
# TODO: Add analytic dashboard for admin to review feature usage and retention
