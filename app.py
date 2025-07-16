import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# Install TextBlob NLP and corpora
try:
    from textblob import TextBlob
    import nltk
    nltk.download("punkt")
except ModuleNotFoundError:
    st.error("Required packages not installed. Please ensure textblob and nltk are listed in requirements.txt.")
    raise

# ---- Config ----
SUPADATA_API_KEY = st.secrets["SUPADATA_API_KEY"] if "SUPADATA_API_KEY" in st.secrets else st.sidebar.text_input("Supadata Transcript API Key", type="password")

# ------------------
# Page Setup
# ------------------
st.set_page_config(page_title="TikTok Vetting Tool", page_icon="🩺", layout="wide")
st.title("🩺 TikTok Vetting Tool - DOL/KOL + Brand Sentiment (with Transcript Analysis)")
st.caption("Analyze TikTok content by HCPs or sentiment around brands using both captions and full video transcripts.")

# ------------------
# Sidebar Config
# ------------------
st.sidebar.header("🔧 Configuration")
apify_token = st.sidebar.text_input("Apify API Token", type="password")

mode = st.sidebar.radio("Scoring Mode", ["By Doctor (DOL Vetting)", "By Brand (Brand Sentiment)"])

# ------------------
# Search Controls
# ------------------
st.sidebar.header("🔍 Search")
default_query = "doctor" if mode == "By Doctor (DOL Vetting)" else "BrandA"
query = st.sidebar.text_input("Search Term", value=default_query)
max_results = st.sidebar.slider("Max TikTok Posts", min_value=5, max_value=30, value=10)

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "antibody", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastrosophageal", "GEA", "GI", "adenocarcinoma", "BTC"]
RESEARCH_TERMS = ["biomarker", "conference", "abstract", "network", "clinical trial", "peer"]
BRAND_KEYWORDS = ["ziihera", "zanidatamab", "herceptin", "rituximab", "brandA"]  # Extend as needed

def classify_kol_dol(score):
    if pd.isna(score):
        return "Unknown"
    elif score >= 8:
        return "KOL"
    elif score >= 5:
        return "DOL"
    else:
        return "Not Suitable"

def classify_sentiment(score):
    if pd.isna(score):
        return "Unknown"
    elif score > 0.15:
        return "Positive"
    elif score < -0.15:
        return "Negative"
    else:
        return "Neutral"

def fetch_transcript_supadata(tiktok_url, api_key):
    """Fetch transcript for a TikTok video via Supadata API."""
    if not api_key or not tiktok_url:
        return ""
    try:
        resp = requests.get(
            "https://api.supadata.ai/v1/transcript",
            params={"url": tiktok_url},
            headers={"x-api-key": api_key}
        )
        if resp.status_code == 200:
            data = resp.json()
            # The API returns a "transcript" key which may be a string or a dict
            transcript = data.get("transcript", "")
            # Some APIs return only the text, others a dict of {start, end, text}
            if isinstance(transcript, str):
                return transcript
            elif isinstance(transcript, list):
                return " ".join(item["text"] for item in transcript if "text" in item)
            else:
                return str(transcript)
        else:
            return ""
    except Exception as e:
        return ""

def generate_llm_dol_rationale(text, transcript, author, score, sentiment):
    """Generate pharma/rwd-style LLM DOL rationale using both caption and transcript."""
    body = text + " " + transcript
    text_low = body.lower()
    tags = {
        "onco": any(w in text_low for w in ONCOLOGY_TERMS),
        "gi": any(w in text_low for w in GI_TERMS),
        "biomarker": any(w in text_low for w in RESEARCH_TERMS),
        "brand": any(w in text_low for w in BRAND_KEYWORDS),
    }

    strengths = []
    if tags["onco"]:
        strengths.append("is highly influential and networked in oncology, frequently discussing topics like monoclonal antibody-based therapies")
    if tags["gi"]:
        strengths.append("shows deep engagement with GI oncology topics, such as biliary tract cancer and gastroesophageal diseases")
    if tags["biomarker"]:
        strengths.append("demonstrates expertise in biomarkers or clinical research, supporting scientific credibility and peer engagement")
    if tags["brand"]:
        strengths.append("references specific therapies or drug names relevant to current campaigns")
    # Compose rationale.
    if strengths:
        rationale = (
            f"{author if author else 'This HCP'} {'; '.join(strengths)}."
            " Their active posting and professional engagement position them as a highly relevant DOL/KOL candidate."
        )
        if score < 8:
            rationale += " However, there is limited mention of highly targeted campaign terms, which may reduce suitability for some specialized efforts."
    else:
        if score >= 8:
            rationale = (
                f"{author if author else 'This creator'} demonstrates scientific authority and active engagement, aligning well with DOL criteria for HCP-focused campaigns."
            )
        elif score >= 5:
            rationale = (
                f"{author if author else 'This creator'} shows moderate relevance and engagement for DOL vetting, though lacks frequent reference to highly specific terms."
            )
        else:
            rationale = (
                f"{author if author else 'This creator'}'s posts show limited direct relevance or engagement with key medical campaign topics, reducing suitability for DOL roles."
            )
    if transcript and transcript.strip():
        rationale += f" Transcript review: \"{transcript[:90].strip()}...\""
    return rationale

def generate_brand_sentiment_rationale(text, transcript, sentiment):
    merged = text + " " + transcript
    if sentiment == "Positive":
        return f"Post and transcript express a favorable opinion toward the brand or therapy: \"{merged[:90]}...\""
    elif sentiment == "Negative":
        return f"Critical or negative feedback detected in text/transcript: \"{merged[:90]}...\""
    else:
        return f"No clear sentiment found in caption or transcript: \"{merged[:90]}...\""

@st.cache_data(show_spinner=False)
def run_scraper(token, query, max_results):
    """Run Apify TikTok scraper."""
    try:
        start_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
        input_data = {
            "searchQueries": [query],
            "resultsPerPage": max_results,
            "searchType": "keyword"
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        start_resp = requests.post(start_url, json=input_data, headers=headers)
        if start_resp.status_code != 201:
            st.error(f"Start Error: {start_resp.status_code}")
            return []
        run_id = start_resp.json()["data"]["id"]
        status_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs/{run_id}"
        with st.spinner("⏳ Scraping..."):
            while True:
                status = requests.get(status_url, headers=headers).json()["data"]["status"]
                if status == "SUCCEEDED":
                    break
                elif status == "FAILED":
                    st.error("Scraper failed.")
                    return []
                time.sleep(5)
        dataset_id = start_resp.json()["data"]["defaultDatasetId"]
        data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        return requests.get(data_url, headers=headers).json()
    except Exception as e:
        st.error(str(e))
        return []


def process_data(raw_data, supadata_api_key):
    """Standardize, score, and analyze caption+transcript for each post."""
    results = []
    for post in raw_data:
        try:
            author = post.get('authorMeta', {}).get('name', 'Unknown')
            verified = post.get('authorMeta', {}).get('verified', False)
            text = post.get('text', '')
            likes = post.get('diggCount', 0)
            shares = post.get('shareCount', 0)
            comments = post.get('commentCount', 0)
            views = post.get('playCount', 0)
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"

            # Transcript retrieval:
            transcript = fetch_transcript_supadata(url, supadata_api_key)
            # Timestamp
            ts = datetime.fromtimestamp(post.get("createTime", 0))
            days_ago = (datetime.now() - ts).days
            if days_ago == 0:
                freshness = "Today"
            elif days_ago == 1:
                freshness = "1 day ago"
            elif days_ago < 7:
                freshness = f"{days_ago} days ago"
            elif days_ago < 30:
                freshness = f"{days_ago // 7} weeks ago"
            elif days_ago < 365:
                freshness = f"{days_ago // 30} months ago"
            else:
                freshness = f"{days_ago // 365} years ago"

            # Sentiment (caption+transcript)
            all_text = (text or "") + " " + (transcript or "")
            blob = TextBlob(all_text)
            polarity = blob.sentiment.polarity
            sentiment = classify_sentiment(polarity)

            dol_score = max(min(round((polarity * 10) + 5), 10), 1)
            kol_dol = classify_kol_dol(dol_score)

            # Generate rationale
            if mode == "By Doctor (DOL Vetting)":
                llm_rationale = generate_llm_dol_rationale(text, transcript, author, dol_score, sentiment)
            else:
                llm_rationale = generate_brand_sentiment_rationale(text, transcript, sentiment)

            results.append({
                "Author": author,
                "Verified": verified,
                "Text": text,
                "Transcript": transcript,
                "Likes": likes,
                "Comments": comments,
                "Shares": shares,
                "Views": views,
                "Timestamp": ts,
                "Freshness": freshness,
                "URL": url,
                "DOL Score": dol_score,
                "KOL/DOL Label": kol_dol,
                "KOL/DOL Status Display": f"{'🌟' if kol_dol == 'KOL' else '👍' if kol_dol == 'DOL' else '❌'} {kol_dol}",
                "Sentiment Score": round(polarity, 3),
                "Brand Sentiment Label": sentiment,
                "Brand Sentiment Display": f"{'😊' if sentiment == 'Positive' else '😞' if sentiment == 'Negative' else '😐'} {sentiment}",
                "LLM DOL Score Rationale": llm_rationale
            })
        except Exception as e:
            continue
    return pd.DataFrame(results)


# ------------------
# Main Flow
# ------------------
if not apify_token or not SUPADATA_API_KEY:
    st.info("Enter your Apify and Supadata API keys to start.")
else:
    if st.button("🚀 Run Analysis"):
        data = run_scraper(apify_token, query, max_results)
        if not data:
            st.warning("No data found.")
        else:
            df = process_data(data, SUPADATA_API_KEY)
            st.success(f"Scraped {len(df)} posts.")

            st.subheader("📅 Filter by Freshness")
            freshness_filter = st.selectbox("Select:", ["All", "Today", "This week", "This month", "Older"])
            df_filtered = df.copy()
            if freshness_filter != "All":
                if freshness_filter == "Today":
                    df_filtered = df[df["Freshness"] == "Today"]
                elif freshness_filter == "This week":
                    df_filtered = df[df["Freshness"].str.contains("day|Today", na=False)]
                elif freshness_filter == "This month":
                    df_filtered = df[df["Freshness"].str.contains("day|week|Today", na=False)]
                elif freshness_filter == "Older":
                    df_filtered = df[df["Freshness"].str.contains("month|year", na=False)]

            st.metric("Filtered Posts", len(df_filtered))

            if mode == "By Brand (Brand Sentiment)":
                st.subheader("💬 Brand Sentiment Summary")
                st.bar_chart(df_filtered['Brand Sentiment Label'].value_counts())
                st.metric("Avg Sentiment Score", f"{df_filtered['Sentiment Score'].mean():.2f}")
            else:
                st.subheader("🌟 KOL / DOL Summary")
                st.bar_chart(df_filtered['KOL/DOL Label'].value_counts())
                st.metric("Avg DOL Score", f"{df_filtered['DOL Score'].mean():.2f}")

            st.subheader("📋 TikTok Posts")

            cols = [
                "Author", "Verified", "Text", "Transcript", "Freshness", "Likes", "Views", 
                "Comments", "Shares", "Timestamp", "URL"
            ]
            if mode == "By Doctor (DOL Vetting)":
                cols += ["KOL/DOL Status Display", "DOL Score", "LLM DOL Score Rationale"]
            else:
                cols += ["Brand Sentiment Display", "Sentiment Score", "LLM DOL Score Rationale"]

            st.dataframe(df_filtered[cols], use_container_width=True)

            st.download_button(
                "📥 Download CSV",
                df_filtered.to_csv(index=False),
                file_name=f"{mode.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
