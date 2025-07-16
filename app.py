import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from textblob import TextBlob

# ------------------
# Page Setup
# ------------------
st.set_page_config(page_title="TikTok Vetting Tool", page_icon="🩺", layout="wide")
st.title("🩺 TikTok Vetting Tool - DOL/KOL + Brand Sentiment")
st.caption("Analyze TikTok content by HCPs or sentiment around brands.")

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
max_results = st.sidebar.slider("Max TikTok Posts", min_value=5, max_value=50, value=20)

# ------------------
# Helper Functions
# ------------------

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

@st.cache_data(show_spinner=False)
def run_scraper(token, query, max_results):
    """Run Apify TikTok scraper"""
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

def process_data(raw_data):
    """Standardize and score posts"""
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

            # Sentiment
            polarity = TextBlob(text).sentiment.polarity
            sentiment = classify_sentiment(polarity)

            # Assign dummy DOL score (placeholder)
            dol_score = max(min(round((polarity * 10) + 5), 10), 1)
            kol_dol = classify_kol_dol(dol_score)

            results.append({
                "Author": author,
                "Verified": verified,
                "Text": text,
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
            })
        except:
            continue
    return pd.DataFrame(results)

# ------------------
# Main Flow
# ------------------
if not apify_token:
    st.info("Enter Apify API token to begin.")
else:
    if st.button("🚀 Run Analysis"):
        data = run_scraper(apify_token, query, max_results)
        if not data:
            st.warning("No data found for the query.")
        else:
            df = process_data(data)
            st.success(f"Scraped {len(df)} posts.")

            # --- Filters ---
            st.subheader("🧼 Filter by Freshness")
            freshness_filter = st.selectbox("Filter by:", ["All", "Today", "This week", "This month", "Older"])
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

            if df_filtered.empty:
                st.warning("No posts match the filter.")
                df_filtered = df.copy()

            # --- Output Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Posts", len(df_filtered))
            col2.metric("Avg Likes", f"{df_filtered['Likes'].mean():.0f}")
            col3.metric("Total Views", f"{df_filtered['Views'].sum():,}")
            col4.metric("Verified Authors", df_filtered['Verified'].sum())

            # --- Mode Specific Display ---
            if mode == "By Brand (Brand Sentiment)":
                st.subheader("💬 Brand Sentiment")
                st.bar_chart(df_filtered['Brand Sentiment Label'].value_counts())
                sent_avg = df_filtered['Sentiment Score'].mean()
                st.metric("Avg Sentiment Score", f"{sent_avg:.2f}")

            if mode == "By Doctor (DOL Vetting)":
                st.subheader("🌟 DOL / KOL Classification")
                st.bar_chart(df_filtered['KOL/DOL Label'].value_counts())
                score_avg = df_filtered['DOL Score'].mean()
                st.metric("Avg DOL Score", f"{score_avg:.2f}")

            # --- Display Table ---
            st.subheader("📋 Results")

            display_cols = [
                "Author", "Verified", "Text", "Freshness", "Likes", "Views", "Comments", "Shares", "Timestamp", "URL"
            ]

            if mode == "By Doctor (DOL Vetting)":
                display_cols += ["KOL/DOL Status Display", "DOL Score"]
            else:
                display_cols += ["Brand Sentiment Display", "Sentiment Score"]

            st.dataframe(df_filtered[display_cols], use_container_width=True)

            # --- Export ---
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f'{mode.replace(" ", "_").lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime="text/csv"
            )
