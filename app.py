import streamlit as st
import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import json
from typing import List, Dict, Optional, Tuple

# ------------------
# Logging Setup
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------
# Page Setup
# ------------------
st.set_page_config(page_title="TikTok Vetting Tool", page_icon="🩺", layout="wide")
st.title("🩺 TikTok Vetting Tool - DOL/KOL + Brand Sentiment")
st.caption("Analyze TikTok content by HCPs or sentiment around brands.")

# ------------------
# Debug Mode Toggle
# ------------------
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)

if debug_mode:
    st.sidebar.info("Debug mode enabled - detailed logs will be shown")

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

# Rate limiting controls
st.sidebar.header("⚙️ Advanced Settings")
polling_interval = st.sidebar.slider("Polling Interval (seconds)", min_value=3, max_value=15, value=5)
max_retries = st.sidebar.slider("Max Retries", min_value=1, max_value=5, value=3)

# ------------------
# Helper Functions
# ------------------

def log_debug(message: str) -> None:
    """Log debug messages if debug mode is enabled"""
    if debug_mode:
        logger.info(f"DEBUG: {message}")
        st.sidebar.text(f"🐛 {message}")

def classify_kol_dol(score: float) -> str:
    """Classify KOL/DOL based on score with input validation"""
    try:
        if pd.isna(score) or score is None:
            return "Unknown"
        
        score = float(score)
        if score >= 8:
            return "KOL"
        elif score >= 5:
            return "DOL"
        else:
            return "Not Suitable"
    except (ValueError, TypeError) as e:
        log_debug(f"Error classifying KOL/DOL score {score}: {e}")
        return "Unknown"

def classify_sentiment(score: float) -> str:
    """Classify sentiment based on score with input validation"""
    try:
        if pd.isna(score) or score is None:
            return "Unknown"
        
        score = float(score)
        if score > 0.15:
            return "Positive"
        elif score < -0.15:
            return "Negative"
        else:
            return "Neutral"
    except (ValueError, TypeError) as e:
        log_debug(f"Error classifying sentiment score {score}: {e}")
        return "Unknown"

def validate_api_token(token: str) -> bool:
    """Validate Apify API token format"""
    if not token:
        return False
    
    # Basic token validation - Apify tokens typically start with 'apify_api_'
    if not token.startswith('apify_api_'):
        st.warning("⚠️ API token should start with 'apify_api_'")
        return False
    
    return True

def safe_get_nested(data: Dict, keys: List[str], default=None):
    """Safely get nested dictionary values"""
    try:
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data
    except (KeyError, TypeError):
        return default

@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def run_scraper(token: str, query: str, max_results: int) -> List[Dict]:
    """Run Apify TikTok scraper with improved error handling"""
    
    if not validate_api_token(token):
        st.error("❌ Invalid API token format")
        return []
    
    log_debug(f"Starting scraper with query: '{query}', max_results: {max_results}")
    
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
        
        log_debug(f"Sending request to: {start_url}")
        
        # Start the scraper
        start_resp = requests.post(start_url, json=input_data, headers=headers, timeout=30)
        
        if start_resp.status_code != 201:
            error_msg = f"Start Error: {start_resp.status_code} - {start_resp.text}"
            log_debug(error_msg)
            st.error(error_msg)
            return []
        
        run_data = start_resp.json()
        run_id = run_data["data"]["id"]
        dataset_id = run_data["data"]["defaultDatasetId"]
        
        log_debug(f"Run ID: {run_id}, Dataset ID: {dataset_id}")
        
        # Poll for completion
        status_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs/{run_id}"
        
        retry_count = 0
        max_poll_time = 300  # 5 minutes max
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while retry_count < max_retries and (time.time() - start_time) < max_poll_time:
            try:
                status_resp = requests.get(status_url, headers=headers, timeout=30)
                
                if status_resp.status_code != 200:
                    log_debug(f"Status check failed: {status_resp.status_code}")
                    retry_count += 1
                    time.sleep(polling_interval)
                    continue
                
                status_data = status_resp.json()
                status = status_data["data"]["status"]
                
                elapsed_time = time.time() - start_time
                progress = min(elapsed_time / max_poll_time, 0.95)  # Cap at 95% until completion
                progress_bar.progress(progress)
                status_text.text(f"⏳ Status: {status} (Elapsed: {elapsed_time:.1f}s)")
                
                log_debug(f"Current status: {status}")
                
                if status == "SUCCEEDED":
                    progress_bar.progress(1.0)
                    status_text.text("✅ Scraping completed successfully!")
                    break
                elif status == "FAILED":
                    error_msg = "❌ Scraper failed"
                    log_debug(error_msg)
                    st.error(error_msg)
                    return []
                elif status in ["ABORTED", "TIMED-OUT"]:
                    error_msg = f"❌ Scraper {status.lower()}"
                    log_debug(error_msg)
                    st.error(error_msg)
                    return []
                
                time.sleep(polling_interval)
                
            except requests.exceptions.Timeout:
                log_debug("Request timeout during status check")
                retry_count += 1
                time.sleep(polling_interval)
            except requests.exceptions.RequestException as e:
                log_debug(f"Request error during status check: {e}")
                retry_count += 1
                time.sleep(polling_interval)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if retry_count >= max_retries:
            st.error("❌ Max retries exceeded")
            return []
        
        # Fetch the data
        data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        log_debug(f"Fetching data from: {data_url}")
        
        data_resp = requests.get(data_url, headers=headers, timeout=60)
        
        if data_resp.status_code != 200:
            error_msg = f"Data fetch error: {data_resp.status_code} - {data_resp.text}"
            log_debug(error_msg)
            st.error(error_msg)
            return []
        
        data = data_resp.json()
        log_debug(f"Successfully fetched {len(data)} posts")
        
        return data
        
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. Please try again.")
        return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet connection.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request error: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON decode error: {str(e)}")
        return []
    except Exception as e:
        log_debug(f"Unexpected error: {str(e)}")
        st.error(f"❌ Unexpected error: {str(e)}")
        return []

def calculate_engagement_rate(likes: int, comments: int, shares: int, views: int) -> float:
    """Calculate engagement rate"""
    try:
        if views == 0:
            return 0.0
        total_engagement = likes + comments + shares
        return (total_engagement / views) * 100
    except (ZeroDivisionError, TypeError):
        return 0.0

def format_number(num: int) -> str:
    """Format large numbers for display"""
    try:
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(num)
    except (TypeError, ValueError):
        return "0"

def get_freshness_category(days_ago: int) -> str:
    """Get freshness category with better logic"""
    try:
        if days_ago == 0:
            return "Today"
        elif days_ago == 1:
            return "Yesterday"
        elif days_ago <= 7:
            return "This week"
        elif days_ago <= 30:
            return "This month"
        elif days_ago <= 90:
            return "Last 3 months"
        else:
            return "Older"
    except (TypeError, ValueError):
        return "Unknown"

def process_data(raw_data: List[Dict]) -> pd.DataFrame:
    """Process and standardize scraped data with improved error handling"""
    
    log_debug(f"Processing {len(raw_data)} raw posts")
    
    results = []
    errors = []
    
    for i, post in enumerate(raw_data):
        try:
            # Safely extract author information
            author_meta = post.get('authorMeta', {})
            author = safe_get_nested(author_meta, ['name'], 'Unknown')
            verified = safe_get_nested(author_meta, ['verified'], False)
            
            # Extract post content
            text = post.get('text', '')
            
            # Extract engagement metrics with defaults
            likes = int(post.get('diggCount', 0) or 0)
            shares = int(post.get('shareCount', 0) or 0)
            comments = int(post.get('commentCount', 0) or 0)
            views = int(post.get('playCount', 0) or 0)
            
            # Calculate engagement rate
            engagement_rate = calculate_engagement_rate(likes, comments, shares, views)
            
            # Extract post ID and create URL
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}" if post_id else "N/A"
            
            # Process timestamp
            create_time = post.get("createTime", 0)
            try:
                if isinstance(create_time, str):
                    create_time = int(create_time)
                ts = datetime.fromtimestamp(create_time)
            except (ValueError, TypeError, OSError):
                ts = datetime.now()
            
            days_ago = (datetime.now() - ts).days
            freshness_category = get_freshness_category(days_ago)
            
            # Detailed freshness display
            if days_ago == 0:
                freshness = "Today"
            elif days_ago == 1:
                freshness = "Yesterday"
            elif days_ago < 7:
                freshness = f"{days_ago} days ago"
            elif days_ago < 30:
                freshness = f"{days_ago // 7} weeks ago"
            elif days_ago < 365:
                freshness = f"{days_ago // 30} months ago"
            else:
                freshness = f"{days_ago // 365} years ago"
            
            # Sentiment analysis with error handling
            try:
                polarity = TextBlob(text).sentiment.polarity if text else 0.0
            except:
                polarity = 0.0
            
            sentiment = classify_sentiment(polarity)
            
            # DOL scoring (improved algorithm)
            # Consider multiple factors: sentiment, engagement, verification status
            base_score = (polarity + 1) * 5  # Convert -1 to 1 range to 0 to 10
            
            # Adjust based on verification
            if verified:
                base_score += 1
            
            # Adjust based on engagement rate
            if engagement_rate > 5:
                base_score += 1
            elif engagement_rate > 10:
                base_score += 2
            
            # Adjust based on follower indicators (if available)
            # This is a placeholder - you might want to adjust based on actual follower data
            
            dol_score = max(min(round(base_score), 10), 1)
            kol_dol = classify_kol_dol(dol_score)
            
            results.append({
                "Author": author,
                "Verified": verified,
                "Text": text[:200] + "..." if len(text) > 200 else text,  # Truncate long text
                "Full Text": text,
                "Likes": likes,
                "Likes Formatted": format_number(likes),
                "Comments": comments,
                "Comments Formatted": format_number(comments),
                "Shares": shares,
                "Shares Formatted": format_number(shares),
                "Views": views,
                "Views Formatted": format_number(views),
                "Engagement Rate": round(engagement_rate, 2),
                "Timestamp": ts,
                "Freshness": freshness,
                "Freshness Category": freshness_category,
                "Days Ago": days_ago,
                "URL": url,
                "DOL Score": dol_score,
                "KOL/DOL Label": kol_dol,
                "KOL/DOL Status Display": f"{'🌟' if kol_dol == 'KOL' else '👍' if kol_dol == 'DOL' else '❌'} {kol_dol}",
                "Sentiment Score": round(polarity, 3),
                "Brand Sentiment Label": sentiment,
                "Brand Sentiment Display": f"{'😊' if sentiment == 'Positive' else '😞' if sentiment == 'Negative' else '😐'} {sentiment}",
                "Post ID": post_id,
            })
            
        except Exception as e:
            error_msg = f"Error processing post {i}: {str(e)}"
            log_debug(error_msg)
            errors.append(error_msg)
            continue
    
    if errors and debug_mode:
        st.sidebar.error(f"⚠️ {len(errors)} processing errors occurred")
        with st.sidebar.expander("View Errors"):
            for error in errors:
                st.text(error)
    
    log_debug(f"Successfully processed {len(results)} posts")
    
    return pd.DataFrame(results)

def apply_filters(df: pd.DataFrame, freshness_filter: str, min_engagement: float = 0.0) -> pd.DataFrame:
    """Apply filters to dataframe"""
    df_filtered = df.copy()
    
    # Apply freshness filter
    if freshness_filter != "All":
        df_filtered = df_filtered[df_filtered["Freshness Category"] == freshness_filter]
    
    # Apply engagement filter
    if min_engagement > 0:
        df_filtered = df_filtered[df_filtered["Engagement Rate"] >= min_engagement]
    
    return df_filtered

def create_summary_metrics(df: pd.DataFrame) -> Dict:
    """Create summary metrics for the dataset"""
    try:
        return {
            "total_posts": len(df),
            "avg_likes": df['Likes'].mean(),
            "avg_engagement": df['Engagement Rate'].mean(),
            "total_views": df['Views'].sum(),
            "verified_count": df['Verified'].sum(),
            "kol_count": len(df[df['KOL/DOL Label'] == 'KOL']),
            "dol_count": len(df[df['KOL/DOL Label'] == 'DOL']),
            "positive_sentiment": len(df[df['Brand Sentiment Label'] == 'Positive']),
            "negative_sentiment": len(df[df['Brand Sentiment Label'] == 'Negative']),
            "neutral_sentiment": len(df[df['Brand Sentiment Label'] == 'Neutral']),
        }
    except Exception as e:
        log_debug(f"Error creating summary metrics: {e}")
        return {}

# ------------------
# Main Application Flow
# ------------------

def main():
    """Main application flow"""
    
    if not apify_token:
        st.info("👆 Please enter your Apify API token in the sidebar to begin.")
        st.markdown("""
        ### How to get an Apify API token:
        1. Go to [Apify Console](https://console.apify.com/)
        2. Sign up or log in
        3. Go to Settings → Integrations
        4. Copy your API token
        """)
        return
    
    if not query.strip():
        st.warning("⚠️ Please enter a search term.")
        return
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Initializing analysis..."):
            log_debug("Starting analysis")
            
            # Run scraper
            raw_data = run_scraper(apify_token, query, max_results)
            
            if not raw_data:
                st.warning("⚠️ No data found for the query. Please try different search terms.")
                return
            
            # Process data
            with st.spinner("Processing data..."):
                df = process_data(raw_data)
                
            if df.empty:
                st.warning("⚠️ No valid posts could be processed.")
                return
            
            st.success(f"✅ Successfully analyzed {len(df)} posts!")
            
            # --- Filters Section ---
            st.subheader("🔍 Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                freshness_options = ["All"] + sorted(df["Freshness Category"].unique().tolist())
                freshness_filter = st.selectbox("Filter by Freshness:", freshness_options)
            
            with col2:
                min_engagement = st.slider("Min Engagement Rate (%):", 
                                         min_value=0.0, 
                                         max_value=float(df["Engagement Rate"].max()), 
                                         value=0.0, 
                                         step=0.1)
            
            # Apply filters
            df_filtered = apply_filters(df, freshness_filter, min_engagement)
            
            if df_filtered.empty:
                st.warning("⚠️ No posts match the current filters. Showing all results.")
                df_filtered = df.copy()
            
            # --- Summary Metrics ---
            st.subheader("📊 Summary Metrics")
            
            metrics = create_summary_metrics(df_filtered)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Posts", metrics.get("total_posts", 0))
            with col2:
                st.metric("Avg Likes", f"{metrics.get('avg_likes', 0):.0f}")
            with col3:
                st.metric("Avg Engagement", f"{metrics.get('avg_engagement', 0):.1f}%")
            with col4:
                st.metric("Total Views", format_number(metrics.get("total_views", 0)))
            with col5:
                st.metric("Verified Authors", metrics.get("verified_count", 0))
            
            # --- Mode-Specific Analysis ---
            if mode == "By Brand (Brand Sentiment)":
                st.subheader("💬 Brand Sentiment Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_counts = df_filtered['Brand Sentiment Label'].value_counts()
                    st.bar_chart(sentiment_counts)
                
                with col2:
                    st.metric("Positive Posts", metrics.get("positive_sentiment", 0))
                    st.metric("Negative Posts", metrics.get("negative_sentiment", 0))
                    st.metric("Neutral Posts", metrics.get("neutral_sentiment", 0))
                    
                    avg_sentiment = df_filtered['Sentiment Score'].mean()
                    st.metric("Avg Sentiment Score", f"{avg_sentiment:.3f}")
            
            elif mode == "By Doctor (DOL Vetting)":
                st.subheader("🌟 DOL / KOL Classification")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    kol_counts = df_filtered['KOL/DOL Label'].value_counts()
                    st.bar_chart(kol_counts)
                
                with col2:
                    st.metric("KOL Posts", metrics.get("kol_count", 0))
                    st.metric("DOL Posts", metrics.get("dol_count", 0))
                    
                    avg_score = df_filtered['DOL Score'].mean()
                    st.metric("Avg DOL Score", f"{avg_score:.1f}")
            
            # --- Results Table ---
            st.subheader("📋 Detailed Results")
            
            # Select columns to display
            base_columns = [
                "Author", "Verified", "Text", "Freshness", 
                "Likes Formatted", "Views Formatted", "Comments Formatted", 
                "Shares Formatted", "Engagement Rate", "URL"
            ]
            
            if mode == "By Doctor (DOL Vetting)":
                display_columns = base_columns + ["KOL/DOL Status Display", "DOL Score"]
            else:
                display_columns = base_columns + ["Brand Sentiment Display", "Sentiment Score"]
            
            # Display table with formatting
            st.dataframe(
                df_filtered[display_columns], 
                use_container_width=True,
                hide_index=True
            )
            
            # --- Export Options ---
            st.subheader("📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv = df_filtered.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'{mode.replace(" ", "_").lower()}_{query}_{timestamp}.csv'
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            
            with col2:
                # JSON Export
                json_data = df_filtered.to_json(orient='records', indent=2)
                json_filename = f'{mode.replace(" ", "_").lower()}_{query}_{timestamp}.json'
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json"
                )
            
            # --- Debug Information ---
            if debug_mode:
                st.subheader("🐛 Debug Information")
                
                with st.expander("Raw Data Sample"):
                    st.json(raw_data[:2] if len(raw_data) >= 2 else raw_data)
                
                with st.expander("Processed Data Info"):
                    st.write(f"**Original posts:** {len(raw_data)}")
                    st.write(f"**Processed posts:** {len(df)}")
                    st.write(f"**Filtered posts:** {len(df_filtered)}")
                    st.write(f"**Columns:** {list(df.columns)}")
                
                with st.expander("Data Quality"):
                    st.write("**Missing Values:**")
                    st.dataframe(df.isnull().sum())

if __name__ == "__main__":
    main()
