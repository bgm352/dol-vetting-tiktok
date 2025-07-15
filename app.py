import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="TikTok Doctor Content Scraper",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 TikTok Doctor Content Scraper")
st.markdown("Search and analyze TikTok content from medical professionals")

# Sidebar
st.sidebar.header("Configuration")
apify_token = st.sidebar.text_input("Apify API Token", type="password")

st.sidebar.header("Search Parameters")
search_query = st.sidebar.text_input("Search Query", value="doctor")
max_results = st.sidebar.number_input("Max Results", min_value=1, max_value=50, value=20)

def run_scraper(token, query, max_results):
    """Run the Apify TikTok scraper"""
    try:
        # Apify API endpoint
        url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
        
        # Input data
        input_data = {
            "searchQueries": [query],
            "resultsPerPage": max_results,
            "searchType": "keyword"
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Start scraper
        response = requests.post(url, json=input_data, headers=headers)
        
        if response.status_code != 201:
            st.error(f"Failed to start scraper: {response.status_code}")
            return None
        
        run_data = response.json()
        run_id = run_data["data"]["id"]
        
        # Wait for completion
        status_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs/{run_id}"
        
        with st.spinner("Scraping TikTok data..."):
            while True:
                status_response = requests.get(status_url, headers=headers)
                status_data = status_response.json()
                
                status = status_data["data"]["status"]
                
                if status == "SUCCEEDED":
                    break
                elif status == "FAILED":
                    st.error("Scraper failed")
                    return None
                
                time.sleep(5)
        
        # Get results
        dataset_id = run_data["data"]["defaultDatasetId"]
        results_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        results_response = requests.get(results_url, headers=headers)
        
        if results_response.status_code == 200:
            return results_response.json()
        else:
            st.error("Failed to get results")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def process_data(raw_data):
    """Process the scraped data"""
    if not raw_data:
        return pd.DataFrame()
    
    processed = []
    for item in raw_data:
        try:
            author_meta = item.get('authorMeta', {})
            processed_item = {
                'author': author_meta.get('name', 'Unknown'),
                'author_verified': author_meta.get('verified', False),
                'text': item.get('text', ''),
                'likes': item.get('diggCount', 0),
                'shares': item.get('shareCount', 0),
                'comments': item.get('commentCount', 0),
                'views': item.get('playCount', 0),
                'url': f"https://www.tiktok.com/@{author_meta.get('name', '')}/video/{item.get('id', '')}"
            }
            processed.append(processed_item)
        except Exception as e:
            continue
    
    return pd.DataFrame(processed)

# Main app
if not apify_token:
    st.warning("Please enter your Apify API token in the sidebar.")
    st.info("Get your token from: https://apify.com/account/integrations")
else:
    if st.button("🚀 Start Scraping"):
        results = run_scraper(apify_token, search_query, max_results)
        
        if results:
            df = process_data(results)
            
            if not df.empty:
                st.success(f"Found {len(df)} TikTok posts!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Posts", len(df))
                with col2:
                    st.metric("Verified Authors", df['author_verified'].sum())
                with col3:
                    st.metric("Avg Likes", f"{df['likes'].mean():.0f}")
                
                # Display data
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"tiktok_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.warning("No results found")
        else:
            st.error("Failed to scrape data")
