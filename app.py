import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="TikTok Doctor Content Scraper",
    page_icon="🩺",
    layout="wide"
)

# Title and description
st.title("🩺 TikTok Doctor Content Scraper")
st.markdown("Search and analyze TikTok content from medical professionals")

# Sidebar for configuration
st.sidebar.header("Configuration")
apify_token = st.sidebar.text_input("Apify API Token", type="password", help="Get your token from https://apify.com/account/integrations")

# Search parameters
st.sidebar.header("Search Parameters")
search_queries = st.sidebar.text_area(
    "Search Queries (one per line)", 
    value="doctor\nmedical advice\nphysician\nhealthcare",
    help="Enter search terms related to doctors/medical content"
)

max_results = st.sidebar.number_input("Max Results per Query", min_value=1, max_value=100, value=20)
search_type = st.sidebar.selectbox("Search Type", ["hashtag", "user", "keyword"])

# Advanced options
st.sidebar.header("Advanced Options")
include_engagement = st.sidebar.checkbox("Include Engagement Metrics", value=True)
filter_verified = st.sidebar.checkbox("Verified Accounts Only", value=False)

def run_apify_scraper(token, queries, max_results, search_type):
    """Run the Apify TikTok scraper"""
    
    # Apify API endpoint
    actor_id = "clockworks/tiktok-scraper"
    url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
    
    # Prepare input data
    input_data = {
        "searchQueries": queries,
        "resultsPerPage": max_results,
        "searchType": search_type,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshowImages": False
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Start the scraper
    response = requests.post(url, json=input_data, headers=headers)
    
    if response.status_code != 201:
        st.error(f"Failed to start scraper: {response.text}")
        return None
    
    run_data = response.json()
    run_id = run_data["data"]["id"]
    
    # Poll for completion
    status_url = f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        
        status = status_data["data"]["status"]
        status_text.text(f"Status: {status}")
        
        if status == "SUCCEEDED":
            progress_bar.progress(100)
            break
        elif status == "FAILED":
            st.error("Scraper failed to complete")
            return None
        
        time.sleep(5)
        progress_bar.progress(min(progress_bar.progress + 10, 90))
    
    # Get results
    results_url = f"https://api.apify.com/v2/datasets/{run_data['data']['defaultDatasetId']}/items"
    results_response = requests.get(results_url, headers=headers)
    
    if results_response.status_code == 200:
        return results_response.json()
    else:
        st.error("Failed to retrieve results")
        return None

def process_results(data):
    """Process and format the scraped data"""
    if not data:
        return pd.DataFrame()
    
    processed_data = []
    for item in data:
        processed_item = {
            'id': item.get('id', ''),
            'text': item.get('text', ''),
            'author': item.get('authorMeta', {}).get('name', ''),
            'author_verified': item.get('authorMeta', {}).get('verified', False),
            'author_followers': item.get('authorMeta', {}).get('fans', 0),
            'likes': item.get('diggCount', 0),
            'shares': item.get('shareCount', 0),
            'comments': item.get('commentCount', 0),
            'views': item.get('playCount', 0),
            'created_time': item.get('createTime', ''),
            'hashtags': ', '.join([tag.get('name', '') for tag in item.get('hashtags', [])]),
            'url': f"https://www.tiktok.com/@{item.get('authorMeta', {}).get('name', '')}/video/{item.get('id', '')}"
        }
        processed_data.append(processed_item)
    
    return pd.DataFrame(processed_data)

def main():
    # Check if token is provided
    if not apify_token:
        st.warning("Please enter your Apify API token in the sidebar to continue.")
        st.markdown("### How to get your Apify API token:")
        st.markdown("1. Go to [Apify](https://apify.com)")
        st.markdown("2. Sign up or log in")
        st.markdown("3. Go to Account → Integrations")
        st.markdown("4. Copy your API token")
        return
    
    # Process search queries
    queries = [q.strip() for q in search_queries.split('\n') if q.strip()]
    
    if not queries:
        st.warning("Please enter at least one search query.")
        return
    
    # Display search parameters
    st.header("Search Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Queries", len(queries))
    with col2:
        st.metric("Max Results per Query", max_results)
    with col3:
        st.metric("Search Type", search_type)
    
    # Run scraper button
    if st.button("🚀 Start Scraping", type="primary"):
        st.header("Scraping Progress")
        
        # Run the scraper
        results = run_apify_scraper(apify_token, queries, max_results, search_type)
        
        if results:
            # Process results
            df = process_results(results)
            
            if not df.empty:
                st.success(f"Successfully scraped {len(df)} TikTok posts!")
                
                # Display results
                st.header("Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", len(df))
                with col2:
                    st.metric("Verified Authors", df['author_verified'].sum())
                with col3:
                    st.metric("Avg Likes", f"{df['likes'].mean():.0f}")
                with col4:
                    st.metric("Avg Views", f"{df['views'].mean():.0f}")
                
                # Filters
                st.subheader("Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_likes = st.slider("Minimum Likes", 0, int(df['likes'].max()), 0)
                with col2:
                    show_verified_only = st.checkbox("Show Verified Only", value=False)
                
                # Apply filters
                filtered_df = df[df['likes'] >= min_likes]
                if show_verified_only:
                    filtered_df = filtered_df[filtered_df['author_verified'] == True]
                
                # Display filtered data
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    column_config={
                        "url": st.column_config.LinkColumn("TikTok URL"),
                        "likes": st.column_config.NumberColumn("Likes", format="%d"),
                        "views": st.column_config.NumberColumn("Views", format="%d"),
                        "shares": st.column_config.NumberColumn("Shares", format="%d"),
                        "comments": st.column_config.NumberColumn("Comments", format="%d"),
                    }
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"tiktok_doctor_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No results found. Try different search queries.")
        else:
            st.error("Failed to scrape data. Please check your API token and try again.")

if __name__ == "__main__":
    main()
