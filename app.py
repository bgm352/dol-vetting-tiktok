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
            
            # Process timestamp - convert from Unix timestamp to readable format
            create_time = item.get('createTime', 0)
            if create_time:
                try:
                    # Convert Unix timestamp to datetime
                    timestamp = datetime.fromtimestamp(create_time)
                    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Calculate how many days ago
                    days_ago = (datetime.now() - timestamp).days
                    if days_ago == 0:
                        freshness = "Today"
                    elif days_ago == 1:
                        freshness = "1 day ago"
                    elif days_ago < 7:
                        freshness = f"{days_ago} days ago"
                    elif days_ago < 30:
                        weeks = days_ago // 7
                        freshness = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                    elif days_ago < 365:
                        months = days_ago // 30
                        freshness = f"{months} month{'s' if months > 1 else ''} ago"
                    else:
                        years = days_ago // 365
                        freshness = f"{years} year{'s' if years > 1 else ''} ago"
                        
                except (ValueError, OSError):
                    formatted_timestamp = "Unknown"
                    freshness = "Unknown"
            else:
                formatted_timestamp = "Unknown"
                freshness = "Unknown"
            
            processed_item = {
                'author': author_meta.get('name', 'Unknown'),
                'author_verified': author_meta.get('verified', False),
                'text': item.get('text', ''),
                'likes': item.get('diggCount', 0),
                'shares': item.get('shareCount', 0),
                'comments': item.get('commentCount', 0),
                'views': item.get('playCount', 0),
                'data_freshness': freshness,
                'timestamp': formatted_timestamp,
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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", len(df))
                with col2:
                    st.metric("Verified Authors", df['author_verified'].sum())
                with col3:
                    st.metric("Avg Likes", f"{df['likes'].mean():.0f}")
                with col4:
                    # Show freshness breakdown
                    recent_posts = len(df[df['data_freshness'].str.contains("Today|day", na=False)])
                    st.metric("Recent Posts", f"{recent_posts}")
                
                # Show data freshness distribution
                st.subheader("📊 Data Freshness Overview")
                freshness_counts = df['data_freshness'].value_counts()
                st.bar_chart(freshness_counts)
                
                # Add freshness filter
                st.subheader("🕒 Filter by Data Freshness")
                
                # Show current freshness distribution
                col1, col2 = st.columns([1, 2])
                with col1:
                    freshness_options = ["All", "Today", "This week", "This month", "Older"]
                    selected_freshness = st.selectbox("Show posts from:", freshness_options)
                with col2:
                    # Show available freshness categories
                    available_freshness = df['data_freshness'].unique()
                    st.write("**Available in data:**", ", ".join(available_freshness[:5]))
                
                # Filter data based on freshness
                filtered_df = df.copy()
                if selected_freshness != "All":
                    if selected_freshness == "Today":
                        filtered_df = df[df['data_freshness'] == "Today"]
                    elif selected_freshness == "This week":
                        filtered_df = df[df['data_freshness'].str.contains("day|Today", na=False)]
                    elif selected_freshness == "This month":
                        filtered_df = df[df['data_freshness'].str.contains("day|week|Today", na=False)]
                    elif selected_freshness == "Older":
                        filtered_df = df[df['data_freshness'].str.contains("month|year", na=False)]
                
                # Update metrics for filtered data
                if not filtered_df.empty:
                    st.subheader("📈 Filtered Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Filtered Posts", len(filtered_df))
                    with col2:
                        st.metric("Verified Authors", filtered_df['author_verified'].sum())
                    with col3:
                        st.metric("Avg Likes", f"{filtered_df['likes'].mean():.0f}")
                    with col4:
                        st.metric("Total Views", f"{filtered_df['views'].sum():,}")
                else:
                    st.warning("No posts match the selected freshness filter.")
                    filtered_df = df  # Reset to show all data
                
                # Display data with column configuration
                st.subheader("📋 TikTok Posts Data")
                
                # Show column info
                st.info("💡 **Data Freshness** shows how recent each post is. Click column headers to sort!")
                
                st.dataframe(
                    filtered_df, 
                    use_container_width=True,
                    column_config={
                        "data_freshness": st.column_config.TextColumn(
                            "Data Freshness",
                            help="How recent the post is (e.g., 'Today', '2 days ago')",
                            width="medium"
                        ),
                        "timestamp": st.column_config.DatetimeColumn(
                            "Timestamp",
                            help="Exact date and time of post",
                            width="medium"
                        ),
                        "author": st.column_config.TextColumn(
                            "Author",
                            help="TikTok username",
                            width="medium"
                        ),
                        "text": st.column_config.TextColumn(
                            "Post Text",
                            help="Caption/description of the TikTok video",
                            width="large"
                        ),
                        "url": st.column_config.LinkColumn(
                            "TikTok URL",
                            help="Link to original TikTok video"
                        ),
                        "likes": st.column_config.NumberColumn(
                            "Likes",
                            format="%d"
                        ),
                        "views": st.column_config.NumberColumn(
                            "Views", 
                            format="%d"
                        ),
                        "comments": st.column_config.NumberColumn(
                            "Comments", 
                            format="%d"
                        ),
                        "shares": st.column_config.NumberColumn(
                            "Shares", 
                            format="%d"
                        ),
                        "author_verified": st.column_config.CheckboxColumn(
                            "Verified",
                            help="Is the author verified?"
                        )
                    }
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
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
