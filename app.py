import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime
from textblob import TextBlob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from typing import List, Dict, Optional
import re

# ------------------
# Logging Setup
# ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------
# Page Setup
# ------------------
st.set_page_config(page_title="TikTok Selenium Scraper", page_icon="🤖", layout="wide")
st.title("🤖 TikTok Selenium Scraper - Direct Web Scraping")
st.caption("Scrape TikTok content directly using Selenium browser automation.")

# ------------------
# Selenium Configuration
# ------------------
st.sidebar.header("🔧 Selenium Configuration")

# Chrome options
headless_mode = st.sidebar.checkbox("Headless Mode", value=True, help="Run browser in background")
use_mobile_agent = st.sidebar.checkbox("Mobile User Agent", value=True, help="Use mobile user agent")
page_load_timeout = st.sidebar.slider("Page Load Timeout (seconds)", 10, 60, 30)
implicit_wait = st.sidebar.slider("Implicit Wait (seconds)", 5, 30, 10)

# Search settings
st.sidebar.header("🔍 Search Settings")
query = st.sidebar.text_input("Search Term", value="doctor")
max_results = st.sidebar.slider("Max Posts to Scrape", 5, 100, 20)
scroll_pause = st.sidebar.slider("Scroll Pause (seconds)", 1, 5, 2)

# ------------------
# Selenium Helper Functions
# ------------------

def setup_chrome_driver(headless: bool = True, mobile: bool = True) -> webdriver.Chrome:
    """Setup Chrome WebDriver with optimal settings for TikTok"""
    
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")
    
    # Essential Chrome options for stability
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Anti-detection measures
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Mobile user agent for better compatibility
    if mobile:
        mobile_user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1"
        chrome_options.add_argument(f"--user-agent={mobile_user_agent}")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.implicitly_wait(implicit_wait)
        driver.set_page_load_timeout(page_load_timeout)
        return driver
    except WebDriverException as e:
        st.error(f"Failed to initialize Chrome driver: {e}")
        st.error("Make sure ChromeDriver is installed and in PATH")
        return None

def extract_post_data(driver: webdriver.Chrome, post_element) -> Dict:
    """Extract data from a single TikTok post element"""
    
    try:
        # Extract author information
        try:
            author_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='video-author-uniqueid']")
            author = author_element.text.strip('@')
        except NoSuchElementException:
            author = "Unknown"
        
        # Extract post description
        try:
            desc_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='video-desc']")
            description = desc_element.text
        except NoSuchElementException:
            description = ""
        
        # Extract engagement metrics
        try:
            like_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='like-count']")
            likes = parse_count(like_element.text)
        except NoSuchElementException:
            likes = 0
        
        try:
            comment_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='comment-count']")
            comments = parse_count(comment_element.text)
        except NoSuchElementException:
            comments = 0
        
        try:
            share_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='share-count']")
            shares = parse_count(share_element.text)
        except NoSuchElementException:
            shares = 0
        
        # Extract video URL
        try:
            video_element = post_element.find_element(By.CSS_SELECTOR, "a[href*='/video/']")
            video_url = video_element.get_attribute('href')
        except NoSuchElementException:
            video_url = ""
        
        # Extract timestamp (if available)
        try:
            time_element = post_element.find_element(By.CSS_SELECTOR, "[data-e2e='video-time']")
            timestamp = time_element.text
        except NoSuchElementException:
            timestamp = ""
        
        return {
            'author': author,
            'description': description,
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'video_url': video_url,
            'timestamp': timestamp,
            'scraped_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error extracting post data: {e}")
        return None

def parse_count(count_text: str) -> int:
    """Parse TikTok count format (e.g., '1.2K', '5.5M') to integer"""
    
    if not count_text:
        return 0
    
    count_text = count_text.strip().upper()
    
    # Remove any non-numeric characters except K, M, B
    count_text = re.sub(r'[^\d.KMB]', '', count_text)
    
    if 'K' in count_text:
        return int(float(count_text.replace('K', '')) * 1000)
    elif 'M' in count_text:
        return int(float(count_text.replace('M', '')) * 1000000)
    elif 'B' in count_text:
        return int(float(count_text.replace('B', '')) * 1000000000)
    else:
        try:
            return int(float(count_text))
        except ValueError:
            return 0

def scrape_tiktok_search(query: str, max_results: int = 20) -> List[Dict]:
    """Scrape TikTok search results using Selenium"""
    
    results = []
    driver = None
    
    try:
        # Setup driver
        driver = setup_chrome_driver(headless=headless_mode, mobile=use_mobile_agent)
        if not driver:
            return results
        
        # Navigate to TikTok search
        search_url = f"https://www.tiktok.com/search/video?q={query}"
        st.info(f"Navigating to: {search_url}")
        
        driver.get(search_url)
        
        # Wait for search results to load
        wait = WebDriverWait(driver, 20)
        
        # Accept cookies if popup appears
        try:
            cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-e2e='cookie-banner-accept']")))
            cookie_button.click()
            time.sleep(2)
        except TimeoutException:
            pass  # No cookie banner appeared
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scroll and collect posts
        posts_collected = 0
        scroll_attempts = 0
        max_scroll_attempts = 50
        
        while posts_collected < max_results and scroll_attempts < max_scroll_attempts:
            try:
                # Find video containers
                video_containers = driver.find_elements(By.CSS_SELECTOR, "[data-e2e='search-video-item']")
                
                if not video_containers:
                    # Alternative selector
                    video_containers = driver.find_elements(By.CSS_SELECTOR, "div[data-e2e*='video']")
                
                # Process new posts
                for container in video_containers[posts_collected:]:
                    if posts_collected >= max_results:
                        break
                    
                    post_data = extract_post_data(driver, container)
                    if post_data:
                        results.append(post_data)
                        posts_collected += 1
                        
                        # Update progress
                        progress = posts_collected / max_results
                        progress_bar.progress(progress)
                        status_text.text(f"Scraped {posts_collected}/{max_results} posts")
                
                # Scroll down to load more content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
                
                scroll_attempts += 1
                
                # Check if we've reached the end
                if len(video_containers) == posts_collected:
                    # Try to load more content
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight + 1000);")
                    time.sleep(scroll_pause * 2)
                
            except Exception as e:
                logger.error(f"Error during scrolling: {e}")
                scroll_attempts += 1
                continue
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Scraping completed! Collected {len(results)} posts")
        
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        logger.error(f"Scraping error: {e}")
    
    finally:
        if driver:
            driver.quit()
    
    return results

def process_scraped_data(raw_data: List[Dict]) -> pd.DataFrame:
    """Process scraped data into analysis-ready format"""
    
    if not raw_data:
        return pd.DataFrame()
    
    processed_data = []
    
    for post in raw_data:
        try:
            # Calculate engagement metrics
            total_engagement = post['likes'] + post['comments'] + post['shares']
            
            # Sentiment analysis
            sentiment_score = TextBlob(post['description']).sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
                sentiment_emoji = "😊"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
                sentiment_emoji = "😞"
            else:
                sentiment_label = "Neutral"
                sentiment_emoji = "😐"
            
            # DOL scoring (simplified)
            dol_score = max(1, min(10, int((sentiment_score + 1) * 5)))
            
            if dol_score >= 8:
                dol_label = "KOL"
                dol_emoji = "🌟"
            elif dol_score >= 5:
                dol_label = "DOL"
                dol_emoji = "👍"
            else:
                dol_label = "Not Suitable"
                dol_emoji = "❌"
            
            processed_data.append({
                'Author': post['author'],
                'Description': post['description'][:200] + "..." if len(post['description']) > 200 else post['description'],
                'Full Description': post['description'],
                'Likes': post['likes'],
                'Comments': post['comments'],
                'Shares': post['shares'],
                'Total Engagement': total_engagement,
                'Video URL': post['video_url'],
                'Timestamp': post['timestamp'],
                'Scraped At': post['scraped_at'],
                'Sentiment Score': round(sentiment_score, 3),
                'Sentiment Label': sentiment_label,
                'Sentiment Display': f"{sentiment_emoji} {sentiment_label}",
                'DOL Score': dol_score,
                'DOL Label': dol_label,
                'DOL Display': f"{dol_emoji} {dol_label}",
            })
            
        except Exception as e:
            logger.error(f"Error processing post: {e}")
            continue
    
    return pd.DataFrame(processed_data)

# ------------------
# Main Application
# ------------------

def main():
    """Main application flow"""
    
    st.warning("⚠️ **Important Notes:**")
    st.markdown("""
    - **ChromeDriver Required**: Install ChromeDriver and add to PATH
    - **Rate Limiting**: TikTok may block aggressive scraping
    - **Legal Compliance**: Ensure compliance with TikTok's ToS
    - **Instability**: Direct scraping can be brittle due to site changes
    """)
    
    # Check if ChromeDriver is available
    if st.button("🔍 Test ChromeDriver"):
        try:
            driver = setup_chrome_driver(headless=True)
            if driver:
                st.success("✅ ChromeDriver is working correctly!")
                driver.quit()
            else:
                st.error("❌ ChromeDriver setup failed")
        except Exception as e:
            st.error(f"❌ ChromeDriver error: {e}")
    
    if st.button("🚀 Start Scraping", type="primary"):
        
        if not query.strip():
            st.error("Please enter a search term")
            return
        
        with st.spinner("Initializing browser..."):
            # Run scraping
            raw_data = scrape_tiktok_search(query, max_results)
        
        if not raw_data:
            st.warning("No data was scraped. This could be due to:")
            st.markdown("""
            - TikTok blocking the scraper
            - Changed website structure
            - Network issues
            - Anti-bot measures
            """)
            return
        
        # Process data
        df = process_scraped_data(raw_data)
        
        if df.empty:
            st.warning("No valid posts could be processed")
            return
        
        st.success(f"✅ Successfully scraped {len(df)} posts!")
        
        # Display results
        st.subheader("📊 Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Posts", len(df))
        col2.metric("Avg Likes", f"{df['Likes'].mean():.0f}")
        col3.metric("Total Engagement", f"{df['Total Engagement'].sum():,}")
        col4.metric("Unique Authors", df['Author'].nunique())
        
        # Sentiment analysis
        st.subheader("💬 Sentiment Analysis")
        sentiment_counts = df['Sentiment Label'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # DOL analysis
        st.subheader("🌟 DOL/KOL Analysis")
        dol_counts = df['DOL Label'].value_counts()
        st.bar_chart(dol_counts)
        
        # Data table
        st.subheader("📋 Scraped Data")
        display_columns = [
            'Author', 'Description', 'Likes', 'Comments', 'Shares',
            'Sentiment Display', 'DOL Display', 'Video URL'
        ]
        st.dataframe(df[display_columns], use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"tiktok_selenium_scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
