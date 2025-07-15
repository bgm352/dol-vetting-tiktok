import re
import pandas as pd
from datetime import datetime
import streamlit as st

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep emojis
    text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF#@]', '', text)
    
    return text

def extract_hashtags(text):
    """Extract hashtags from text"""
    if not text:
        return []
    
    hashtags = re.findall(r'#\w+', text)
    return [tag.lower() for tag in hashtags]

def extract_mentions(text):
    """Extract mentions from text"""
    if not text:
        return []
    
    mentions = re.findall(r'@\w+', text)
    return [mention.lower() for mention in mentions]

def is_medical_content(text, hashtags=None):
    """Check if content is medical-related"""
    if not text:
        return False
    
    medical_keywords = [
        'doctor', 'physician', 'medical', 'health', 'medicine', 'treatment',
        'patient', 'diagnosis', 'symptoms', 'therapy', 'healthcare', 'clinic',
        'hospital', 'nurse', 'surgery', 'prescription', 'vaccine', 'wellness',
        'medstudent', 'residency', 'medschool', 'anatomy', 'pathology'
    ]
    
    medical_hashtags = [
        '#doctor', '#physician', '#medical', '#health', '#medicine',
        '#medstudent', '#medschool', '#healthcare', '#wellness', '#medicalfacts',
        '#healthtips', '#medicalstudent', '#residency', '#medicalschool'
    ]
    
    text_lower = text.lower()
    
    # Check for medical keywords in text
    for keyword in medical_keywords:
        if keyword in text_lower:
            return True
    
    # Check for medical hashtags
    if hashtags:
        for hashtag in hashtags:
            if hashtag.lower() in medical_hashtags:
                return True
    
    return False

def calculate_engagement_rate(likes, views, shares=0, comments=0):
    """Calculate engagement rate"""
    if views == 0:
        return 0
    
    total_engagement = likes + shares + comments
    return (total_engagement / views) * 100

def format_number(num):
    """Format large numbers for display"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

def validate_apify_token(token):
    """Validate Apify API token format"""
    if not token:
        return False
    
    # Basic validation - Apify tokens are typically 32 characters
    if len(token) < 20:
        return False
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_-]+$', token):
        return False
    
    return True

def create_search_summary(queries, max_results, search_type):
    """Create a summary of search parameters"""
    return {
        'queries': queries,
        'total_queries': len(queries),
        'max_results_per_query': max_results,
        'search_type': search_type,
        'timestamp': datetime.now().isoformat()
    }

def filter_dataframe(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Apply engagement filters
    if 'min_likes' in filters:
        filtered_df = filtered_df[filtered_df['likes'] >= filters['min_likes']]
    
    if 'min_views' in filters:
        filtered_df = filtered_df[filtered_df['views'] >= filters['min_views']]
    
    if 'min_comments' in filters:
        filtered_df = filtered_df[filtered_df['comments'] >= filters['min_comments']]
    
    # Apply verification filter
    if filters.get('verified_only', False):
        filtered_df = filtered_df[filtered_df['author_verified'] == True]
    
    # Apply date filters
    if 'start_date' in filters and 'end_date' in filters:
        # Convert created_time to datetime if it's not already
        if 'created_time' in filtered_df.columns:
            filtered_df['created_time'] = pd.to_datetime(filtered_df['created_time'])
            filtered_df = filtered_df[
                (filtered_df['created_time'] >= filters['start_date']) &
                (filtered_df['created_time'] <= filters['end_date'])
            ]
    
    return filtered_df

def generate_analytics_report(df):
    """Generate analytics report from scraped data"""
    if df.empty:
        return None
    
    report = {
        'total_posts': len(df),
        'unique_authors': df['author'].nunique(),
        'verified_authors': df['author_verified'].sum(),
        'avg_likes': df['likes'].mean(),
        'avg_views': df['views'].mean(),
        'avg_shares': df['shares'].mean(),
        'avg_comments': df['comments'].mean(),
        'top_hashtags': [],
        'most_popular_post': None,
        'most_engaged_author': None
    }
    
    # Calculate engagement rates
    df['engagement_rate'] = df.apply(
        lambda x: calculate_engagement_rate(x['likes'], x['views'], x['shares'], x['comments']),
        axis=1
    )
    
    # Find most popular post
    if not df.empty:
        most_popular_idx = df['likes'].idxmax()
        report['most_popular_post'] = {
            'author': df.loc[most_popular_idx, 'author'],
            'likes': df.loc[most_popular_idx, 'likes'],
            'views': df.loc[most_popular_idx, 'views'],
            'text': df.loc[most_popular_idx, 'text'][:100] + '...'
        }
    
    # Find most engaged author
    author_engagement = df.groupby('author').agg({
        'likes': 'sum',
        'views': 'sum',
        'engagement_rate': 'mean'
    }).sort_values('engagement_rate', ascending=False)
    
    if not author_engagement.empty:
        top_author = author_engagement.index[0]
        report['most_engaged_author'] = {
            'name': top_author,
            'total_likes': author_engagement.loc[top_author, 'likes'],
            'total_views': author_engagement.loc[top_author, 'views'],
            'avg_engagement_rate': author_engagement.loc[top_author, 'engagement_rate']
        }
    
    return report

@st.cache_data
def load_cached_data(file_path):
    """Load cached data if available"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

def save_data(df, filename):
    """Save dataframe to file"""
    df.to_csv(filename, index=False)
    return filename
