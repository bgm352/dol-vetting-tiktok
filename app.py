import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# Sample Data Generation
# Replace with real data from APIs
# -------------------------------
locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
dates = pd.date_range("2025-07-01", periods=30)

# Keyword rankings per location
keyword_data = pd.DataFrame({
    "Date": np.tile(dates, len(locations)),
    "Location": np.repeat(locations, len(dates)),
    "Keyword": np.random.choice(["plumber near me", "emergency repair", "local plumber"], len(locations) * len(dates)),
    "Rank": np.random.randint(1, 50, len(locations) * len(dates)),
    "Clicks": np.random.randint(10, 300, len(locations) * len(dates))
})

# Review data
review_data = pd.DataFrame({
    "Location": locations,
    "Avg Rating": np.random.uniform(3.5, 5.0, len(locations)),
    "Total Reviews": np.random.randint(50, 500, len(locations)),
    "Positive %": np.random.uniform(70, 95, len(locations)),
    "Negative %": lambda df: 100 - df["Positive %"]
})

# Engagement metrics
engagement_data = pd.DataFrame({
    "Location": locations,
    "Clicks for Directions": np.random.randint(100, 1000, len(locations)),
    "Calls": np.random.randint(50, 500, len(locations)),
    "Website Visits": np.random.randint(200, 2000, len(locations))
})

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Geo SEO Scoreboard", layout="wide")

st.sidebar.title("Settings")
selected_location = st.sidebar.selectbox("Select Location", ["All"] + locations)
date_range = st.sidebar.date_input("Select Date Range", [dates.min(), dates.max()])

st.title("📊 Geo SEO Scoreboard")

# Tabs
tabs = st.tabs(["Overview", "Keyword Performance", "Reviews", "Engagement", "Technical SEO"])

# -------------------------------
# Overview Tab
# -------------------------------
with tabs[0]:
    st.subheader("SEO Overview")
    if selected_location != "All":
        filtered_data = keyword_data[keyword_data["Location"] == selected_location]
    else:
        filtered_data = keyword_data

    avg_rank = filtered_data["Rank"].mean()
    total_clicks = filtered_data["Clicks"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Rank", f"{avg_rank:.2f}")
    col2.metric("Total Clicks", f"{total_clicks:,}")

    # Map visualization (heatmap style)
    coords = {
        "New York": [40.7128, -74.0060],
        "Los Angeles": [34.0522, -118.2437],
        "Chicago": [41.8781, -87.6298],
        "Houston": [29.7604, -95.3698],
        "Phoenix": [33.4484, -112.0740],
    }
    map_df = pd.DataFrame({
        "Location": locations,
        "lat": [coords[loc][0] for loc in locations],
        "lon": [coords[loc][1] for loc in locations],
        "Clicks": engagement_data["Website Visits"]
    })
    fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="Clicks", hover_name="Location",
                                color="Clicks", color_continuous_scale="Turbo", zoom=3)
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------
# Keyword Performance Tab
# -------------------------------
with tabs[1]:
    st.subheader("Keyword Rankings by Location")
    if selected_location != "All":
        data_kw = keyword_data[keyword_data["Location"] == selected_location]
    else:
        data_kw = keyword_data

    fig_kw = px.line(data_kw, x="Date", y="Rank", color="Keyword",
                     title="Daily Keyword Rankings", markers=True)
    fig_kw.update_yaxes(autorange="reversed")  # Lower rank is better
    st.plotly_chart(fig_kw, use_container_width=True)
    st.dataframe(data_kw.head(15))

# -------------------------------
# Reviews Tab
# -------------------------------
with tabs[2]:
    st.subheader("Review Sentiment")
    fig_reviews = px.bar(review_data, x="Location", y="Avg Rating", color="Avg Rating",
                         text_auto=".2f", title="Average Star Ratings per Location")
    st.plotly_chart(fig_reviews, use_container_width=True)
    st.dataframe(review_data)

# -------------------------------
# Engagement Tab
# -------------------------------
with tabs[3]:
    st.subheader("Local Engagement Metrics")
    fig_eng = px.bar(engagement_data.melt(id_vars="Location"),
                     x="Location", y="value", color="variable",
                     barmode="group", title="Engagement Actions")
    st.plotly_chart(fig_eng, use_container_width=True)

# -------------------------------
# Technical SEO Tab
# -------------------------------
with tabs[4]:
    st.subheader("Technical SEO Health (Example)")
    st.write("Example data – should be connected to crawling tools like Screaming Frog API")
    tech_data = pd.DataFrame({
        "Location": locations,
        "Mobile Usability Issues": np.random.randint(0, 5, len(locations)),
        "Page Speed (ms)": np.random.randint(1500, 4000, len(locations)),
        "Crawl Errors": np.random.randint(0, 10, len(locations))
    })
    st.dataframe(tech_data)

