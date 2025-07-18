import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk

# Friendly missing package guard for apify-client
try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error(
        "The required package 'apify-client' is not installed. "
        "Please add 'apify-client' to requirements.txt or run `pip install apify-client` in your terminal. "
        "See https://pypi.org/project/apify-client/ for more info."
    )
    st.stop()  # Halts Streamlit app safely

# LLM libraries (with friendly guards)
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from textblob import TextBlob
    nltk.download("punkt")
except ModuleNotFoundError:
    st.error("The required packages 'textblob' and/or 'nltk' are not installed! Please add them to requirements.txt.")
    st.stop()

st.set_page_config(page_title="TikTok/Threads Vetting Tool",
                   layout="wide",
                   page_icon="🩺")
st.title("🩺 TikTok & Threads Vetting Tool (DOL/KOL)")

# --- SECRETS/API KEYS ---
apify_api_key = st.sidebar.text_input(
    "Apify API Token", type="password", help="Get from https://console.apify.com/account#/integrations")

# --- LLM PROVIDER UI ---
st.sidebar.header("LLM Options")
llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    options=["OpenAI GPT", "Google Gemini"],
    help="Select which LLM to use for notes/scoring."
)
openai_api_key = None
gemini_api_key = None
if llm_provider == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Get from https://platform.openai.com/api-keys")
elif llm_provider == "Google Gemini":
    gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Get from https://aistudio.google.com/app/apikey")

# --- PLATFORM MODE ---
mode = st.sidebar.radio("Scoring Strategy", [
    "By Doctor (DOL Vetting)",
    "By Brand (Brand Sentiment)"
], help="DOL/KOL: ranks experts/influencers, Brand: rates brand sentiment.")

query = st.sidebar.text_input(
    "TikTok Search Term",
    value="doctor" if "Doctor" in mode else "brandA",
    help="Keyword/topic to search TikTok for."
)
max_results = st.sidebar.slider(
    "Max TikTok Posts", 5, 50, 10, help="# of TikTok posts to analyze."
)

# --- THREADS UX ---
st.sidebar.header("Meta Threads Scraper")
threads_enabled = st.sidebar.checkbox("Include Meta Threads Data", value=False)
threads_username = st.sidebar.text_input(
    "Threads Username (optional)", value="", help="Optional, public Threads username."
)

# --- FAQ / ONBOARDING ---
with st.sidebar.expander("❓ FAQ & Help"):
    st.markdown("""
    - **How does scoring work?**  
      Doctor/brand vetting uses sentiment + keyword analysis and may use LLM notes/scoring if enabled.
    - **What are my data/privacy rights?**  
      Processing is per session, your queries aren't stored centrally unless on admin panel.
    - **Want to contribute?**  
      See the project repo or email support.
    - **If you see a 'missing package' error, your admin must add the indicated library to requirements.txt (`apify-client`, `textblob`, etc) and redeploy.**
    """)

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

if "top_kols" not in st.session_state:
    st.session_state["top_kols"] = []
if "analysis_count" not in st.session_state:
    st.session_state["analysis_count"] = 0
if "feedback_logs" not in st.session_state:
    st.session_state["feedback_logs"] = []
if "last_fetch_time" not in st.session_state:
    st.session_state["last_fetch_time"] = None

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
            rationale += f'. Transcript snippet: “{transcript[:90].strip()}...”'
        else:
            rationale += f". {transcript}"
        rationale += f" (Score: {score}/10)"
    else:
        rationale = f"{name} appears to express {sentiment.lower()} brand sentiment."
        if transcript and "not found" not in transcript:
            rationale += f' Transcript: “{transcript[:90].strip()}...”'
        else:
            rationale += f". {transcript}"
    return rationale

def get_llm_response(prompt, model="gpt-3.5-turbo", provider="OpenAI GPT", openai_api_key=None, gemini_api_key=None):
    if provider == "OpenAI GPT":
        if not openai:
            return "OpenAI SDK not installed."
        if not openai_api_key:
            return "No OpenAI key provided."
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.6,
        )
        return response.choices[0].message['content'].strip()
    elif provider == "Google Gemini":
        if not genai:
            return "Gemini SDK not installed."
        if not gemini_api_key:
            return "No Gemini key provided."
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        resp = model.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, 'text') else str(resp)
    else:
        return "Unknown provider"

def generate_llm_notes(posts_df, note_template, provider, openai_api_key=None, gemini_api_key=None):
    posts_texts = "\n\n".join([
        f"{i+1}. Author: {row['Author']}\nContent: {row['Text']}\nTranscript: {row['Transcript']}"
        for i, row in posts_df.iterrows()
    ])
    prompt = f"""
    Using the following social posts, generate notes for DOL/KOL vetting in this structure:
    {note_template}

    Social posts:
    {posts_texts}
    Return in markdown, each section with a title (Summary, Relevance, Strengths, Weaknesses, Red Flags, Brand Mentions, Research Notes, etc).
    """
    return get_llm_response(
        prompt,
        provider=provider,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key
    )

def generate_llm_score(notes, provider, openai_api_key=None, gemini_api_key=None):
    prompt = f"""
    You are a medical affairs expert. Based on these vetting notes and the evidence, assign a DOL suitability score (1=poor, 10=ideal) for pharma collaboration and give a rationale.

    Notes: {notes}

    Respond in this format (YAML):
    score: <1-10>
    rationale: <short explanation>
    """
    return get_llm_response(
        prompt,
        provider=provider,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key
    )

def fetch_tiktok_transcripts_apify(api_token, video_urls):
    client = ApifyClient(api_token)
    run_input = { "videos": video_urls }
    run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input=run_input)
    transcripts = {}
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        video_id = item.get("id")
        transcript = item.get("transcript", "")
        transcripts[video_id] = transcript
    return transcripts

@st.cache_data(show_spinner=False)
def run_apify_scraper(api_key, query, max_results):
    try:
        run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
        start = requests.post(run_url, headers={"Authorization": f"Bearer {api_key}"}, json={
            "searchQueries": [query], "resultsPerPage": max_results, "searchType": "keyword"
        }).json()
        run_id = start.get("data", {}).get("id")
        for _ in range(60):  # up to 5 min
            resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}).json()
            if resp["data"]["status"] == "SUCCEEDED":
                dataset_id = resp["data"].get("defaultDatasetId")
                break
            time.sleep(5)
        else:
            st.error("TikTok scraper run timed out.")
            return []
        data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        return requests.get(data_url).json()
    except Exception:
        st.error("💥 Failed to fetch TikTok data from Apify.")
        return []

def process_posts(posts, transcript_map, fetch_time=None, last_fetch_time=None):
    results = []
    for i, post in enumerate(posts):
        try:
            author = post.get("authorMeta", {}).get("name", "")
            text = post.get("text", "")
            post_id = post.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"
            tscript = transcript_map.get(post_id, "Transcript not found")
            ts = datetime.fromtimestamp(post.get("createTime", time.time()))
            body = f"{text} {tscript}"
            sentiment_score = TextBlob(body).sentiment.polarity
            sentiment = classify_sentiment(sentiment_score)
            dol_score = max(min(round((sentiment_score * 10) + 5), 10), 1)
            kol_dol_label = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, tscript, author, dol_score, sentiment, mode)
            is_new = "🟢 New" if last_fetch_time is None or ts > last_fetch_time else "Old"
            results.append({
                "Author": author,
                "Text": text.strip(),
                "Transcript": tscript or "Transcript not found",
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
                "LLM DOL Score Rationale": rationale,
                "Data Fetched At": fetch_time,
                "Is New": is_new,
            })
        except Exception as e:
            st.warning(f"⛔ Skipped 1 post due to error: {e}")
            continue
    return pd.DataFrame(results)

# ----------- MAIN APP FLOW -----------
notes_text = ""
run_analysis = st.button("Go 🚀", use_container_width=True)

if run_analysis and apify_api_key:
    st.session_state["analysis_count"] += 1

    fetch_time = datetime.now()
    last_fetch_time = st.session_state.get("last_fetch_time", None)

    tiktok_data = run_apify_scraper(apify_api_key, query, max_results)
    if not tiktok_data:
        st.warning("No TikTok posts found.")
    else:
        video_urls = [f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}' for p in tiktok_data]
        transcript_map = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
        st.success(f"✅ {len(tiktok_data)} TikTok posts scraped.")
        df = process_posts(
            tiktok_data,
            transcript_map=transcript_map,
            fetch_time=fetch_time,
            last_fetch_time=last_fetch_time,
        )
        st.session_state["last_fetch_time"] = fetch_time

        st.metric("TikTok Posts", len(df))

        if not df.empty:
            kols = df[df["KOL/DOL Status"].str.contains("KOL|DOL", regex=True)]["Author"].unique().tolist()
            st.session_state["top_kols"].extend(x for x in kols if x not in st.session_state["top_kols"])

        st.subheader("📋 TikTok Analysis Results")

        tiktok_cols = [
            "Author", "Text", "Transcript", "Likes", "Views", "Comments", "Shares",
            "DOL Score", "Sentiment Score", "Post URL", "KOL/DOL Status",
            "Brand Sentiment Label", "LLM DOL Score Rationale",
            "Timestamp", "Data Fetched At", "Is New"
        ]

        display_option = st.radio("Choose display columns:", [
            "All columns",
            "Only main info",
            "Just DOL / Sentiment"
        ])
        if display_option == "All columns":
            columns = tiktok_cols
        elif display_option == "Only main info":
            columns = ["Author", "Text", "Likes", "Views", "Comments", "Shares", "DOL Score", "Timestamp", "Is New"]
        else:
            columns = ["Author", "KOL/DOL Status", "DOL Score", "Sentiment Score", "Brand Sentiment Label", "Is New"]

        dol_min, dol_max = st.slider("Select DOL Score Range", 1, 10, (1, 10))
        filtered_df = df[(df["DOL Score"] >= dol_min) & (df["DOL Score"] <= dol_max)]

        st.dataframe(filtered_df[columns], use_container_width=True)
        st.markdown("**🟢 New** = Collected since your last run; **Old** = Already seen before.")

        st.download_button("Download TikTok CSV", filtered_df[columns].to_csv(index=False),
                          file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

        if st.checkbox("Show Raw TikTok Data"):
            st.subheader("Raw TikTok Data")
            st.dataframe(df, use_container_width=True)

        # ------- LLM Notes and Scoring -------
        st.subheader("📝 LLM Notes & Suitability Scoring")
        st.markdown("Generate DOL/KOL-style vetting notes and a rationalized score with your chosen LLM.")

        default_template = """Summary:
Relevance:
Strengths:
Weaknesses:
Red Flags:
Brand Mentions:
Research Notes:
"""
        note_template = st.text_area("Customize LLM Notes Template", value=default_template, height=150)

        if st.button("Generate LLM Vetting Notes"):
            with st.spinner("Calling LLM to generate notes..."):
                notes_text = generate_llm_notes(
                    filtered_df, note_template, provider=llm_provider,
                    openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                    gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None
                )
            st.markdown("#### LLM Vetting Notes")
            st.markdown(notes_text)
            st.download_button(
                label="Download LLM Vetting Notes",
                data=notes_text,
                file_name="llm_vetting_notes.txt",
                mime="text/plain"
            )
            if st.button("Generate LLM Score & Rationale"):
                with st.spinner("Calling LLM for scoring..."):
                    score_result = generate_llm_score(
                        notes_text, provider=llm_provider,
                        openai_api_key=openai_api_key if llm_provider == "OpenAI GPT" else None,
                        gemini_api_key=gemini_api_key if llm_provider == "Google Gemini" else None
                    )
                st.markdown("#### LLM DOL/KOL Score & Rationale")
                st.code(score_result, language="yaml")

    if threads_enabled and threads_username.strip():
        st.info("Scraping Threads data...")
        raw_threads = run_apify_scraper(apify_api_key, threads_username.strip(), max_results)
        if raw_threads:
            threads_df = process_posts(raw_threads, transcript_map={}, fetch_time=fetch_time, last_fetch_time=last_fetch_time)
            st.subheader("🧵 Meta Threads Results")
            st.dataframe(threads_df, use_container_width=True)
            st.download_button("Download Threads CSV", threads_df.to_csv(index=False),
                              file_name="threads_posts.csv")
        else:
            st.info("No Threads data found or available.")

    if st.session_state["analysis_count"] >= 5:
        st.sidebar.success("🎉 Power User: 5+ Analyses!")
        st.balloons()

    if st.session_state["top_kols"]:
        st.sidebar.markdown("👩‍⚕️ **Top DOL/KOLs analyzed (this session):**")
        for author in st.session_state["top_kols"]:
            st.sidebar.write(f"- {author}")

    feedback = st.radio("Did you find this analysis useful?", ["👍 Yes", "👎 No"], horizontal=True)
    if feedback:
        st.session_state["feedback_logs"].append({
            "time": datetime.now().isoformat(),
            "feedback": feedback,
            "last_query": query,
        })
        st.info(f"Thanks for your feedback: {feedback}")

elif not apify_api_key:
    st.info("🔑 Please provide your Apify API key to begin.")

if st.sidebar.checkbox("Show Analytics (Admin only)"):
    st.subheader("🔎 Usage Analytics")
    st.json({
        "analyses_this_session": st.session_state["analysis_count"],
        "feedback_this_session": st.session_state["feedback_logs"],
        "unique_DOL/KOLs": st.session_state["top_kols"]
    })

# ------------ requirements.txt ----------
# streamlit
# apify-client
# requests
# pandas
# textblob
# nltk
# openai
# google-generativeai



