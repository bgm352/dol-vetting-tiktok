import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import nltk

# Setup & Package Guards
try:
    from apify_client import ApifyClient
except ModuleNotFoundError:
    st.error("Install `apify-client` (`pip install apify-client`).")
    st.stop()
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
    st.error("Install `textblob` and `nltk`.")
    st.stop()

#### ---- Streamlit UI ---- ####

st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")

# --- Sidebar Params ---
apify_api_key = st.sidebar.text_input("Apify API Token", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password") if llm_provider=="OpenAI GPT" else None
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password") if llm_provider=="Google Gemini" else None

st.sidebar.header("Scrape Controls")
query = st.sidebar.text_input("TikTok Search Term", "doctor")
target_total = st.sidebar.number_input("Total TikTok Videos", min_value=10, value=200, step=10, help="How many videos to collect (max possible, not guaranteed).")
batch_size = st.sidebar.number_input("Batch Size per Run", min_value=10, max_value=200, value=50, help="How many posts per Apify job (don't set too high).")
run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

if "top_kols" not in st.session_state: st.session_state["top_kols"] = []
if "analysis_count" not in st.session_state: st.session_state["analysis_count"] = 0
if "feedback_logs" not in st.session_state: st.session_state["feedback_logs"] = []
if "last_fetch_time" not in st.session_state: st.session_state["last_fetch_time"] = None

# --- Helper Functions ---

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
        if tags["onco"]: rationale += " frequently engaging in oncology content"
        if tags["gi"]: rationale += ", particularly in GI-focused diseases"
        if tags["res"]: rationale += " and demonstrating strong research credibility"
        if tags["brand"]: rationale += ", mentioning monoclonal therapies or campaign drugs specifically"
        if transcript and "not found" not in transcript:
            rationale += f'. Transcript: “{transcript[:90].strip()}...”'
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

def get_llm_response(prompt, provider, openai_api_key=None, gemini_api_key=None):
    if provider == "OpenAI GPT":
        if not openai: return "OpenAI SDK not installed."
        if not openai_api_key: return "No OpenAI key."
        openai.api_key = openai_api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.6)
        return resp.choices[0].message['content'].strip()
    elif provider == "Google Gemini":
        if not genai: return "Gemini SDK not installed."
        if not gemini_api_key: return "No Gemini key."
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        resp = model.generate_content(prompt)
        return getattr(resp, 'text', str(resp)).strip()
    else:
        return "Unknown provider"

def generate_llm_notes(posts_df, note_template, provider, openai_api_key=None, gemini_api_key=None):
    posts_texts = "\n\n".join([
        f"{i+1}. Author: {row['Author']}\nContent: {row['Text']}\nTranscript: {row['Transcript']}"
        for i, row in posts_df.iterrows()
    ])
    prompt = f"""Using the following social posts, generate notes for KOL/DOL vetting in this structure:
{note_template}
Social posts:
{posts_texts}
Return in markdown, each section with a title."""
    return get_llm_response(prompt, provider, openai_api_key, gemini_api_key)

def generate_llm_score(notes, provider, openai_api_key=None, gemini_api_key=None):
    prompt = f"""You are a medical affairs expert. Based on these vetting notes and evidence, assign a DOL suitability score (1=poor, 10=ideal) for pharma and give a rationale.
Notes: {notes}
Respond in YAML:
score: <1-10>
rationale: <short explanation>
"""
    return get_llm_response(prompt, provider, openai_api_key, gemini_api_key)

def fetch_tiktok_transcripts_apify(api_token, video_urls):
    client = ApifyClient(api_token)
    run_input = {"videos": video_urls}
    run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input=run_input)
    transcripts = {}
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        video_id = item.get("id")
        transcript = item.get("transcript", "")
        transcripts[video_id] = transcript
    return transcripts

@st.cache_data(show_spinner=False, persist="disk")   # for scale
def run_apify_scraper_batched(api_key, query, target_total, batch_size):
    '''Run Apify TikTok Scraper in batches to collect up to target_total posts'''
    result = []
    run_url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    offset = 0
    failures = 0
    pbar = st.progress(0.0, text="Scraping TikTok...")
    try:
        while len(result) < target_total and failures < 5:
            st.info(f"Launching batch {1+offset//batch_size}: {len(result)} of {target_total}")
            start = requests.post(
                run_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"searchQueries":[query], "resultsPerPage":batch_size, "searchType":"keyword", "pageNumber":offset//batch_size}
            ).json()
            run_id = start.get("data", {}).get("id")
            for i in range(60):
                resp = requests.get(f"{run_url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}).json()
                if resp.get("data", {}).get("status") == "SUCCEEDED":
                    dataset_id = resp["data"].get("defaultDatasetId")
                    break
                time.sleep(5)
            else:
                failures += 1
                st.error("Batch run timed out, retrying.")
                continue
            data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
            batch_posts = requests.get(data_url).json()
            if not batch_posts:
                failures += 1
                st.error("No posts after batch, retrying.")
                continue
            for p in batch_posts:
                if p not in result:
                    result.append(p)
            offset += batch_size
            pbar.progress(min(1.0, len(result) / float(target_total)))
            if len(batch_posts) < batch_size: break  # exhausted
    except Exception as e:
        st.error(f"Apify error: {e}")
        return []
    pbar.progress(1.0)
    return result[:target_total]

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
            rationale = generate_rationale(text, tscript, author, dol_score, sentiment, run_mode)
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
            st.warning(f"⛔ Skipped 1 post: {e}")
            continue
    return pd.DataFrame(results)

# ----------- MAIN APP FLOW -----------
notes_text = ""
run_analysis = st.button("Go 🚀", use_container_width=True)

if run_analysis and apify_api_key:
    st.session_state["analysis_count"] += 1
    fetch_time = datetime.now()
    last_fetch_time = st.session_state.get("last_fetch_time", None)
    tiktok_data = run_apify_scraper_batched(apify_api_key, query, int(target_total), int(batch_size))
    if not tiktok_data:
        st.warning("No TikTok posts found.")
    else:
        video_urls = [f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}' for p in tiktok_data]
        transcript_map = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
        st.success(f"✅ {len(tiktok_data)} TikTok posts scraped.")
        df = process_posts(tiktok_data, transcript_map=transcript_map, fetch_time=fetch_time, last_fetch_time=last_fetch_time)
        st.session_state["last_fetch_time"] = fetch_time
        st.metric("TikTok Posts", len(df))

        if not df.empty:
            kols = df[df["KOL/DOL Status"].str.contains("KOL|DOL", regex=True)]["Author"].unique().tolist()
            st.session_state["top_kols"].extend(x for x in kols if x not in st.session_state["top_kols"])

        st.subheader("📋 TikTok Analysis Results")
        tiktok_cols = [
            "Author", "Text", "Transcript", "Likes", "Views", "Comments", "Shares",
            "DOL Score", "Sentiment Score", "Post URL", "KOL/DOL Status",
            "Brand Sentiment Label", "LLM DOL Score Rationale", "Timestamp", "Data Fetched At", "Is New"
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
        st.markdown("**🟢 New** = New since last run; **Old** = Previously collected.")

        st.download_button("Download TikTok CSV", filtered_df[columns].to_csv(index=False),
                          file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

        if st.checkbox("Show Raw TikTok Data"):
            st.subheader("Raw TikTok Data")
            st.dataframe(df, use_container_width=True)

        # ------- LLM Notes and Scoring -------
        st.subheader("📝 LLM Notes & Suitability Scoring")
        st.markdown("Generate DOL/KOL-style vetting notes and score.")

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
                notes_text = generate_llm_notes(filtered_df, note_template, provider=llm_provider,
                                                openai_api_key=openai_api_key if llm_provider=="OpenAI GPT" else None,
                                                gemini_api_key=gemini_api_key if llm_provider=="Google Gemini" else None)
            st.markdown("#### LLM Vetting Notes")
            st.markdown(notes_text)
            st.download_button(
                label="Download LLM Vetting Notes",
                data=notes_text,
                file_name="llm_vetting_notes.txt",
                mime="text/plain")
            if st.button("Generate LLM Score & Rationale"):
                with st.spinner("Calling LLM for scoring..."):
                    score_result = generate_llm_score(notes_text, provider=llm_provider,
                                                      openai_api_key=openai_api_key if llm_provider=="OpenAI GPT" else None,
                                                      gemini_api_key=gemini_api_key if llm_provider=="Google Gemini" else None)
                st.markdown("#### LLM DOL/KOL Score & Rationale")
                st.code(score_result, language="yaml")

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

# ------------ requirements.txt -------
# streamlit
# apify-client
# requests
# pandas
# textblob
# nltk
# openai
# google-generativeai

