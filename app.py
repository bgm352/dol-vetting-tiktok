# streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import nltk

# ---- Module Imports (modularize as your app grows) ----
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
    st.error("Please install dependencies: `pip install textblob nltk`.")
    st.stop()

# ---- Caching/State Best Practices ----
@st.cache_data(show_spinner=False, persist="disk")  # For large/batched TikTok fetches
def fetch_tiktok_data(api_key, query, target_total, batch_size):
    result, offset, failures = [], 0, 0
    url = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs"
    while len(result) < target_total and failures < 3:
        st.info(f"Batch {1 + offset // batch_size}: {len(result)}/{target_total}")
        run_resp = requests.post(
            url, headers={"Authorization": f"Bearer {api_key}"},
            json={"searchQueries": [query], "resultsPerPage": batch_size, "searchType": "keyword", "pageNumber": offset // batch_size}
        ).json()
        run_id = run_resp.get("data", {}).get("id")
        for _ in range(60):
            r = requests.get(f"{url}/{run_id}", headers={"Authorization": f"Bearer {api_key}"}).json()
            if r.get("data", {}).get("status") == "SUCCEEDED":
                dataset_id = r["data"].get("defaultDatasetId")
                break
            time.sleep(4)
        else:
            failures += 1
            continue
        posts = requests.get(f"https://api.apify.com/v2/datasets/{dataset_id}/items").json()
        for p in posts:
            if p not in result:
                result.append(p)
        offset += batch_size
        if len(posts) < batch_size: break
    return result[:target_total]

@st.cache_data(show_spinner="Loading transcripts...", persist="disk")
def fetch_transcripts(api_token, video_urls):
    client = ApifyClient(api_token)
    run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input={"videos": video_urls})
    transcripts = {}
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        transcripts[item.get("id")] = item.get("transcript", "")
    return transcripts

# ---- Utility Functions ----
def classify_kol_dol(score): return "KOL" if score >= 8 else "DOL" if score >= 5 else "Not Suitable"
def classify_sentiment(score): return "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"

def rationale(text, transcript, author, score, sentiment, mode):
    ONCO = ["oncology", "cancer", "checkpoint", "immunotherapy"]
    GI = ["biliary", "gastric", "gi", "adenocarcinoma"]
    RES = ["biomarker", "clinical trial", "congress"]
    BRAND = ["ziihera", "zanidatamab", "brandA", "pd-l1"]
    all_text = f"{text or ''} {transcript or ''}".lower()
    tags = {
        "onco": any(t in all_text for t in ONCO), "gi": any(t in all_text for t in GI),
        "res": any(t in all_text for t in RES), "brand": any(t in all_text for t in BRAND)
    }
    name = author or "Creator"
    base = f"{name} is highly influential," if score >= 8 else (
        f"{name} has moderate relevance," if score >= 5 else f"{name} does not actively discuss campaign topics,"
    )
    taglines = []
    if tags["onco"]: taglines.append("frequent oncology content")
    if tags["gi"]: taglines.append("GI diseases")
    if tags["res"]: taglines.append("research credibility")
    if tags["brand"]: taglines.append("mentions monoclonal therapies/drugs")
    snip = f'“{(transcript or "")[:80]}...”' if transcript and "not found" not in transcript else f". {transcript or ''}"
    return f"{base} {', '.join(taglines)}. {snip} (Score: {score}/10)" if "Doctor" in mode else f"{name} expresses {sentiment.lower()} brand sentiment. {snip}"

def get_llm_response(prompt, provider, openai_key=None, gemini_key=None):
    if provider == "OpenAI GPT":
        openai.api_key = openai_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=512, temperature=0.6)
        return resp.choices[0].message['content'].strip()
    elif provider == "Google Gemini":
        genai.configure(api_key=gemini_key)
        m = genai.GenerativeModel('gemini-pro')
        r = m.generate_content(prompt)
        return getattr(r, "text", str(r)).strip()
    return "Unknown LLM provider"

def prepare_llm_prompt(df, template):
    posts_texts = "\n\n".join([f"{i+1}. {row['Author']}: {row['Text']} \nTranscript: {row['Transcript']}" for i, row in df.iterrows()])
    return f"{template}\nSocial posts:\n{posts_texts}\nReturn results in markdown, sections titled."

def vetting_notes(df, template, provider, openai_key, gemini_key):
    prompt = prepare_llm_prompt(df, template)
    return get_llm_response(prompt, provider, openai_key, gemini_key)

def vetting_score(notes, provider, openai_key, gemini_key):
    prompt = f"Based on these DOL vetting notes, assign a DOL suitability score for pharma campaigns (1=poor, 10=ideal) and explain shortly in YAML.\n\nNotes: {notes}\nFormat:\nscore: <1-10>\nrationale: <short>"
    return get_llm_response(prompt, provider, openai_key, gemini_key)

def process_posts(posts, transcript_map, fetch_time, last_fetch_time, mode):
    results = []
    for post in posts:
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
            rat = rationale(text, tscript, author, dol_score, sentiment, mode)
            is_new = "🟢 New" if last_fetch_time is None or ts > last_fetch_time else "Old"
            results.append({
                "Author": author, "Text": text.strip(), "Transcript": tscript or "Transcript not found",
                "Likes": post.get("diggCount", 0), "Views": post.get("playCount", 0), "Comments": post.get("commentCount", 0),
                "Shares": post.get("shareCount", 0), "Timestamp": ts, "Post URL": url, "DOL Score": dol_score,
                "Sentiment Score": sentiment_score, "KOL/DOL Status": f"{'🌟' if kol_dol_label == 'KOL' else '👍' if kol_dol_label == 'DOL' else '❌'} {kol_dol_label}",
                "Brand Sentiment Label": sentiment, "LLM DOL Score Rationale": rat, "Data Fetched At": fetch_time, "Is New": is_new
            })
        except Exception as e:
            st.warning(f"⛔ Skipped 1 post: {e}")
    return pd.DataFrame(results)

# ---- Streamlit App Logic (proper state!) ----
st.title("🩺 TikTok DOL/KOL Vetting Tool (Scalable, Robust)")
with st.form(key="scrape_form"):
    go = st.form_submit_button("Go 🚀")
if go and apify_api_key:
    fetch_time = datetime.now()
    last_fetch_time = st.session_state.get("last_fetch_time", None)
    tiktok_data = fetch_tiktok_data(apify_api_key, query, int(target_total), int(batch_size))
    if not tiktok_data:
        st.warning("No TikTok posts found.")
    else:
        video_urls = [f'https://www.tiktok.com/@{p.get("authorMeta", {}).get("name","")}/video/{p.get("id","")}' for p in tiktok_data]
        transcript_map = fetch_transcripts(apify_api_key, video_urls)
        st.success(f"✅ {len(tiktok_data)} TikTok posts scraped.")
        df = process_posts(tiktok_data, transcript_map, fetch_time, last_fetch_time, run_mode)
        st.session_state["last_fetch_time"] = fetch_time
        st.session_state["tiktok_df"] = df

df = st.session_state.get("tiktok_df", pd.DataFrame())
if not df.empty:
    st.metric("TikTok Posts", len(df))
    st.subheader("📋 TikTok Analysis Results")
    columns = [col for col in df.columns]
    display_option = st.radio("Display columns:", ["All columns", "Only main info", "Just DOL / Sentiment"])
    if display_option == "Only main info":
        columns = ["Author", "Text", "Likes", "Views", "Comments", "Shares", "DOL Score", "Timestamp", "Is New"]
    elif display_option == "Just DOL / Sentiment":
        columns = ["Author", "KOL/DOL Status", "DOL Score", "Sentiment Score", "Brand Sentiment Label", "Is New"]
    dol_min, dol_max = st.slider("DOL Score Range", 1, 10, (1, 10))
    filtered_df = df[(df["DOL Score"] >= dol_min) & (df["DOL Score"] <= dol_max)]
    st.dataframe(filtered_df[columns], use_container_width=True)
    st.download_button("Download TikTok CSV", filtered_df[columns].to_csv(index=False), file_name=f"tiktok_analysis_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

    if st.checkbox("Show Raw TikTok Data"):
        st.dataframe(df, use_container_width=True)

    st.subheader("📝 LLM Notes & Suitability Scoring")
    default_template = """Summary:
Relevance:
Strengths:
Weaknesses:
Red Flags:
Brand Mentions:
Research Notes:
"""
    note_template = st.text_area("LLM Notes Template", value=default_template, height=150, key="notes_template")
    if st.button("Generate LLM Vetting Notes"):
        with st.spinner("Calling LLM to generate notes..."):
            notes_text = vetting_notes(
                filtered_df, note_template, llm_provider, openai_api_key, gemini_api_key)
        st.session_state["notes_text"] = notes_text
        st.session_state["llm_score_result"] = ""  # Reset old score

    if st.session_state.get("notes_text", ""):
        st.markdown("#### LLM Vetting Notes")
        st.markdown(st.session_state["notes_text"])
        st.download_button(
            label="Download Notes",
            data=st.session_state["notes_text"],
            file_name="llm_vetting_notes.txt",
            mime="text/plain"
        )
        if st.button("Generate LLM Score & Rationale"):
            with st.spinner("Scoring..."):
                score_result = vetting_score(
                    st.session_state["notes_text"], llm_provider, openai_api_key, gemini_api_key)
            st.session_state["llm_score_result"] = score_result

    if st.session_state.get("llm_score_result", ""):
        st.markdown("#### LLM DOL/KOL Score & Rationale")
        st.code(st.session_state["llm_score_result"], language="yaml")

    st.sidebar.markdown("👩‍⚕️ **Top DOL/KOLs (this session):**")
    if not df.empty:
        kols = df[df["KOL/DOL Status"].str.contains("KOL|DOL", regex=True)]["Author"].unique().tolist()
        for author in kols: st.sidebar.write(f"- {author}")

    st.sidebar.write("Analyses this session:", st.session_state.get("analysis_count", 0))

if st.sidebar.checkbox("Show Analytics"):
    st.subheader("🔎 Usage Analytics")
    st.json({
        "analyses_this_session": st.session_state.get("analysis_count", 0),
        "feedback_this_session": st.session_state.get("feedback_logs", []),
    })

# -------- requirements.txt ----------
# streamlit
# apify-client
# requests
# pandas
# textblob
# nltk
# openai
# google-generativeai





