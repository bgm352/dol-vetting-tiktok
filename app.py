import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import logging

# ============ SETUP LOGGING ============
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# Track missing deps
missing_deps = []

try:
    import nltk
    nltk.download("punkt", quiet=True)
except ImportError:
    missing_deps.append("nltk")

try:
    from textblob import TextBlob
except ImportError:
    missing_deps.append("textblob")

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None
    missing_deps.append("apify-client")

try:
    import openai
except ImportError:
    openai = None
    missing_deps.append("openai")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    missing_deps.append("google-generativeai")

try:
    import plotly.express as px
except ImportError:
    px = None
    missing_deps.append("plotly")

# ============ PAGE CONFIG ============
st.set_page_config("TikTok DOL/KOL Vetting Tool", layout="wide", page_icon="🩺")
st.title("🩺 TikTok DOL/KOL Vetting Tool - Multi-Batch, LLM, Export")

if missing_deps:
    st.warning(f"Missing dependencies: {', '.join(missing_deps)}. "
               f"Install with: pip install {' '.join(missing_deps)}")

# ============ SIDEBAR ============
apify_api_key = st.sidebar.text_input("Apify API Token", type="password")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI GPT", "Google Gemini"])
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password") if llm_provider == "OpenAI GPT" else None
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password") if llm_provider == "Google Gemini" else None

st.sidebar.header("Scrape Controls")
query = st.sidebar.text_input("TikTok Search Term", "doctor")
target_total = st.sidebar.number_input("Total TikTok Videos", min_value=10, value=50, step=10)
batch_size = st.sidebar.number_input("Batch Size per Run", min_value=10, max_value=200, value=25)
run_mode = st.sidebar.radio("Analysis Type", ["Doctor Vetting (DOL/KOL)", "Brand Vetting (Sentiment)"])

# Keywords for classification
ONCOLOGY_TERMS = ["oncology", "cancer", "monoclonal", "checkpoint", "immunotherapy"]
GI_TERMS = ["biliary tract", "gastric", "gea", "gi", "adenocarcinoma"]
RESEARCH_TERMS = ["biomarker", "clinical trial", "abstract", "network", "congress"]
BRAND_TERMS = ["ziihera", "zanidatamab", "brandA", "pd-l1"]

# Example NPI lookup placeholder
NPI_LOOKUP = {
    "drjohnsmith": {"npi": "1234567890", "hospital": "Hope Medical Center"},
    "janedoe": {"npi": "0987654321", "hospital": "City Health System"},
}

# ============ SESSION STATE ============
for k, d in [
    ("last_fetch_time", None),
    ("tiktok_df", pd.DataFrame()),
    ("llm_notes_text", ""),
    ("llm_score_result", "")
]:
    if k not in st.session_state:
        st.session_state[k] = d

# ============ HELPERS ============
def classify_kol_dol(score):
    return "KOL" if score >= 8 else "DOL" if score >= 5 else "Not Suitable"

def classify_sentiment(score):
    return "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"

def enrich_with_npi(author_name):
    key = author_name.lower().replace(" ", "")
    match = NPI_LOOKUP.get(key)
    return {
        "NPI Match": bool(match),
        "NPI": match.get("npi") if match else "",
        "Hospital Affiliation": match.get("hospital") if match else ""
    }

def advanced_kol_score(sent_score, views, likes, comments, has_npi, medical_rel):
    engagement = ((likes + comments) / views) if views else 0
    base = (sent_score * 4) + (engagement * 10)
    if has_npi: base += 2
    if medical_rel: base += 2
    return min(max(round(base), 1), 10)

def generate_rationale(text, transcript, author, score, sentiment, mode):
    all_text = f"{text} {transcript}".lower()
    tags = {
        "onco": any(t in all_text for t in ONCOLOGY_TERMS),
        "gi": any(t in all_text for t in GI_TERMS),
        "res": any(t in all_text for t in RESEARCH_TERMS),
        "brand": any(t in all_text for t in BRAND_TERMS)
    }
    name = author or "This creator"
    rationale = f"{name} score {score}/10"
    if tags["onco"]: rationale += " | Onco content"
    if tags["gi"]: rationale += " | GI"
    if tags["res"]: rationale += " | Research"
    if tags["brand"]: rationale += " | Brand mention"
    return rationale

def get_llm_response(prompt):
    try:
        if llm_provider == "OpenAI GPT" and openai and openai_api_key:
            openai.api_key = openai_api_key
            resp = openai.ChatCompletion.create(
                model="gpt-4o",  # upgraded default
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512, temperature=0.6
            )
            return resp.choices[0].message['content'].strip()
        elif llm_provider == "Google Gemini" and genai and gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')  # upgraded default
            resp = model.generate_content(prompt)
            return getattr(resp, 'text', str(resp)).strip()
        else:
            return "LLM not available or API key missing."
    except Exception as e:
        logging.exception("LLM call error")
        return f"Error generating LLM response: {e}"

def generate_llm_notes(posts_df, note_template):
    posts_texts = "\n\n".join([
        f"{i+1}. Author: {row['Author']}\nContent: {row['Text']}\nTranscript: {row['Transcript']}"
        for i, row in posts_df.iterrows()
    ])
    prompt = f"""Using the following social posts, generate notes for KOL/DOL vetting in this structure:
{note_template}
Social posts:
{posts_texts}
Return in markdown, each section with a title."""
    return get_llm_response(prompt)

def generate_llm_score(notes):
    prompt = f"""You are a medical affairs expert. Based on these vetting notes and evidence, assign a DOL suitability score (1=poor, 10=ideal) for pharma and give a rationale.
Notes: {notes}
Respond in YAML:
score: <1-10>
rationale: <short explanation>
"""
    return get_llm_response(prompt)

def fetch_tiktok_transcripts_apify(api_token, video_urls):
    if not ApifyClient:
        return {}
    try:
        client = ApifyClient(api_token)
        run_input = {"videos": video_urls}
        run = client.actor("scrape-creators/best-tiktok-transcripts-scraper").call(run_input=run_input)
        transcripts = {}
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            transcripts[item.get("id")] = item.get("transcript", "")
        return transcripts
    except Exception as e:
        logging.exception("Transcript fetch error")
        return {}

def run_apify_scraper_batched(api_key, query, target_total, batch_size):
    if not ApifyClient:
        st.error("ApifyClient not installed.")
        return []
    result = []
    try:
        client = ApifyClient(api_key)
        run = client.actor("clockworks/tiktok-scraper").call(run_input={
            "searchQueries": [query],
            "resultsPerPage": batch_size,
            "searchType": "keyword"
        })
        dataset_id = run.get("defaultDatasetId", "")
        for item in client.dataset(dataset_id).iterate_items():
            result.append(item)
            if len(result) >= target_total:
                break
    except Exception as e:
        st.error(f"Apify scrape error: {e}")
        logging.exception("Apify scrape error")
    return result[:target_total]

def process_posts(posts, transcript_map):
    out = []
    for p in posts:
        try:
            author = p.get("authorMeta", {}).get("name", "")
            text = p.get("text", "")
            post_id = p.get("id", "")
            url = f"https://www.tiktok.com/@{author}/video/{post_id}"
            tscript = transcript_map.get(post_id, "(Transcript not found)")
            body = f"{text} {tscript}"
            sent_score = TextBlob(body).sentiment.polarity if "textblob" not in missing_deps else 0
            sentiment = classify_sentiment(sent_score)
            med_rel = any(t in body.lower() for t in ONCOLOGY_TERMS + GI_TERMS + RESEARCH_TERMS + BRAND_TERMS)
            enrich = enrich_with_npi(author)
            dol_score = advanced_kol_score(sent_score, p.get("playCount", 0),
                                           p.get("diggCount", 0), p.get("commentCount", 0),
                                           enrich["NPI Match"], med_rel)
            status = classify_kol_dol(dol_score)
            rationale = generate_rationale(text, tscript, author, dol_score, sentiment, run_mode)
            out.append({
                "Author": author,
                "Text": text,
                "Transcript": tscript,
                "Likes": p.get("diggCount", 0),
                "Views": p.get("playCount", 0),
                "Comments": p.get("commentCount", 0),
                "Shares": p.get("shareCount", 0),
                "DOL Score": dol_score,
                "Sentiment Score": sent_score,
                "KOL/DOL Status": status,
                "Brand Sentiment Label": sentiment,
                "Rationale": rationale,
                "NPI Match": enrich["NPI Match"],
                "NPI": enrich["NPI"],
                "Hospital Affiliation": enrich["Hospital Affiliation"],
                "Post URL": url
            })
        except Exception as e:
            logging.exception("Process post error")
    return pd.DataFrame(out)

# ============ MAIN APP ============
if st.button("Go 🚀") and apify_api_key:
    st.session_state.last_fetch_time = datetime.now()
    tiktok_data = run_apify_scraper_batched(apify_api_key, query, int(target_total), int(batch_size))
    if tiktok_data:
        video_urls = [f"https://www.tiktok.com/@{p.get('authorMeta', {}).get('name','')}/video/{p.get('id','')}" for p in tiktok_data]
        transcripts = fetch_tiktok_transcripts_apify(apify_api_key, video_urls)
        df = process_posts(tiktok_data, transcripts)
        st.session_state.tiktok_df = df
    else:
        st.warning("No posts found.")

df = st.session_state.tiktok_df
if not df.empty:
    st.dataframe(df)

    with st.expander("📊 Stats & Benchmarks"):
        st.write(f"Total creators: {len(df)}")
        st.write(f"KOL %: {100 * df['KOL/DOL Status'].str.contains('KOL').sum() / len(df):.1f}%")
        st.write(f"NPI Matches: {df['NPI Match'].sum()}")
        st.write(f"Estimated Data Cost: ${(len(df) * 0.02):.2f}")

    if px:
        col1, col2 = st.columns(2)
        with col1:
            fig_score = px.histogram(df, x="DOL Score", nbins=10, title="DOL Score Distribution")
            st.plotly_chart(fig_score, use_container_width=True)
        with col2:
            sentiment_counts = df["Brand Sentiment Label"].value_counts().reset_index()
            fig_sent = px.pie(sentiment_counts, names="index", values="Brand Sentiment Label", title="Sentiment Breakdown")
            st.plotly_chart(fig_sent, use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False), "tiktok_analysis.csv")

    # LLM Notes & Scoring
    st.subheader("📝 LLM Notes & Suitability Scoring")
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
        notes_text = generate_llm_notes(df, note_template)
        st.session_state.llm_notes_text = notes_text
    if st.session_state.llm_notes_text:
        st.markdown("#### LLM Vetting Notes")
        st.markdown(st.session_state.llm_notes_text)
        st.download_button("Download LLM Vetting Notes", st.session_state.llm_notes_text, "llm_vetting_notes.txt")
        if st.button("Generate LLM Score & Rationale"):
            score_result = generate_llm_score(st.session_state.llm_notes_text)
            st.session_state.llm_score_result = score_result
    if st.session_state.llm_score_result:
        st.markdown("#### LLM Score & Rationale")
        st.code(st.session_state.llm_score_result, language="yaml")

