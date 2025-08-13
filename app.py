import os
import streamlit as st
import pandas as pd
from datetime import datetime
import nltk
import random
import time
import logging
from textblob import TextBlob
from typing import List, Dict, Any
import altair as alt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download("punkt", quiet=True)

try:
    from apify_client import ApifyClient
    from apify_client._errors import ApifyApiError
except ModuleNotFoundError:
    st.error("Install `apify-client`: pip install apify-client")
    st.stop()

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Keywords ---
ONCOLOGY_TERMS=["oncology","cancer","monoclonal","checkpoint","immunotherapy"]
GI_TERMS=["biliary tract","gastric","gea","gi","adenocarcinoma"]
RESEARCH_TERMS=["biomarker","clinical trial","abstract","network","congress"]
BRAND_TERMS=["ziihera","zanidatamab","brandA","pd-l1"]

def keyword_hits(text: str) -> Dict[str,bool]:
    t=(text or "").lower()
    return {
        "onco": any(k in t for k in ONCOLOGY_TERMS),
        "gi": any(k in t for k in GI_TERMS),
        "res": any(k in t for k in RESEARCH_TERMS),
        "brand": any(k in t for k in BRAND_TERMS)
    }

def classify_sentiment(score: float) -> str:
    if score > 0.15: return "Positive"
    elif score < -0.15: return "Negative"
    return "Neutral"

def classify_kol_dol(score: float) -> str:
    if score >= 8: return "KOL"
    elif score >= 5: return "DOL"
    return "Not Suitable"

def retry_with_backoff(func=None, *, max_retries=3, base_delay=2):
    def decorator(f):
        def wrapper(*args,**kwargs):
            attempt,last_exception=0,None
            while attempt<max_retries:
                try: return f(*args,**kwargs)
                except Exception as e:
                    last_exception=e; attempt+=1
                    delay = base_delay ** attempt + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt}/{max_retries} after error: {e}, retrying in {delay:.1f}s")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator(func) if func else decorator

# --- Clients ---
@st.cache_resource
def get_apify_client(api_token:str) -> ApifyClient:
    return ApifyClient(api_token)

@st.cache_resource
def get_openai_client(api_key:str):
    if not openai: raise RuntimeError("OpenAI SDK not installed.")
    return openai.OpenAI(api_key=api_key)

@retry_with_backoff
def call_openai(prompt:str, api_key:str, model:str, temperature:float, max_tokens:int) -> str:
    client=get_openai_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()

@retry_with_backoff
def call_gemini(prompt:str, api_key:str, model:str, temperature:float, max_tokens:int) -> str:
    if not genai: raise RuntimeError("Gemini SDK not installed.")
    genai.configure(api_key=api_key)
    m=genai.GenerativeModel(model)
    resp=m.generate_content(prompt=prompt,temperature=temperature,
                            max_output_tokens=max_tokens if max_tokens>0 else None)
    return getattr(resp,"text",str(resp)).strip()

# --- Robust scraper with retry ---
def run_threads_scraper(api_key: str, usernames: List[str], max_posts: int = 50) -> List[Dict[str, Any]]:
    if not api_key or not api_key.strip():
        st.error("Apify API Token is required.")
        return []
    if not usernames:
        st.error("At least one username is required.")
        return []

    urls = [f"https://www.threads.net/@{u.strip().lstrip('@')}" for u in usernames if u.strip()]
    if not urls:
        st.error("No valid usernames provided.")
        return []

    run_input = {"urls": urls, "maxPosts": max_posts, "includeComments": False}
    client = get_apify_client(api_key)

    max_retries = 3
    backoff_base = 2
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            run = client.actor("red.cars/threads-scraper").call(run_input=run_input)
            break
        except ApifyApiError as ae:
            last_exception = ae
            logger.error(f"[Attempt {attempt}] Apify API error: {ae}")
            if attempt < max_retries:
                delay = backoff_base ** attempt + random.uniform(0, 1)
                st.warning(f"Retrying Apify actor call (attempt {attempt}/{max_retries}) in {delay:.1f}s...")
                time.sleep(delay)
            else:
                st.error("Apify API error after multiple retries. Check token, input, or quota.")
                return []
        except Exception as e:
            last_exception = e
            logger.error(f"[Attempt {attempt}] Error calling Apify: {e}")
            if attempt < max_retries:
                delay = backoff_base ** attempt + random.uniform(0, 1)
                st.warning(f"Retrying Apify actor call (attempt {attempt}/{max_retries}) in {delay:.1f}s...")
                time.sleep(delay)
            else:
                st.error("Failed to contact Apify actor after multiple retries.")
                return []
    else:
        return []

    dsid = run.get("defaultDatasetId")
    if not dsid:
        st.error("No dataset returned from actor run.")
        return []

    try:
        return list(client.dataset(dsid).iterate_items())
    except Exception as e:
        logger.error(f"Dataset fetch failed: {e}")
        st.error(f"Failed to fetch dataset: {e}")
        return []

# --- Processing ---
def process_profiles_and_posts(data: List[Dict[str,Any]]):
    profiles, posts_list = {}, []
    for item in data:
        if item.get("type")=="profile":
            u=item.get("username","")
            profiles[u]={"Username":u,"Full Name":item.get("displayName",""),
                         "Bio":item.get("bio",""),
                         "Follower Count":item.get("followers",0),
                         "Profile URL":f"https://www.threads.net/@{u}",
                         "Is Verified":item.get("isVerified",False),
                         "Relevant Posts":0,
                         "Total Engagement (Relevant)":0,
                         "Avg Engagement (Relevant)":0,
                         "Keyword Hits Bio": keyword_hits(item.get("bio",""))}
    for item in data:
        if item.get("type")=="post":
            uname=item.get("ownerUsername","")
            txt=item.get("caption","")
            s_score=TextBlob(txt).sentiment.polarity if txt else 0.0
            s_label=classify_sentiment(s_score)
            tags=keyword_hits(txt)
            engagement=(item.get("likesCount",0)or 0)+(item.get("commentsCount",0)or 0)
            posts_list.append({"Username":uname,"Post Text":txt,"Likes":item.get("likesCount",0),
                               "Comments":item.get("commentsCount",0),"Engagement":engagement,
                               "Sentiment":s_label,
                               "Keywords":", ".join([k for k,v in tags.items() if v])})
            if any(tags.values()) and uname in profiles:
                profiles[uname]["Relevant Posts"]+=1
                profiles[uname]["Total Engagement (Relevant)"]+=engagement
    profile_rows=[]
    for uname,p in profiles.items():
        rel=p["Relevant Posts"]
        avg_eng=(p["Total Engagement (Relevant)"]/rel) if rel else 0
        p["Avg Engagement (Relevant)"]=round(avg_eng,1)
        bio_score=TextBlob(p["Bio"]).sentiment.polarity if p["Bio"] else 0.0
        score=(bio_score*10)+min(rel,5)+min(avg_eng/50,3)
        score=max(min(round(score),10),1)
        p["DOL Score"]=score
        p["Sentiment"]=classify_sentiment(bio_score)
        p["KOL/DOL"]=classify_kol_dol(score)
        p["Rationale"]=f"Bio sentiment scaled: {round(bio_score*10,1)}; Relevant posts: {min(rel,5)}; Avg engagement scaled: {round(min(avg_eng/50,3),2)}"
        profile_rows.append(p)
    return pd.DataFrame(profile_rows), pd.DataFrame(posts_list)

# --- Streamlit UI ---
st.sidebar.header("Scraper Settings")
api_key=st.sidebar.text_input("Apify API Token", type="password", value=os.getenv("APIFY_API_TOKEN",""))
usernames_str=st.sidebar.text_area("Threads Usernames (comma separated)", "elonmusk")
usernames=[u.strip().lstrip("@") for u in usernames_str.split(",") if u.strip()]
max_posts=st.sidebar.slider("Max posts per profile",1,200,50)
st.sidebar.header("Top Target Thresholds")
high_score_threshold=st.sidebar.slider("High Score Threshold",5,10,8)
high_engagement_threshold=st.sidebar.number_input("High Engagement Threshold",0,500,50)

st.sidebar.header("LLM Vetting Settings")
default_prompt=("Summary:\nRelevance:\nStrengths:\nWeaknesses:\nRed Flags:\nBrand Mentions:\nResearch Notes:\n")
prompt_template=st.sidebar.text_area("LLM Prompt Template", value=default_prompt, height=180)
llm_provider=st.sidebar.selectbox("Provider", ["OpenAI GPT","Google Gemini"])
temperature=st.sidebar.slider("Temperature",0.0,2.0,0.6)
max_tokens=st.sidebar.number_input("Max Tokens",0,4096,512)
openai_api_key=gemini_api_key=None
openai_model=gemini_model=None
if llm_provider=="OpenAI GPT":
    openai_api_key=st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY",""))
    openai_model=st.sidebar.selectbox("OpenAI Model", ["gpt-4","gpt-3.5-turbo"])
else:
    gemini_api_key=st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY",""))
    gemini_model=st.sidebar.selectbox("Gemini Model", ["gemini-2.5-pro","gemini-2.5-flash"])

if "last_scrape_time" not in st.session_state:
    st.session_state.last_scrape_time=None
st.sidebar.markdown(f"**Last Scrape:** {st.session_state.last_scrape_time or 'Never'}")

if st.sidebar.button("Scrape & Analyze 🚀"):
    data=run_threads_scraper(api_key, usernames, max_posts=max_posts)
    if data:
        profiles_df, posts_df=process_profiles_and_posts(data)
        st.session_state.profiles_df=profiles_df
        st.session_state.posts_df=posts_df
        st.session_state.last_scrape_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Profiles Section ---
if "profiles_df" in st.session_state and not st.session_state.profiles_df.empty:
    profiles_df=st.session_state.profiles_df
    st.subheader("Profiles Overview & Charts")
    df_plot = profiles_df.dropna(subset=['DOL Score', 'KOL/DOL']).copy()
    df_plot['DOL Score'] = df_plot['DOL Score'].astype(str)
    c1,c2=st.columns(2)
    with c1:
        score_chart=alt.Chart(df_plot).mark_bar().encode(
            x=alt.X('DOL Score:N'), y=alt.Y('count()'), color=alt.Color('KOL/DOL:N')).properties(title="KOL/DOL Score Distribution")
        st.altair_chart(score_chart, use_container_width=True)
    with c2:
        sentiment_data=profiles_df.groupby("Sentiment").size().reset_index(name="count")
        pie=alt.Chart(sentiment_data).mark_arc().encode(theta="count:Q", color="Sentiment:N", tooltip=["Sentiment","count"])
        st.altair_chart(pie, use_container_width=True)

    profiles_df["Top Target"]= (profiles_df["DOL Score"]>=high_score_threshold) & (profiles_df["Avg Engagement (Relevant)"]>=high_engagement_threshold)
    scatter = alt.Chart(profiles_df).mark_circle(size=80).encode(
        x='Avg Engagement (Relevant):Q', y='DOL Score:Q', color='Top Target:N',
        tooltip=['Username','DOL Score','Avg Engagement (Relevant)','Relevant Posts','Follower Count']).interactive()
    st.altair_chart(scatter, use_container_width=True)

    top_targets_df = profiles_df[profiles_df["Top Target"]]
    st.markdown(f"**Top Targets (Score ≥ {high_score_threshold}, Engagement ≥ {high_engagement_threshold})**")
    st.dataframe(top_targets_df)
    st.download_button("Download Top Targets CSV", top_targets_df.to_csv(index=False), "top_targets.csv","text/csv")

    f_status=st.multiselect("Filter by KOL/DOL",["KOL","DOL","Not Suitable"],["KOL","DOL","Not Suitable"])
    f_sent=st.multiselect("Filter by Sentiment",["Positive","Neutral","Negative"],["Positive","Neutral","Negative"])
    f_df=profiles_df[(profiles_df["KOL/DOL"].isin(f_status))&(profiles_df["Sentiment"].isin(f_sent))]
    st.dataframe(f_df)
    st.download_button("Download Profiles CSV", f_df.to_csv(index=False), "profiles.csv","text/csv")

    if st.button("Generate Vetting Notes with LLM"):
        data_text=f_df.to_string(index=False)
        prompt=prompt_template+"\n\nProfiles Data:\n"+data_text
        with st.spinner("Calling LLM..."):
            if llm_provider=="OpenAI GPT":
                notes=call_openai(prompt, openai_api_key, openai_model, temperature, max_tokens)
            else:
                notes=call_gemini(prompt, gemini_api_key, gemini_model, temperature, max_tokens)
        st.session_state.llm_notes=notes

    if "llm_notes" in st.session_state:
        st.subheader("LLM Vetting Notes")
        st.markdown(st.session_state.llm_notes)
        st.download_button("Download Vetting Notes", st.session_state.llm_notes, "vetting_notes.txt","text/plain")

# --- Posts Section ---
if "posts_df" in st.session_state and not st.session_state.posts_df.empty:
    posts_df=st.session_state.posts_df
    st.subheader("Posts Overview & Chart")
    all_keywords=[]
    for kws in posts_df["Keywords"]:
        if kws: all_keywords.extend([k.strip() for k in kws.split(",") if k.strip()])
    kw_freq=pd.Series(all_keywords).value_counts().reset_index()
    kw_freq.columns=["Keyword","Count"]
    if not kw_freq.empty:
        kw_chart=alt.Chart(kw_freq).mark_bar().encode(x='Keyword:N',y='Count:Q',tooltip=['Keyword','Count'])
        st.altair_chart(kw_chart, use_container_width=True)
    kw_filter=st.multiselect("Filter by Keywords", sorted(kw_freq["Keyword"]), sorted(kw_freq["Keyword"]))
    f_sent_posts=st.multiselect("Filter by Post Sentiment",["Positive","Neutral","Negative"],["Positive","Neutral","Negative"])
    fp_df=posts_df[(posts_df["Sentiment"].isin(f_sent_posts))&(posts_df["Keywords"].apply(lambda x:any(k in x for k in kw_filter)))]
    st.dataframe(fp_df)
    st.download_button("Download Posts CSV", fp_df.to_csv(index=False), "posts.csv","text/csv")
