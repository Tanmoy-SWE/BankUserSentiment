#app.py
import streamlit as st
import pandas as pd
import os
import glob
from src.data_processor import DataProcessor
from src.insights_generator import InsightsGenerator
from src.visualizations import *
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Prime Bank Analytics Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Helper function to identify text column ---
def find_text_column(df):
    if df.empty:
        return None
    text_columns = [
        'text', 'Text', 'content', 'Content', 'message', 'Message',
        'review', 'Review', 'comment', 'Comment', 'post', 'Post',
        'review_text', 'Review Text', 'post_text', 'Post Text',
        'comment_text', 'Comment Text', 'description', 'Description'
    ]
    for col in text_columns:
        if col in df.columns:
            return col
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'content', 'review', 'comment', 'post']):
            return col
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head()
            if not sample.empty:
                try:
                    if sample.astype(str).str.len().mean() > 20:
                        return col
                except:
                    continue
    return None

# --- Caching for Performance ---
# @st.cache_data  # Temporarily disabled to fix API key issue
def load_and_process_data():
    DATA_DIR = 'data/uploads'
    PERFECTED_DATA_DIR = 'perfected_data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PERFECTED_DATA_DIR):
        os.makedirs(PERFECTED_DATA_DIR)

    all_files = glob.glob(os.path.join(DATA_DIR, '*'))
    # --- CHANGED: Look for a CSV file for perfected data to preserve links and other metadata ---
    perfected_data_file = os.path.join(PERFECTED_DATA_DIR, 'all_posts_with_comments.csv')

    if not all_files and not os.path.exists(perfected_data_file):
        return pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()

    post_files = [f for f in all_files if 'post' in os.path.basename(f).lower() and f.endswith('.csv')]
    comment_files = [f for f in all_files if 'comment' in os.path.basename(f).lower() and f.endswith('.csv')]
    txt_files = [f for f in all_files if f.endswith('.txt')]
    other_csv_files = [f for f in all_files if f.endswith('.csv') and f not in post_files and f not in comment_files]

    def read_files(files_list, file_type):
        dfs = []
        for f in files_list:
            try:
                if f.endswith('.csv'):
                    df = pd.read_csv(f, on_bad_lines='skip')
                else:  # txt
                    with open(f, 'r', encoding='utf-8') as file:
                        df = pd.DataFrame({'text': [p.strip() for p in file.read().split('\n') if p.strip()]})
                
                text_col = find_text_column(df)
                if not text_col:
                    continue
                if text_col != 'text':
                    df = df.rename(columns={text_col: 'text'})
                
                df['source_file'] = os.path.basename(f)
                df['file_type'] = file_type
                df = df[df['text'].notna() & (df['text'].str.strip() != '')]
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                st.error(f"Error reading {os.path.basename(f)}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Load regular data for general insights
    raw_posts_df = read_files(post_files + other_csv_files, 'post')
    raw_comments_df = read_files(comment_files + txt_files, 'comment')

    # --- MODIFIED: Load perfected data from CSV for AI recommendations & Action Items ---
    perfected_df = pd.DataFrame()
    if os.path.exists(perfected_data_file):
        try:
            perfected_df = pd.read_csv(perfected_data_file, on_bad_lines='skip')
            perfected_df['source_file'] = os.path.basename(perfected_data_file)
            perfected_df['file_type'] = 'perfected'
            # Ensure text column is standardized
            text_col = find_text_column(perfected_df)
            if text_col and text_col != 'text':
                perfected_df = perfected_df.rename(columns={text_col: 'text'})
            perfected_df = perfected_df[perfected_df['text'].notna() & (perfected_df['text'].str.strip() != '')]
        except Exception as e:
            st.error(f"Error reading perfected data file '{perfected_data_file}': {e}")

    # --- Pass the API key from your .env file to the processors ---
    openai_key = os.getenv("OPENAI_API_KEY")
    processor = DataProcessor(openai_api_key=openai_key)
    insight_gen = InsightsGenerator(openai_api_key=openai_key)

    # Process all data sets
    processed_posts_df = processor.process_all_data(raw_posts_df)
    processed_comments_df = processor.process_all_data(raw_comments_df)
    processed_perfected_df = processor.process_all_data(perfected_df) if not perfected_df.empty else pd.DataFrame()

    # Create combined dataframe for general dashboard (excluding perfected data)
    all_text_df = pd.concat([processed_posts_df, processed_comments_df], ignore_index=True)
    if all_text_df.empty and processed_perfected_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()

    # Generate general insights from regular data
    insights = insight_gen.generate_all_insights(posts_df=processed_posts_df, all_text_df=all_text_df)

    # --- MODIFIED: Generate AI Recommendations ONLY from perfected data ---
    insights['ai_recommendations'] = {}
    if not processed_perfected_df.empty and 'prime_mentions' in processed_perfected_df.columns:
        prime_perfected_df = processed_perfected_df[processed_perfected_df['prime_mentions'] > 0].copy()
        if not prime_perfected_df.empty:
            insights['ai_recommendations'] = insight_gen.generate_ai_recommendations(prime_perfected_df)
        
    return processed_posts_df, all_text_df, insights, processed_perfected_df

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics")

# Manual API key input if not found in environment
if not os.getenv("OPENAI_API_KEY"):
    api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password", help="Key not found in environment")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- MODIFIED: Renamed variable for clarity ---
posts_df, all_text_df, insights, processed_perfected_df = load_and_process_data()

if all_text_df.empty and processed_perfected_df.empty:
    st.error("No data files found. Please add CSV/TXT files to 'data/uploads' or 'all_posts_with_comments.csv' to 'perfected_data'.")
    st.info("The 'AI Recommendations' and 'Action Items' tabs are powered exclusively by 'perfected_data/all_posts_with_comments.csv'.")
    st.stop()

# Filter regular data for Prime Bank mentions for general insights
prime_posts_df = posts_df[posts_df['prime_mentions'] > 0].copy() if 'prime_mentions' in posts_df.columns else pd.DataFrame()
prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0].copy() if 'prime_mentions' in all_text_df.columns else pd.DataFrame()

# --- KPI Section (uses general data from 'uploads' folder) ---
st.header("📈 Prime Bank Key Performance Indicators")
total_mentions = all_text_df['prime_mentions'].sum() if 'prime_mentions' in all_text_df.columns else 0
total_posts_with_mentions = len(prime_posts_df)
new_metrics = create_summary_metrics(all_text_df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Mentions (Posts & Comments)", f"{int(total_mentions):,}")
kpi2.metric("Posts Mentioning Prime Bank", f"{total_posts_with_mentions:,}")
kpi3.metric("Bank Sentiment Score", new_metrics['Bank Sentiment Score'], help="Positive Mentions - Negative Mentions. A positive score is good.")
kpi4.metric("Engagement-Weighted Sentiment", new_metrics['Engagement-Weighted Sentiment'], help="A combined score of sentiment polarity and virality (likes, shares, etc.). Higher is better.")
st.markdown("---")

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sentiment & Virality (Posts)",
    "Emotion & Categories (All Text)",
    "Strategic Overview",
    "🤖 AI Recommendations",
    "Action Items",
    "Full Data View"
])

# --- Tab 1 to 3 use GENERAL data from 'uploads' folder ---
with tab1:
    st.header("Sentiment & Virality Analysis (Posts Only)")
    if not prime_posts_df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sentiment of Posts")
            st.plotly_chart(create_sentiment_pie(prime_posts_df), use_container_width=True)
        with col2:
            st.subheader("Top Viral Posts")
            viral_chart = create_viral_posts_chart(prime_posts_df)
            if viral_chart:
                st.plotly_chart(viral_chart, use_container_width=True)
            else:
                st.info("No viral score data (likes, shares, comments) found to display chart.")
    else:
        st.info("No posts mentioning Prime Bank were found in the data.")

with tab2:
    st.header("Emotion & Category Analysis (Posts & Comments)")
    if not prime_all_text_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emotion Detection")
            st.plotly_chart(create_emotion_bar(prime_all_text_df), use_container_width=True)
        with col2:
            st.subheader("Post & Comment Categories")
            st.plotly_chart(create_category_donut(prime_all_text_df), use_container_width=True)
    else:
        st.info("No text mentioning Prime Bank was found in the data.")

with tab3:
    st.header("Strategic Overview")
    st.write("High-level insights into market position and geographic distribution.")
    col1, col2 = st.columns(2)
    with col1:
        bank_comp_chart = create_bank_comparison_chart(all_text_df)
        if bank_comp_chart:
            st.plotly_chart(bank_comp_chart, use_container_width=True)
        else:
            st.info("Not enough data to compare bank mentions.")
    with col2:
        geo_map = create_geolocation_map(all_text_df)
        if geo_map:
            st.plotly_chart(geo_map, use_container_width=True)
        else:
            # Message is handled inside the create_geolocation_map function
            pass

# --- Tab 4: AI Recommendations (uses PERFECTED data) ---
with tab4:
    st.header("🤖 AI-Powered Strategic Recommendations")
    st.info("These recommendations are generated exclusively from the curated data in `perfected_data/all_posts_with_comments.csv`.")
    if insights and 'ai_recommendations' in insights and insights['ai_recommendations']:
        recs = insights['ai_recommendations']
        st.subheader("For Customer Complaints")
        with st.expander("Show AI Insight on Complaints", expanded=True):
            st.markdown(f"💡 {recs.get('Complaint', 'No specific recommendation available.')}")

        st.subheader("For Customer Suggestions")
        with st.expander("Show AI Insight on Suggestions"):
            st.markdown(f"💡 {recs.get('Suggestion', 'No specific recommendation available.')}")
        
        st.subheader("For Customer Praise")
        with st.expander("Show AI Insight on Praise"):
            st.markdown(f"💡 {recs.get('Praise', 'No specific recommendation available.')}")
            
        st.subheader("For Customer Inquiries")
        with st.expander("Show AI Insight on Inquiries"):
            st.markdown(f"💡 {recs.get('Inquiry', 'No specific recommendation available.')}")
    else:
        st.warning("No AI recommendations could be generated. This requires a valid `perfected_data/all_posts_with_comments.csv` file with relevant data and a configured OpenAI API key.")

# --- Tab 5: Action Items (uses PERFECTED data) ---
# --- Tab 5: Action Items (uses PERFECTED data) ---
with tab5:
    st.header("Posts & Comments That Need Attention")
    st.info("This prioritized list is generated exclusively from the curated data in `perfected_data/all_posts_with_comments.csv`.")
    
    if not processed_perfected_df.empty and 'prime_mentions' in processed_perfected_df.columns:
        prime_perfected_df = processed_perfected_df[processed_perfected_df['prime_mentions'] > 0].copy()
        
        attention_df = prime_perfected_df[
            (prime_perfected_df['sentiment'] == 'Negative') | 
            (prime_perfected_df['category'].isin(['Complaint', 'Inquiry']))
        ].copy()

        if not attention_df.empty:
            # Calculate priority score
            attention_df['priority_score'] = (
                (attention_df['sentiment'] == 'Negative') * 2 +
                (attention_df['category'] == 'Complaint') * 1.5 +
                (attention_df['category'] == 'Inquiry') * 1
            )
            attention_df.sort_values(by=['priority_score', 'viral_score'], ascending=[False, False], inplace=True)
            
            # Prepare display columns and add clickable source link
            display_columns = ['text', 'sentiment', 'category', 'emotion', 'viral_score']
            link_col = None
            if 'link' in attention_df.columns:
                link_col = 'link'
            elif 'url' in attention_df.columns:
                link_col = 'url'

            if link_col:
                # Check if links are real or just placeholders
                attention_df['Source'] = attention_df[link_col].apply(
                    lambda url: "Customer Post" if pd.notna(url) and 'example.com' in str(url) 
                    else (f"[Open Post ↗]({url})" if pd.notna(url) and str(url).startswith('http') 
                    else "Direct Feedback")
                )
                display_columns.insert(1, 'Source')
            
            st.dataframe(attention_df[display_columns], use_container_width=True, hide_index=True)
        else:
            st.success("✅ No negative comments or inquiries found in the perfected data that require attention.")
    else:
        st.warning("No data found in `perfected_data/all_posts_with_comments.csv` to analyze for action items.")

# --- Tab 6: Data View (shows both general and perfected data separately) ---
with tab6:
    st.header("Explore the Raw and Processed Data")
    
    if not processed_perfected_df.empty:
        st.subheader("Processed Perfected Data (from `perfected_data`)")
        st.dataframe(processed_perfected_df.head(100))

    if not posts_df.empty:
        st.subheader("Processed Posts Data (from `data/uploads`)")
        st.dataframe(posts_df.head(100))

    comments_df = all_text_df[all_text_df['file_type'] == 'comment'] if 'file_type' in all_text_df.columns else pd.DataFrame()
    if not comments_df.empty:
        st.subheader("Processed Comments & Reviews Data (from `data/uploads`)")
        st.dataframe(comments_df.head(100))