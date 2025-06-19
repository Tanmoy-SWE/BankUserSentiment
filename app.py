# app.py

import streamlit as st
import pandas as pd
import os
import glob
from src.data_processor import DataProcessor
from src.insights_generator import InsightsGenerator
from src.visualizations import *

# --- Page Configuration ---
st.set_page_config(
    page_title="Prime Bank Analytics Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Caching for Performance ---
@st.cache_data
def load_and_process_data():
    """
    Loads data by differentiating between posts and comments based on filename,
    processes them, and generates insights.
    """
    DATA_DIR = 'data/uploads'
    
    # NEW: Load files based on naming convention
    post_files = glob.glob(os.path.join(DATA_DIR, '*_posts.csv'))
    comment_files = glob.glob(os.path.join(DATA_DIR, '*_comments.csv'))
    txt_files = glob.glob(os.path.join(DATA_DIR, '*.txt')) # Treat TXT as comments/reviews

    if not post_files and not comment_files and not txt_files:
        return None, None, None # Return None if no data

    def read_files(files):
        dfs = []
        for f in files:
            try:
                # Use different readers for different file types
                if f.endswith('.csv'):
                    df = pd.read_csv(f)
                else: # for .txt files
                    with open(f, 'r', encoding='utf-8') as file:
                        content = file.read()
                    posts = content.split('\n')
                    df = pd.DataFrame({'text': [p.strip() for p in posts if p.strip()]})
                
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
            except Exception as e:
                st.error(f"Error reading {os.path.basename(f)}: {e}")
        return dfs

    # Create separate lists of DataFrames
    post_dfs = read_files(post_files)
    comment_dfs = read_files(comment_files + txt_files)

    if not post_dfs and not comment_dfs:
        return None, None, None

    # Process DataFrames if they exist
    processor = DataProcessor()
    processed_posts_df = pd.DataFrame()
    if post_dfs:
        combined_posts = pd.concat(post_dfs, ignore_index=True)
        processed_posts_df = processor.process_all_data(combined_posts)

    processed_comments_df = pd.DataFrame()
    if comment_dfs:
        combined_comments = pd.concat(comment_dfs, ignore_index=True)
        processed_comments_df = processor.process_all_data(combined_comments)
    
    # Create the 'all_text_df' for combined analysis
    all_text_df = pd.concat([processed_posts_df, processed_comments_df], ignore_index=True)

    # Generate insights using the new structure
    insight_gen = InsightsGenerator()
    insights = insight_gen.generate_all_insights(
        posts_df=processed_posts_df, 
        all_text_df=all_text_df
    )
    
    return processed_posts_df, all_text_df, insights

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics")

# Load data
posts_df, all_text_df, insights = load_and_process_data()

if posts_df is None or all_text_df is None or insights is None:
    st.error("No data files found in 'data/uploads'. Please ensure your files are named correctly (e.g., 'prime_bank_posts.csv', 'prime_bank_comments.csv').")
    st.stop()

# Filter for Prime Bank specific data
prime_posts_df = posts_df[posts_df['prime_mentions'] > 0].copy()
prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0].copy()

# --- KPI Section ---
st.header("📈 Prime Bank Mention KPIs")
kpi1, kpi2 = st.columns(2)
total_mentions = all_text_df['prime_mentions'].sum()
total_posts_with_mentions = len(posts_df[posts_df['prime_mentions'] > 0])
kpi1.metric(
    label="Total Prime Bank Mentions (in Posts & Comments)",
    value=f"{int(total_mentions):,}"
)
kpi2.metric(
    label="Total Posts Mentioning Prime Bank",
    value=f"{total_posts_with_mentions:,}"
)
st.markdown("---")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["Sentiment & Virality (Posts)", "Emotion & Categories (All Text)", "Full Data View"])

# --- Tab 1: Posts Only Analysis ---
with tab1:
    st.header("Sentiment & Virality Analysis (Posts Only)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Sentiment of Posts")
        st.plotly_chart(create_sentiment_pie(prime_posts_df), use_container_width=True)
        with st.expander("Read Sentiment Insights"):
            sentiment_insight = insights.get('sentiment', {})
            st.markdown(f"**Summary:** {sentiment_insight.get('summary', 'N/A')}")
            st.markdown(f"**Positive Themes:** {sentiment_insight.get('positive_themes', 'N/A')}")
            st.markdown(f"**Negative Themes:** {sentiment_insight.get('negative_themes', 'N/A')}")
            st.write("**Examples of Negative Posts:**")
            for example in sentiment_insight.get('negative_examples', []):
                st.warning(f"- \"{example[:150]}...\"")
    
    with col2:
        st.subheader("Top Viral Posts")
        st.plotly_chart(create_viral_posts_chart(prime_posts_df), use_container_width=True)

# --- Tab 2: Posts and Comments Analysis ---
with tab2:
    st.header("Emotion & Category Analysis (Posts & Comments)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Emotion Detection")
        st.plotly_chart(create_emotion_bar(prime_all_text_df), use_container_width=True)
        with st.expander("Read Emotion Insights"):
            emotion_insight = insights.get('emotion', {})
            st.markdown(f"**Summary:** {emotion_insight.get('summary', 'N/A')}")
            for emotion, data in emotion_insight.get('details', {}).items():
                st.markdown(f"**{emotion} is often about:** {data['themes']}")
                st.write(f"Example:")
                st.info(f"- \"{data['example'][:150]}...\"")

    with col2:
        st.subheader("Post & Comment Categories")
        st.plotly_chart(create_category_donut(prime_all_text_df), use_container_width=True)
        with st.expander("Read Category Insights"):
            category_insight = insights.get('category', {})
            st.markdown(f"**Summary:** {category_insight.get('summary', 'N/A')}")
            for category, data in category_insight.get('details', {}).items():
                st.markdown(f"**{category} topics include:** {data['themes']}")

# --- Tab 3: Data View ---
with tab3:
    st.header("Explore the Raw and Processed Data")
    st.subheader("Processed Posts Data")
    st.dataframe(posts_df)
    st.subheader("Processed Comments & Reviews Data")
    st.dataframe(all_text_df[len(posts_df):].reset_index(drop=True)) # Show only comments