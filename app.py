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
    Loads data by differentiating between posts and comments, processes them,
    and generates insights with robust handling for missing file types.
    """
    DATA_DIR = 'data/uploads'
    
    # Locate all relevant files
    post_files = glob.glob(os.path.join(DATA_DIR, '*_posts.csv'))
    comment_files = glob.glob(os.path.join(DATA_DIR, '*_comments.csv'))
    txt_files = glob.glob(os.path.join(DATA_DIR, '*.txt'))

    # Helper function to read a list of files into a single DataFrame
    def read_files_to_dataframe(files_list):
        dfs = []
        for f in files_list:
            try:
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
        
        # Return a single combined DataFrame or an empty one
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # --- NEW ROBUST LOGIC ---
    # 1. Combine raw data first
    raw_posts_df = read_files_to_dataframe(post_files)
    raw_comments_df = read_files_to_dataframe(comment_files + txt_files)

    processor = DataProcessor()

    # 2. Process only if the dataframe is not empty
    processed_posts_df = processor.process_all_data(raw_posts_df) if not raw_posts_df.empty else pd.DataFrame()
    processed_comments_df = processor.process_all_data(raw_comments_df) if not raw_comments_df.empty else pd.DataFrame()

    # 3. Create the 'all_text_df' for combined analysis from processed parts
    all_text_df = pd.concat([processed_posts_df, processed_comments_df], ignore_index=True)

    # 4. Check if there's any data at all before generating insights
    if all_text_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None # Return empty structures

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

if all_text_df.empty or insights is None:
    st.error("No data files found or processed in 'data/uploads'. Please ensure your files are named correctly (e.g., 'prime_bank_posts.csv', 'prime_bank_comments.csv').")
    st.stop()

# Filter for Prime Bank specific data
prime_posts_df = posts_df[posts_df['prime_mentions'] > 0].copy() if 'prime_mentions' in posts_df else pd.DataFrame()
prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0].copy() if 'prime_mentions' in all_text_df else pd.DataFrame()


# --- KPI Section ---
st.header("📈 Prime Bank Mention KPIs")
kpi1, kpi2 = st.columns(2)
total_mentions = all_text_df['prime_mentions'].sum() if 'prime_mentions' in all_text_df else 0
total_posts_with_mentions = len(prime_posts_df)
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
    
    if not prime_posts_df.empty:
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
    else:
        st.info("No posts mentioning Prime Bank were found in the data.")


# --- Tab 2: Posts and Comments Analysis ---
with tab2:
    st.header("Emotion & Category Analysis (Posts & Comments)")
    
    if not prime_all_text_df.empty:
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
    else:
        st.info("No posts or comments mentioning Prime Bank were found in the data.")

# --- Tab 3: Data View ---
with tab3:
    st.header("Explore the Raw and Processed Data")
    if not posts_df.empty:
        st.subheader("Processed Posts Data")
        st.dataframe(posts_df)
    if not all_text_df.empty and len(all_text_df) > len(posts_df):
        st.subheader("Processed Comments & Reviews Data")
        st.dataframe(all_text_df.iloc[len(posts_df):].reset_index(drop=True))