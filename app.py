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

# --- Helper function to identify text column ---
def find_text_column(df):
    if df.empty: return None
    text_columns = [
        'text', 'Text', 'content', 'Content', 'message', 'Message',
        'review', 'Review', 'comment', 'Comment', 'post', 'Post',
        'review_text', 'Review Text', 'post_text', 'Post Text',
        'comment_text', 'Comment Text', 'description', 'Description'
    ]
    for col in text_columns:
        if col in df.columns: return col
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'content', 'review', 'comment', 'post']): return col
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head()
            if not sample.empty:
                try:
                    if sample.astype(str).str.len().mean() > 20: return col
                except: continue
    return None

# --- Caching for Performance ---
@st.cache_data
def load_and_process_data():
    DATA_DIR = 'data/uploads'
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    all_files = glob.glob(os.path.join(DATA_DIR, '*'))
    if not all_files: return pd.DataFrame(), pd.DataFrame(), None

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
                else: # txt
                    with open(f, 'r', encoding='utf-8') as file:
                        df = pd.DataFrame({'text': [p.strip() for p in file.read().split('\n') if p.strip()]})
                
                text_col = find_text_column(df)
                if not text_col: continue
                if text_col != 'text': df = df.rename(columns={text_col: 'text'})
                
                df['source_file'] = os.path.basename(f)
                df['file_type'] = file_type
                df = df[df['text'].notna() & (df['text'].str.strip() != '')]
                if not df.empty: dfs.append(df)
            except Exception as e:
                st.error(f"Error reading {os.path.basename(f)}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    raw_posts_df = read_files(post_files + other_csv_files, 'post')
    raw_comments_df = read_files(comment_files + txt_files, 'comment')

    processor = DataProcessor()
    processed_posts_df = processor.process_all_data(raw_posts_df)
    processed_comments_df = processor.process_all_data(raw_comments_df)
    
    all_text_df = pd.concat([processed_posts_df, processed_comments_df], ignore_index=True)
    if all_text_df.empty: return pd.DataFrame(), pd.DataFrame(), None

    insight_gen = InsightsGenerator()
    insights = insight_gen.generate_all_insights(posts_df=processed_posts_df, all_text_df=all_text_df)
    
    return processed_posts_df, all_text_df, insights

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics")

posts_df, all_text_df, insights = load_and_process_data()

if all_text_df.empty or insights is None:
    st.error("No data files found or processed in 'data/uploads'. Please add CSV or TXT files.")
    st.info("Ensure filenames contain 'post' for post data or 'comment' for comment data for best results.")
    st.stop()

# Filter for Prime Bank mentions
prime_posts_df = posts_df[posts_df['prime_mentions'] > 0].copy() if 'prime_mentions' in posts_df.columns else pd.DataFrame()
prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0].copy() if 'prime_mentions' in all_text_df.columns else pd.DataFrame()

# --- KPI Section ---
st.header("📈 Prime Bank Key Performance Indicators")
total_mentions = all_text_df['prime_mentions'].sum()
total_posts_with_mentions = len(prime_posts_df)
new_metrics = create_summary_metrics(all_text_df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Mentions (Posts & Comments)", f"{int(total_mentions):,}")
kpi2.metric("Posts Mentioning Prime Bank", f"{total_posts_with_mentions:,}")
kpi3.metric("Bank Sentiment Score", new_metrics['Bank Sentiment Score'], help="Positive Mentions - Negative Mentions. A positive score is good.")
kpi4.metric("Engagement-Weighted Sentiment", new_metrics['Engagement-Weighted Sentiment'], help="A combined score of sentiment polarity and virality (likes, shares, etc.). Higher is better.")
st.markdown("---")

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment & Virality (Posts)",
    "Emotion & Categories (All Text)",
    "Strategic Overview",
    "Action Items",
    "Full Data View"
])

# --- Tab 1: Posts Only Analysis ---
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

# --- Tab 2: All Text Analysis ---
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

# --- Tab 3: Strategic Overview ---
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
            # This message is now handled inside create_geolocation_map
            pass

# --- Tab 4: Action Items ---
with tab4:
    st.header("Posts & Comments That Need Attention")
    st.write("A prioritized list of negative or inquiry-based comments mentioning Prime Bank.")

    if not prime_all_text_df.empty:
        attention_df = prime_all_text_df[
            (prime_all_text_df['sentiment'] == 'Negative') |
            (prime_all_text_df['category'].isin(['Complaint', 'Inquiry']))
        ].copy()

        if not attention_df.empty:
            attention_df['priority_score'] = (
                (attention_df['sentiment'] == 'Negative') * 2 +
                (attention_df['category'] == 'Complaint') * 1.5 +
                (attention_df['category'] == 'Inquiry') * 1
            )
            attention_df.sort_values(by='priority_score', ascending=False, inplace=True)
            
            # Define columns to display initially
            display_columns = ['text', 'sentiment', 'category', 'emotion', 'viral_score']
            
            # Check for a link column ('link' or 'url')
            link_col = None
            if 'link' in attention_df.columns:
                link_col = 'link'
            elif 'url' in attention_df.columns:
                link_col = 'url'
            
            # Configure the dataframe display
            column_config = {}
            if link_col:
                # Add the link column to the list of displayed columns
                display_columns.insert(1, link_col)
                # Configure the link column to be clickable
                column_config[link_col] = st.column_config.LinkColumn(
                    "Source Link",
                    display_text="Open Post ↗"
                )
            
            # Display the dataframe with or without the link column
            st.dataframe(
                attention_df[display_columns],
                use_container_width=True,
                column_config=column_config,
                hide_index=True  # Hiding the index for a cleaner look
            )
        else:
            st.success("✅ No negative comments or inquiries found that require attention.")
    else:
        st.info("No data mentioning Prime Bank to analyze for action items.")

# --- Tab 5: Data View ---
with tab5:
    st.header("Explore the Raw and Processed Data")
    if not posts_df.empty:
        st.subheader("Processed Posts Data")
        st.dataframe(posts_df.head(100))

    comments_df = all_text_df[all_text_df['file_type'] == 'comment'] if 'file_type' in all_text_df.columns else pd.DataFrame()
    if not comments_df.empty:
        st.subheader("Processed Comments & Reviews Data")
        st.dataframe(comments_df.head(100))