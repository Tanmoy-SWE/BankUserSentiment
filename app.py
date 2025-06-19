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

# --- Helper function to identify text column ---
def find_text_column(df):
    """Find the text column in a dataframe"""
    if df.empty:
        return None
        
    # Common text column names
    text_columns = ['text', 'Text', 'content', 'Content', 'message', 'Message', 
                   'review', 'Review', 'comment', 'Comment', 'post', 'Post',
                   'review_text', 'Review Text', 'post_text', 'Post Text',
                   'comment_text', 'Comment Text', 'description', 'Description',
                   'Review', 'Post Content', 'Comment Content']
    
    for col in text_columns:
        if col in df.columns:
            return col
    
    # If no standard column found, look for columns containing certain keywords
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'content', 'message', 'review', 'comment', 'post']):
            return col
    
    # If still no column found, check if there's a column with string data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it contains text-like content
            sample = df[col].dropna().head()
            if len(sample) > 0:
                # Check if values are strings and have reasonable length
                try:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Assume text content is longer than 20 chars
                        return col
                except:
                    continue
    
    return None

# --- Caching for Performance ---
@st.cache_data
def load_and_process_data():
    """
    Loads data by differentiating between posts and comments, processes them,
    and generates insights with robust handling for missing file types.
    """
    DATA_DIR = 'data/uploads'
    
    # Get all files
    all_files = glob.glob(os.path.join(DATA_DIR, '*'))
    
    # Separate files by type
    post_files = []
    comment_files = []
    txt_files = []
    other_csv_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path).lower()
        if filename.endswith('.csv'):
            if 'post' in filename:
                post_files.append(file_path)
            elif 'comment' in filename:
                comment_files.append(file_path)
            else:
                # Files like prime_bank_analysis.csv might be posts
                other_csv_files.append(file_path)
        elif filename.endswith('.txt'):
            txt_files.append(file_path)

    def read_files_to_dataframe(files_list, file_type="general"):
        dfs = []
        for f in files_list:
            try:
                if f.endswith('.csv'):
                    # Try different encodings
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(f, encoding=encoding, on_bad_lines='skip')
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        st.error(f"Could not read {os.path.basename(f)} with any encoding")
                        continue
                    
                    # Find text column
                    text_col = find_text_column(df)
                    
                    if text_col is None:
                        st.warning(f"Could not find text column in {os.path.basename(f)}. Skipping...")
                        continue
                    
                    # Rename text column to 'text'
                    if text_col != 'text':
                        df = df.rename(columns={text_col: 'text'})
                    
                else:  # txt files
                    with open(f, 'r', encoding='utf-8') as file:
                        content = file.read()
                    posts = content.split('\n')
                    df = pd.DataFrame({'text': [p.strip() for p in posts if p.strip()]})
                
                # Add source file info
                df['source_file'] = os.path.basename(f)
                df['file_type'] = file_type
                
                # Only keep non-empty text
                df = df[df['text'].notna() & (df['text'].str.strip() != '')]
                
                if not df.empty:
                    dfs.append(df)
                    
            except Exception as e:
                st.error(f"Error reading {os.path.basename(f)}: {e}")
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Read all files
    raw_posts_df = read_files_to_dataframe(post_files + other_csv_files, 'post')
    raw_comments_df = read_files_to_dataframe(comment_files + txt_files, 'comment')

    # If no posts found, check if we have any CSV files to use as posts
    if raw_posts_df.empty and other_csv_files:
        st.info("No files with 'post' in filename found. Using all CSV files as posts.")
        raw_posts_df = read_files_to_dataframe(other_csv_files, 'post')

    # Process the data
    processor = DataProcessor()

    processed_posts_df = processor.process_all_data(raw_posts_df) if not raw_posts_df.empty else pd.DataFrame()
    processed_comments_df = processor.process_all_data(raw_comments_df) if not raw_comments_df.empty else pd.DataFrame()
    
    # Combine all text for analysis
    all_text_df = pd.concat([processed_posts_df, processed_comments_df], ignore_index=True)

    if all_text_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    # Generate insights
    insight_gen = InsightsGenerator()
    insights = insight_gen.generate_all_insights(
        posts_df=processed_posts_df, 
        all_text_df=all_text_df
    )
    
    return processed_posts_df, all_text_df, insights

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics")

# Load and process data
posts_df, all_text_df, insights = load_and_process_data()

if all_text_df.empty or insights is None:
    st.error("No data files found or processed in 'data/uploads'. Please check your files.")
    st.info("Expected file formats: CSV files with text content, or TXT files with reviews/comments")
    st.stop()

# Filter for Prime Bank mentions
prime_posts_df = posts_df[posts_df['prime_mentions'] > 0].copy() if not posts_df.empty and 'prime_mentions' in posts_df.columns else pd.DataFrame()
prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0].copy() if not all_text_df.empty and 'prime_mentions' in all_text_df.columns else pd.DataFrame()

# --- KPI Section ---
st.header("📈 Prime Bank Mention KPIs")
kpi1, kpi2 = st.columns(2)
total_mentions = all_text_df['prime_mentions'].sum() if 'prime_mentions' in all_text_df.columns else 0
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
                    if example:
                        st.warning(f"- \"{example[:150]}...\"" if len(example) > 150 else f"- \"{example}\"")
        
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
                    st.markdown(f"**{emotion} is often about:** {data.get('themes', 'N/A')}")
                    if data.get('example') and data['example'] != "N/A":
                        st.write("Example:")
                        example_text = data['example']
                        st.info(f"- \"{example_text[:150]}...\"" if len(example_text) > 150 else f"- \"{example_text}\"")

        with col2:
            st.subheader("Post & Comment Categories")
            st.plotly_chart(create_category_donut(prime_all_text_df), use_container_width=True)
            with st.expander("Read Category Insights"):
                category_insight = insights.get('category', {})
                st.markdown(f"**Summary:** {category_insight.get('summary', 'N/A')}")
                for category, data in category_insight.get('details', {}).items():
                    st.markdown(f"**{category} topics include:** {data.get('themes', 'N/A')}")
    else:
        st.info("No posts or comments mentioning Prime Bank were found in the data.")

# --- Tab 3: Data View ---
with tab3:
    st.header("Explore the Raw and Processed Data")
    
    if not posts_df.empty:
        st.subheader("Processed Posts Data")
        st.dataframe(posts_df.head(100))  # Show first 100 rows
    
    if not all_text_df.empty and len(all_text_df) > len(posts_df):
        st.subheader("Processed Comments & Reviews Data")
        comments_df = all_text_df[all_text_df['file_type'] == 'comment'] if 'file_type' in all_text_df.columns else all_text_df.iloc[len(posts_df):]
        st.dataframe(comments_df.head(100))  # Show first 100 rows