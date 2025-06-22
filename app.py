import streamlit as st
import pandas as pd
import os
import openai # Import for specific error handling
from src.data_processor import DataProcessor
from src.insights_generator import InsightsGenerator
from src.visualizations import *
from dotenv import load_dotenv

# This will load the OPENAI_API_KEY from your HF Secret (or .env for local)
load_dotenv()

st.set_page_config(page_title="Prime Bank Analytics Dashboard", page_icon="🏦", layout="wide")

# --- DEBUGGING: Function to print status to logs ---
def log_status(message):
    print(f"--- [APP STATUS] --- {message}")

@st.cache_data(show_spinner="Processing data...")
def load_and_process_data(openai_api_key):
    # The app now ONLY reads the clean CSV file created by the parser.
    PROCESSED_CSV_PATH = os.path.join('perfected_data', 'all_posts_with_comments.csv')
    log_status(f"Looking for processed data at: {PROCESSED_CSV_PATH}")

    if not os.path.exists(PROCESSED_CSV_PATH):
        log_status("Processed data file NOT FOUND.")
        st.error(f"Fatal Error: Processed data file not found at {PROCESSED_CSV_PATH}. The parser may have failed. Check logs.")
        return pd.DataFrame(), {}

    try:
        df = pd.read_csv(PROCESSED_CSV_PATH)
        log_status(f"Successfully loaded CSV with {len(df)} rows.")
    except pd.errors.EmptyDataError:
        log_status("CSV file is empty. No data to process.")
        return pd.DataFrame(), {} # Return empty if the CSV is empty
    except Exception as e:
        log_status(f"Failed to read CSV. Error: {e}")
        st.error(f"Could not read the processed CSV file: {e}")
        return pd.DataFrame(), {}

    if df.empty:
        log_status("DataFrame is empty after loading. No data to process.")
        return pd.DataFrame(), {}

    processor = DataProcessor(openai_api_key=openai_api_key)
    insight_gen = InsightsGenerator(openai_api_key=openai_api_key)

    # Wrap the most likely failure point (API calls) in a try/except block
    try:
        log_status("Starting data processing...")
        processed_df = processor.process_all_data(df.copy())
        log_status("Data processing finished.")
        
        insights = {}
        prime_df = processed_df[processed_df.get('prime_mentions', 0) > 0]
        if not prime_df.empty:
            log_status("Generating AI recommendations...")
            insights['ai_recommendations'] = insight_gen.generate_ai_recommendations(prime_df)
            log_status("AI recommendations generated.")
        else:
            log_status("No Prime Bank mentions found, skipping AI recommendations.")
            insights['ai_recommendations'] = {} # Ensure key exists
            
        return processed_df, insights

    except openai.AuthenticationError as e:
        log_status("!!! OpenAI Authentication Error !!!")
        st.error("OpenAI Authentication Error: The API key is invalid or your OpenAI account has billing issues (e.g., no credits). Please verify your key and account status.", icon="🚨")
        return pd.DataFrame(), {} # Return empty to stop
        
    except Exception as e:
        log_status(f"!!! An unexpected error occurred during processing: {e} !!!")
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame(), {}

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics")

openai_key = os.getenv("OPENAI_API_KEY")

# --- DEBUGGING: Add a clear visual indicator for the key status ---
if openai_key and "sk-" in openai_key:
    st.sidebar.success("OpenAI API Key found!", icon="✅")
else:
    st.sidebar.error("OpenAI API Key MISSING from secrets!", icon="🚨")
    st.error("The OpenAI API Key is not configured correctly in your Hugging Face repository secrets. Please check your README.md and the repository 'Settings' tab.")
    st.stop()

# Now, we run the main function, passing the key as an argument
perfected_df, insights = load_and_process_data(openai_key)

if perfected_df.empty:
    st.warning("No data was loaded or processed successfully. Check the logs for errors related to the parser or OpenAI API calls.")
    st.stop()

# --- THE APP TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Recommendations", "Action Items"])

with tab1:
    st.header("Overall Analytics Dashboard")
    st.info("This dashboard provides a high-level overview of all processed comments and posts.")
    
    prime_df = perfected_df[perfected_df.get('prime_mentions', 0) > 0]
    
    if not prime_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sentiment_pie(prime_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_emotion_bar(prime_df), use_container_width=True)
        
        st.plotly_chart(create_category_donut(prime_df), use_container_width=True)
    else:
        st.info("No mentions of Prime Bank found in the processed data.")


with tab2:
    st.header("🤖 AI-Powered Strategic Recommendations")
    if insights and insights.get('ai_recommendations'):
        recs = insights['ai_recommendations']
        if not recs:
            st.info("AI recommendations could not be generated. There might not be enough data in specific categories (like Complaints or Suggestions).")
        for cat, rec in recs.items():
            with st.expander(f"AI Insight on {cat}", expanded=(cat=="Complaint")):
                st.markdown(f"💡 {rec}")
    else:
        st.warning("No AI recommendations were generated. Check your data and API key status.")

with tab3:
    st.header("Posts & Comments That Need Attention")
    attention_df = perfected_df[
        (perfected_df['sentiment'] == 'Negative') |
        (perfected_df['category'].isin(['Complaint', 'Inquiry']))
    ].copy()
    
    if not attention_df.empty:
        # Calculate priority score
        attention_df['priority_score'] = (
            (attention_df['sentiment'] == 'Negative') * 2 +
            (attention_df['category'] == 'Complaint') * 1.5 +
            (attention_df['category'] == 'Inquiry') * 1
        )
        attention_df.sort_values(by='priority_score', ascending=False, inplace=True)
        
        # This will now work because the 'link' column exists in the CSV
        if 'link' in attention_df.columns:
            attention_df['Source'] = attention_df['link'].apply(lambda url: f"[View Post ↗]({url})" if pd.notna(url) and 'http' in str(url) else "No Link")
            display_cols = ['Source', 'text', 'sentiment', 'category', 'emotion']
        else:
            display_cols = ['text', 'sentiment', 'category', 'emotion']
        
        # Ensure all columns exist before trying to display them
        final_display_cols = [col for col in display_cols if col in attention_df.columns]
        
        st.dataframe(attention_df[final_display_cols], use_container_width=True, hide_index=True)
    else:
        st.success("✅ No items requiring attention found in the perfected data.")