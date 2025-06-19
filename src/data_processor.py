# src/data_processor.py

import pandas as pd
import re
from textblob import TextBlob
import numpy as np
import json
import streamlit as st # <-- THIS IS THE FIX. Import Streamlit here.

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class DataProcessor:
    def __init__(self, openai_api_key=None):
        self.processed_data = None
        
        # Initialize VADER if available
        if NLTK_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except Exception:
                self.sia = None
        else:
            self.sia = None
            
        # Set up OpenAI if key provided and library available
        self.use_gpt = False
        if openai_api_key and OPENAI_AVAILABLE:
            openai.api_key = openai_api_key
            self.use_gpt = True
            
        # Banking patterns - INCLUDING OTHER BANKS
        self.bank_patterns = {
            'prime_bank': [r'prime\s*bank', r'primebank', r'@primebank', r'prime\s*b\.?'],
            'eastern_bank': [r'eastern\s*bank', r'ebl', r'@easternbank'],
            'brac_bank': [r'brac\s*bank', r'@bracbank'],
            'city_bank': [r'city\s*bank', r'@citybank'],
            'dutch_bangla': [r'dutch\s*bangla', r'dbbl', r'@dutchbangla']
        }
        
    def load_data_from_files(self, csv_files=None, txt_files=None):
        """Load data from CSV and TXT files"""
        all_data = []
        
        if csv_files:
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file_path.split('/')[-1]
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if txt_files:
            for file_path in txt_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    posts = content.split('\n')
                    
                    df = pd.DataFrame({
                        'text': [post.strip() for post in posts if post.strip()],
                        'source_file': file_path.split('/')[-1]
                    })
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def identify_bank(self, text):
        """Identify which bank is mentioned in the text"""
        if pd.isna(text):
            return 'none', []
        
        text_lower = str(text).lower()
        mentioned_banks = []
        
        for bank, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    mentioned_banks.append(bank)
                    break
        
        if not mentioned_banks:
            return 'none', []
        elif len(mentioned_banks) == 1:
            return mentioned_banks[0], mentioned_banks
        else:
            return 'multiple', mentioned_banks
    
    def count_bank_mentions(self, text, bank='prime_bank'):
        """Count mentions of specific bank"""
        if pd.isna(text):
            return 0
        
        text_lower = str(text).lower()
        total_mentions = 0
        
        if bank in self.bank_patterns:
            for pattern in self.bank_patterns[bank]:
                mentions = len(re.findall(pattern, text_lower))
                total_mentions += mentions
        
        return total_mentions
    
    def analyze_sentiment(self, text):
        """Analyze sentiment - use VADER if available, else TextBlob"""
        if pd.isna(text) or str(text).strip() == '':
            return 'Neutral', 0
        
        text_str = str(text)
        
        if self.sia:
            scores = self.sia.polarity_scores(text_str)
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'Positive', compound
            elif compound <= -0.05:
                return 'Negative', compound
            else:
                return 'Neutral', compound
        
        try:
            blob = TextBlob(text_str)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        except Exception:
            return 'Neutral', 0
    
    def detect_emotion(self, text):
        """Detect emotion in text with context"""
        if pd.isna(text):
            return 'Neutral', []
        
        text_lower = str(text).lower()
        
        emotions = {
            'Joy': {'keywords': ['happy', 'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'love', 'best', 'thank you', 'appreciate']},
            'Frustration': {'keywords': ['frustrated', 'angry', 'terrible', 'horrible', 'worst', 'hate', 'annoyed', 'disappointed', 'pathetic']},
            'Confusion': {'keywords': ['confused', 'unclear', "don't understand", 'what', 'how', 'why', '?', 'help me', 'lost']},
            'Anxiety': {'keywords': ['worried', 'concern', 'anxious', 'nervous', 'scared', 'fear', 'panic', 'urgent']}
        }
        
        emotion_scores = {}
        detected_keywords = {}
        
        for emotion, data in emotions.items():
            keywords_found = [kw for kw in data['keywords'] if kw in text_lower]
            score = len(keywords_found)
            emotion_scores[emotion] = score
            if keywords_found:
                detected_keywords[emotion] = keywords_found
        
        if max(emotion_scores.values()) > 0:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            return primary_emotion, detected_keywords.get(primary_emotion, [])
        
        return 'Neutral', []
    
    def categorize_post(self, text):
        """Categorize post type with reason"""
        if pd.isna(text):
            return 'Other', 'No text content'
        
        text_lower = str(text).lower()
        
        if '?' in text_lower or any(phrase in text_lower for phrase in ['how do', 'what is', 'when', 'where', 'can i', 'could you']):
            return 'Inquiry', 'Contains questions or information seeking'
        elif any(word in text_lower for word in ['complaint', 'problem', 'issue', 'error', 'failed', 'not working', 'terrible', 'worst']):
            return 'Complaint', 'Contains complaint or problem description'
        elif any(word in text_lower for word in ['thank', 'great', 'excellent', 'love', 'best', 'appreciate', 'amazing']):
            return 'Praise', 'Contains positive feedback or appreciation'
        elif any(word in text_lower for word in ['suggest', 'should', 'could', 'recommend', 'request', 'please add']):
            return 'Suggestion', 'Contains suggestions or feature requests'
        else:
            return 'Other', 'General discussion or observation'
    
    def process_all_data(self, df):
        """Apply all processing to dataframe"""
        if df.empty:
            return df

        text_columns = ['text', 'content', 'message', 'review', 'comment', 'post', 'Text', 'Content', 'Post', 'Review Text']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            st.warning("Could not find a text column in one of the data sources.")
            return pd.DataFrame(columns=df.columns)

        if text_col != 'text':
            df.rename(columns={text_col: 'text'}, inplace=True)
        
        df[['primary_bank', 'all_banks_mentioned']] = df['text'].apply(lambda x: pd.Series(self.identify_bank(x)))
        df['prime_mentions'] = df['text'].apply(lambda x: self.count_bank_mentions(x, 'prime_bank'))
        df[['sentiment', 'polarity']] = df['text'].apply(lambda x: pd.Series(self.analyze_sentiment(x)))
        df[['emotion', 'emotion_keywords']] = df['text'].apply(lambda x: pd.Series(self.detect_emotion(x)))
        df[['category', 'category_reason']] = df['text'].apply(lambda x: pd.Series(self.categorize_post(x)))
        
        df['viral_score'] = 0
        if 'likes' in df.columns:
            df['viral_score'] += df['likes'].fillna(0)
        if 'shares' in df.columns:
            df['viral_score'] += df['shares'].fillna(0) * 2
        if 'comments' in df.columns:
            df['viral_score'] += df['comments'].fillna(0) * 1.5
        
        if not df.empty and 'prime_mentions' in df.columns:
            df.loc[df['prime_mentions'] > 0, 'viral_score'] *= 1.2
        
        return df