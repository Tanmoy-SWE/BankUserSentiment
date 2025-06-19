import pandas as pd
import re
from textblob import TextBlob
import numpy as np
import json

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed. GPT features will be disabled.")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. Using TextBlob only.")

class DataProcessor:
    def __init__(self, openai_api_key=None):
        self.processed_data = None
        
        # Initialize VADER if available
        if NLTK_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except:
                self.sia = None
        else:
            self.sia = None
            
        # Set up OpenAI if key provided and library available
        self.use_gpt = False
        if openai_api_key and OPENAI_AVAILABLE:
            openai.api_key = openai_api_key
            self.use_gpt = True
            
        # Banking-specific patterns
        self.banking_keywords = {
            'service_quality': ['customer service', 'staff', 'support', 'help', 'assistance'],
            'transaction': ['transfer', 'deposit', 'withdraw', 'payment', 'transaction'],
            'account': ['account', 'savings', 'checking', 'balance'],
            'loan': ['loan', 'mortgage', 'credit', 'interest rate'],
            'digital': ['app', 'online banking', 'mobile', 'website', 'digital'],
            'branch': ['branch', 'atm', 'location', 'queue', 'waiting']
        }
        
    def process_csv_files(self, uploaded_files):
        """Process multiple CSV files"""
        all_dataframes = []
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                df['source_file'] = uploaded_file.name
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {uploaded_file.name}: {e}")
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df
        return pd.DataFrame()
    
    def process_txt_file(self, txt_file):
        """Process text file with reviews"""
        content = txt_file.read().decode('utf-8')
        reviews = content.split('\n')
        
        df = pd.DataFrame({
            'text': [review.strip() for review in reviews if review.strip()],
            'source_file': txt_file.name
        })
        return df
    
    def analyze_sentiment(self, text):
        """Analyze sentiment - use VADER if available, else TextBlob"""
        if pd.isna(text) or str(text).strip() == '':
            return 'Neutral', 0
        
        text_str = str(text)
        
        # Try VADER first if available
        if self.sia:
            scores = self.sia.polarity_scores(text_str)
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'Positive', compound
            elif compound <= -0.05:
                return 'Negative', compound
            else:
                return 'Neutral', compound
        
        # Fallback to TextBlob
        try:
            blob = TextBlob(text_str)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        except:
            return 'Neutral', 0
    
    def detect_emotion(self, text):
        """Detect emotion in text"""
        if pd.isna(text):
            return 'Neutral'
        
        text_lower = str(text).lower()
        
        # Emotion keywords
        emotions = {
            'Joy': ['happy', 'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'love', 'best', 'thank you'],
            'Frustration': ['frustrated', 'angry', 'terrible', 'horrible', 'worst', 'hate', 'annoyed', 'disappointed'],
            'Confusion': ['confused', 'unclear', "don't understand", 'what', 'how', 'why', '?', 'help me']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotions.items():
            score = sum(keyword in text_lower for keyword in keywords)
            emotion_scores[emotion] = score
        
        if max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        return 'Neutral'
    
    def categorize_post(self, text):
        """Categorize post type"""
        if pd.isna(text):
            return 'Other'
        
        text_lower = str(text).lower()
        
        if '?' in text_lower or any(word in text_lower for word in ['how', 'what', 'when', 'where']):
            return 'Inquiry'
        elif any(word in text_lower for word in ['complaint', 'problem', 'issue', 'bad', 'terrible']):
            return 'Complaint'
        elif any(word in text_lower for word in ['thank', 'great', 'excellent', 'love', 'best']):
            return 'Praise'
        else:
            return 'Other'
    
    def count_prime_mentions(self, text):
        """Count Prime Bank mentions"""
        if pd.isna(text):
            return 0
        
        text_lower = str(text).lower()
        patterns = [
            r'prime\s*bank',
            r'primebank',
            r'@primebank'
        ]
        
        total_mentions = 0
        for pattern in patterns:
            mentions = len(re.findall(pattern, text_lower))
            total_mentions += mentions
            
        return total_mentions
    
    def process_all_data(self, df):
        """Apply all processing to dataframe"""
        # Find text column
        text_columns = ['text', 'content', 'message', 'review', 'comment', 'post']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col and text_col != 'text':
            df['text'] = df[text_col]
        
        if 'text' not in df.columns:
            return df
        
        # Apply all analyses
        df[['sentiment', 'polarity']] = df['text'].apply(
            lambda x: pd.Series(self.analyze_sentiment(x))
        )
        
        df['emotion'] = df['text'].apply(self.detect_emotion)
        df['category'] = df['text'].apply(self.categorize_post)
        df['prime_mentions'] = df['text'].apply(self.count_prime_mentions)
        
        # Calculate viral score
        df['viral_score'] = df['prime_mentions'] * 10
        if 'likes' in df.columns:
            df['viral_score'] += df['likes'].fillna(0)
        if 'shares' in df.columns:
            df['viral_score'] += df['shares'].fillna(0) * 2
            
        return df