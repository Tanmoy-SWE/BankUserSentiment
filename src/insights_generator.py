#insights_generator.py
import pandas as pd
from collections import Counter
import re

# ### CHANGED: Import the new OpenAI client ###
import openai

class InsightsGenerator:
    """
    Analyzes processed data to generate rich, qualitative insights for the dashboard.
    """
import os
from openai import OpenAI

class InsightsGenerator:
    def __init__(self, openai_api_key=None):
        self.insights = {}
        self.client = None
        if openai_api_key and OPENAI_AVAILABLE:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            openai.api_key = openai_api_key
            self.client = True

    # ... (the other functions are fine) ...
    def generate_all_insights(self, posts_df, all_text_df):
        if 'prime_mentions' not in posts_df.columns: prime_posts_df = pd.DataFrame()
        else: prime_posts_df = posts_df[posts_df['prime_mentions'] > 0]
        if 'prime_mentions' not in all_text_df.columns: prime_all_text_df = pd.DataFrame()
        else: prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0]
        self.insights['sentiment'] = self._generate_sentiment_insights(prime_posts_df)
        self.insights['emotion'] = self._generate_emotion_insights(prime_all_text_df)
        self.insights['category'] = self._generate_category_insights(prime_all_text_df)
        return self.insights

    def _get_common_words(self, text_series, top_n=5):
        if text_series.empty: return "No data"
        stop_words = {'the', 'a', 'an', 'is', 'i', 'to', 'for', 'in', 'it', 'and', 'my', 'of', 'prime', 'bank', 'banker', 'was', 'do', 'with', 'that', 'this', 'have', 'has', 'are', 'not', 'er', 'ta', 'ki', 'ami', 'amar', 'kore', 'hocche', 'bhalo', 'asholei', 'onek', 'apnar', 'sir', 'bro', 'please', 'help', 'need', 'know', 'want'}
        all_text = ' '.join(text_series.astype(str).tolist()).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        filtered_words = [word for word in words if word not in stop_words]
        if not filtered_words: return "general topics"
        return ', '.join([word for word, count in Counter(filtered_words).most_common(top_n)])

    def _generate_sentiment_insights(self, df):
        if df.empty: return {'summary': 'No posts found for sentiment analysis.', 'positive_themes': 'N/A', 'negative_themes': 'N/A', 'negative_examples': []}
        dist = df['sentiment'].value_counts(normalize=True) * 100
        positive_df = df[df['sentiment'] == 'Positive']
        negative_df = df[df['sentiment'] == 'Negative']
        positive_themes = self._get_common_words(positive_df['text'])
        negative_themes = self._get_common_words(negative_df['text'])
        return {'summary': f"Positive: {dist.get('Positive', 0):.1f}%, Negative: {dist.get('Negative', 0):.1f}%, Neutral: {dist.get('Neutral', 0):.1f}%", 'positive_themes': f"Customers are happy about: {positive_themes}.", 'negative_themes': f"Customers are unhappy about: {negative_themes}.", 'negative_examples': negative_df['text'].head(3).tolist()}

    def _generate_emotion_insights(self, df):
        if df.empty: return {'summary': 'No text found for emotion analysis.', 'details': {}}
        emotion_dist = df['emotion'].value_counts()
        top_emotion = emotion_dist.index[0] if not emotion_dist.empty else "N/A"
        insight_details = {}
        for emotion in ['Joy', 'Frustration', 'Confusion', 'Anxiety']:
            if emotion in df['emotion'].values:
                emotion_df = df[df['emotion'] == emotion]
                insight_details[emotion] = {'themes': self._get_common_words(emotion_df['text'], 3), 'example': emotion_df['text'].iloc[0] if not emotion_df.empty else "N/A"}
        return {'summary': f"The most common emotion is '{top_emotion}'.", 'details': insight_details}

    def _generate_category_insights(self, df):
        if df.empty: return {'summary': 'No text found for category analysis.', 'details': {}}
        insight_details = {}
        for category in ['Complaint', 'Inquiry', 'Praise', 'Suggestion']:
            if category in df['category'].values:
                category_df = df[df['category'] == category]
                insight_details[category] = {'themes': self._get_common_words(category_df['text'], 4)}
        return {'summary': f"The most frequent category is '{df['category'].mode()[0]}'. Complaints and Inquiries are key areas to watch.", 'details': insight_details}

    def _call_gpt_for_summary(self, prompt, max_tokens=150):
        if not self.client:
            return "OpenAI API key not configured. Cannot generate the AI recommendations."
        try:
            # ### CHANGED: Use the new client to make the API call ###
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sharp, concise banking strategy analyst. Your goal is to provide actionable advice based on customer feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    def generate_ai_recommendations(self, df):
        if df.empty: return {}
        recommendations = {}
        category_prompts = {
            'Complaint': "Based on these customer complaints, identify the main theme and suggest one concrete action Prime Bank could take to resolve these issues for future customer satisfaction. Complaints:\n\n{}",
            'Suggestion': "Based on these customer suggestions, what is the most impactful feature or service improvement Prime Bank should prioritize? Briefly explain why. Suggestions:\n\n{}",
            'Praise': "Based on this positive feedback, what is Prime Bank doing right that they should double-down on or use in their marketing? Praise:\n\n{}",
            'Inquiry': "These are common questions from customers. What is the most frequent topic of confusion? Suggest how Prime Bank could clarify this through their website FAQ or app. Inquiries:\n\n{}"
        }
        for category, prompt_template in category_prompts.items():
            category_df = df[df['category'] == category]
            if not category_df.empty:
                snippets = "\n- ".join(category_df['text'].head(20).tolist())
                full_prompt = prompt_template.format(snippets)
                ai_summary = self._call_gpt_for_summary(full_prompt)
                recommendations[category] = ai_summary
            else:
                recommendations[category] = "No data found for this new category."
        return recommendations
