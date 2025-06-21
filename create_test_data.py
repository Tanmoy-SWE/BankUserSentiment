# create_test.py

import pandas as pd
import os

# --- Create a directory for uploads if it doesn't exist ---
UPLOAD_DIR = 'data/uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- Create sample POSTS CSV data with more specific examples ---
sample_data = pd.DataFrame({
    'text': [
        'Prime Bank has the best customer service! Love their mobile app QR payment feature.',
        'Worst experience at Prime Bank. My credit card was declined for no reason and support was useless!',
        'How do I apply for a student account at Prime Bank? The website is unclear.',
        'Prime Bank ATM is not working again. So frustrated!',
        'You guys should really add international transaction alerts to the Prime Bank app. It would be so helpful!',
        'What are Prime Bank interest rates on fixed deposits?',
        'Prime Bank online banking is confusing. I can never find the statement page.',
        'Excellent service at Prime Bank downtown branch! The staff were so helpful.',
        'Prime Bank charged me hidden fees on my savings account. Very disappointed.',
        'Can someone explain Prime Bank credit card reward points? It makes no sense.',
        'Heard good things about Eastern Bank, but Prime Bank is still my go-to.',
        'Comparing BRAC Bank and Prime Bank for a new account.',
        'City Bank has a nice app, but their service is slow.',
        'DBBL needs to improve their network coverage.',
        'My account balance is zero and I did not authorize these transactions! Panicking!', # Good test for hybrid sentiment
    ],
    'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
                            '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15']),
    'likes': [45, 12, 5, 89, 34, 8, 15, 67, 102, 22, 18, 33, 50, 41, 250],
    'shares': [5, 2, 1, 15, 8, 1, 3, 12, 25, 4, 2, 6, 9, 7, 98],
    'comments': [10, 3, 2, 22, 9, 3, 5, 14, 30, 5, 4, 8, 11, 10, 120],
    'location': ['Dhaka', 'Chittagong', 'Dhaka', 'Sylhet', 'Dhaka', 'Rajshahi',
                 'Dhaka', 'Chittagong', 'Sylhet', 'Dhaka', 'Dhaka', 'Chittagong',
                 'Dhaka', 'Sylhet', 'Dhaka'],
    'link': [f'https://social.example.com/posts/{i}' for i in range(1, 16)]
})

# Save as a file with "post" in the name
posts_filepath = os.path.join(UPLOAD_DIR, 'test_social_media_posts.csv')
sample_data.to_csv(posts_filepath, index=False)
print(f"✅ Created {posts_filepath}")

# --- Create sample COMMENTS TXT file ---
reviews = """Prime Bank provides exceptional service. Highly recommend!
Terrible experience with Prime Bank customer support. They never solve the actual problem.
The Prime Bank mobile app keeps crashing whenever I try to check my balance. Please fix this!
Love the new features in Prime Bank online banking, especially the budget tracker.
Why does Prime Bank charge so many fees? It feels like a rip-off.
I suggest you add a dark mode to the app. It would save battery.
BRAC Bank is also a good option for students."""

comments_filepath = os.path.join(UPLOAD_DIR, 'test_review_comments.txt')
with open(comments_filepath, 'w') as f:
    f.write(reviews)
print(f"✅ Created {comments_filepath}")