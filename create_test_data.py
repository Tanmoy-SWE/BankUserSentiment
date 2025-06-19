import pandas as pd

# Create sample CSV data
sample_data = pd.DataFrame({
    'text': [
        'Prime Bank has the best customer service! Love their mobile app.',
        'Worst experience at Prime Bank branch today. Waited 2 hours!',
        'How do I apply for a loan at Prime Bank?',
        'Prime Bank ATM is not working again. So frustrated!',
        'Thank you Prime Bank staff for helping with my account.',
        'What are Prime Bank interest rates?',
        'Prime Bank online banking is confusing.',
        'Excellent service at Prime Bank downtown branch!',
        'Prime Bank charged me hidden fees. Very disappointed.',
        'Can someone explain Prime Bank credit card benefits?'
    ],
    'date': pd.date_range('2024-01-01', periods=10),
    'likes': [45, 12, 5, 89, 34, 8, 15, 67, 102, 22],
    'shares': [5, 2, 1, 15, 8, 1, 3, 12, 25, 4]
})

# Save as CSV
sample_data.to_csv('test_social_media_data.csv', index=False)
print("✅ Created test_social_media_data.csv")

# Create sample TXT file with reviews
reviews = """Prime Bank provides exceptional service. Highly recommend!
Terrible experience with Prime Bank customer support.
Prime Bank mobile app keeps crashing. Please fix this!
Love the new features in Prime Bank online banking.
Why does Prime Bank charge so many fees?"""

with open('test_reviews.txt', 'w') as f:
    f.write(reviews)
print("✅ Created test_reviews.txt")