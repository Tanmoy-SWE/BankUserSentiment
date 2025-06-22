import pandas as pd
import re
import os

def run_parser():
    RAW_FILE_PATH = os.path.join('perfected_data', 'raw_posts_to_parse.txt')
    OUTPUT_CSV_PATH = os.path.join('perfected_data', 'all_posts_with_comments.csv')
    os.makedirs('perfected_data', exist_ok=True)
    print(f"--- [PARSER STATUS] --- Starting parser.")
    if not os.path.exists(RAW_FILE_PATH):
        print(f"--- [PARSER STATUS] --- Raw data file not found. Cannot create clean CSV.")
        if not os.path.exists(OUTPUT_CSV_PATH):
             pd.DataFrame(columns=['text', 'link']).to_csv(OUTPUT_CSV_PATH, index=False)
        return
    print(f"--- [PARSER STATUS] --- Reading raw data from '{RAW_FILE_PATH}'...")
    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f: content = f.read()
    posts = content.split('==================================================')
    all_rows = []
    for post_block in posts:
        if not post_block.strip(): continue
        post_id = re.search(r'Post ID:\s*(\S+)', post_block)
        post_id = post_id.group(1) if post_id else None
        post_text_match = re.search(r'POST:\n(.*?)\nCOMMENTS:', post_block, re.DOTALL)
        if post_text_match:
            post_text = post_text_match.group(1).replace('\n', ' ').strip()
            all_rows.append({'post_id': post_id, 'text': f"POST: {post_text}", 'type': 'post'})
        if 'COMMENTS:' in post_block:
            comments_section = post_block.split('COMMENTS:')[1]
            for line in comments_section.strip().split('\n'):
                if line.strip(): all_rows.append({'post_id': post_id, 'text': line.strip(), 'type': 'comment'})
    if not all_rows:
        print("--- [PARSER STATUS] --- No data parsed."); return
    df = pd.DataFrame(all_rows)
    def create_link(pid):
        if not pid or 'PR_' not in pid: return "https://www.facebook.com"
        actual_id = pid.split('PR_')[1]
        return f"https://www.facebook.com/posts/{actual_id}"
    df['link'] = df['post_id'].apply(create_link)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"--- [PARSER STATUS] --- ✅ Successfully created clean CSV with {len(df)} rows.")
    return