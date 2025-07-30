# AI Career Guidance System (Command-Line Version)
# -------------------------------------------------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Sample Career Dataset
data = {
    'Career': [
        'Data Scientist',
        'Software Engineer',
        'Mechanical Engineer',
        'Digital Marketer',
        'UI/UX Designer'
    ],
    'Description': [
        'Machine learning, data analysis, statistics, python, AI',
        'Coding, system design, Java, Python, problem-solving',
        'CAD, thermodynamics, mechanical systems, design',
        'Social media, SEO, content creation, ads, branding',
        'Design thinking, wireframes, Figma, user research'
    ]
}
df = pd.DataFrame(data)

# 2. Get User Input
print("🔍 Welcome to AI Career Guidance System")
user_input = input("Enter your interests, skills or subjects: ").strip()

if not user_input:
    print("⚠️ Please enter some input to get recommendations.")
    exit()

# 3. Combine Descriptions + User Input
corpus = df['Description'].tolist()
corpus.append(user_input)

# 4. TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)

cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
scores = cosine_sim.flatten()
top_indices = scores.argsort()[::-1][:3]

# 5. Output Recommendations
print("\n✅ Top Career Recommendations for you:")
for i in top_indices:
    print(f"- {df['Career'][i]} → {df['Description'][i]}")
