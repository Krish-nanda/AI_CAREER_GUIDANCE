# AI Career Guidance System with Tkinter GUI
# -------------------------------------------

import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
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
    ],
        'Projects': [
        'House price predictor, fraud detection system',
        'Chat app, portfolio website, job portal',
        '3D engine model, bicycle CAD design',
        'SEO case study, social media content calendar',
        'Redesign an app, build a portfolio with Figma'
    ],
    'Internships': [
        'LinkedIn (Data Science), Internshala (ML roles)',
        'LinkedIn (SDE roles), HackerEarth',
        'LetsIntern, Internshala (Mechanical)',
        'Upwork, Internshala (Marketing)',
        'Internshala (Design), Behance Jobs'
    ]
}
df = pd.DataFrame(data)

# Recommendation logic
def recommend():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter your interests or skills.")
        return

    corpus = df['Description'].tolist()
    corpus.append(user_input)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    scores = cosine_sim.flatten()
    top_index = scores.argmax()

    result_text = f"Recommended Career: {df['Career'][top_index]}\n\n"

    result_text += f" Projects: {df['Projects'][top_index]}\n"
    result_text += f" Internships: {df['Internships'][top_index]}"
    result_box.config(state='normal')
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, result_text)
    result_box.config(state='disabled')

# GUI setup
root = tk.Tk()
root.title("AI Career Guidance System")
root.geometry("600x500")
root.resizable(False, False)

tk.Label(root, text="Enter your interests, skills or subjects:", font=("Arial", 12)).pack(pady=10)
entry = tk.Text(root, height=4, width=60, font=("Arial", 11))
entry.pack(pady=5)

tk.Button(root, text="Get Career Recommendation", command=recommend, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

tk.Label(root, text="Results:", font=("Arial", 12, "bold")).pack()
result_box = tk.Text(root, height=10, width=70, font=("Arial", 11), wrap=tk.WORD, state='disabled')
result_box.pack(pady=5)

tk.Label(root, text="thank you", font=("Arial", 9)).pack(side=tk.BOTTOM, pady=5)

root.mainloop()
