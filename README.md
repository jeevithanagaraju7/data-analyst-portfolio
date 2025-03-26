# nlp_insights.py
from transformers import pipeline

# Fake customer comments to analyze
comments = [
    "Love this payment app, so fast!",
    "Ugh, my transaction failed again",
    "Pretty good service, I guess"
]

# Set up NLP tool for sentiment analysis
nlp_tool = pipeline("sentiment-analysis")

# Analyze and print each comment
for comment in comments:
    result = nlp_tool(comment)[0]
    feeling = result["label"]  # POSITIVE or NEGATIVE
    score = result["score"]    # Confidence (0-1)
    print(f"Comment: {comment}")
    print(f"Feeling: {feeling}, Score: {score:.2f}\n")

# Save results to a file
with open("customer_insights.txt", "w") as file:
    for comment in comments:
        result = nlp_tool(comment)[0]
        file.write(f"{comment} -> {result['label']}\n")
