import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

# Load datasets
trump_reviews = pd.read_csv("Trumpall2.csv")
biden_reviews = pd.read_csv("Bidenall2.csv")

# Quick look at the data
print("Trump dataset sample:")
print(trump_reviews.head())
print("\nBiden dataset sample:")
print(biden_reviews.head())

# Sentiment analysis using TextBlob
def find_pol(review):
    return TextBlob(str(review)).sentiment.polarity

trump_reviews["Sentiment Polarity"] = trump_reviews["text"].apply(find_pol)
biden_reviews["Sentiment Polarity"] = biden_reviews["text"].apply(find_pol)

print("\nTrump sentiment sample:")
print(trump_reviews[["text", "Sentiment Polarity"]].head())
print("\nBiden sentiment sample:")
print(biden_reviews[["text", "Sentiment Polarity"]].head())

# Add Expression Label
trump_reviews["Expression Label"] = np.where(trump_reviews["Sentiment Polarity"] > 0, "positive", "negative")
trump_reviews.loc[trump_reviews["Sentiment Polarity"] == 0, "Expression Label"] = "Neutral"

biden_reviews["Expression Label"] = np.where(biden_reviews["Sentiment Polarity"] > 0, "positive", "negative")
biden_reviews.loc[biden_reviews["Sentiment Polarity"] == 0, "Expression Label"] = "Neutral"

# Drop neutral tweets
trump_reviews = trump_reviews[trump_reviews["Expression Label"] != "Neutral"]
biden_reviews = biden_reviews[biden_reviews["Expression Label"] != "Neutral"]

print("\nTrump dataset after dropping neutral tweets:", trump_reviews.shape)
print("Biden dataset after dropping neutral tweets:", biden_reviews.shape)

# Balance the datasets
np.random.seed(10)
remove_n_trump = len(trump_reviews) - 1000 if len(trump_reviews) > 1000 else 0
remove_n_biden = len(biden_reviews) - 1000 if len(biden_reviews) > 1000 else 0

if remove_n_trump > 0:
    drop_indices_trump = np.random.choice(trump_reviews.index, remove_n_trump, replace=False)
    trump_reviews = trump_reviews.drop(drop_indices_trump)

if remove_n_biden > 0:
    drop_indices_biden = np.random.choice(biden_reviews.index, remove_n_biden, replace=False)
    biden_reviews = biden_reviews.drop(drop_indices_biden)

print("\nBalanced Trump dataset shape:", trump_reviews.shape)
print("Balanced Biden dataset shape:", biden_reviews.shape)

# Analyze sentiment counts
count_trump = trump_reviews.groupby('Expression Label').count()
count_biden = biden_reviews.groupby('Expression Label').count()

print("\nTrump sentiment counts:")
print(count_trump[["Sentiment Polarity"]])
print("\nBiden sentiment counts:")
print(count_biden[["Sentiment Polarity"]])

# Calculate percentages
negative_per_trump = (count_trump.loc["negative", "Sentiment Polarity"] / len(trump_reviews)) * 100
positive_per_trump = (count_trump.loc["positive", "Sentiment Polarity"] / len(trump_reviews)) * 100

negative_per_biden = (count_biden.loc["negative", "Sentiment Polarity"] / len(biden_reviews)) * 100
positive_per_biden = (count_biden.loc["positive", "Sentiment Polarity"] / len(biden_reviews)) * 100

Politicians = ['Joe Biden', 'Donald Trump']
lis_pos = [positive_per_biden, positive_per_trump]
lis_neg = [negative_per_biden, negative_per_trump]

# Plot results
fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])
fig.update_layout(barmode='group', title="Sentiment Analysis of US Election Tweets")
fig.show()

# Optional: Save the plot as HTML
fig.write_html("us_election_sentiment_analysis.html")

print("\nAnalysis complete. See the interactive plot for results.")