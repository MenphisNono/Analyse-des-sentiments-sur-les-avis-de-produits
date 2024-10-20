import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def analyze_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return 'positif'
    elif blob.sentiment.polarity == 0:
        return 'neutre'
    else:
        return 'négatif'

def main():
    # Chargement des données d'avis
    reviews = pd.read_csv('data/reviews.csv')
    reviews['sentiment'] = reviews['review_text'].apply(analyze_sentiment)

    # Visualisation des résultats
    sentiment_counts = reviews['sentiment'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Répartition des sentiments dans les avis")
    plt.savefig('visuals/sentiment_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()