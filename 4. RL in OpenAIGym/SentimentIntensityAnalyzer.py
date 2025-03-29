from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

text_ru = "Простите, но я опаздал на концерт"
text_de = "Entschuldigung, aber ich bin zu spät zum Konzert"


#text_ru = "Очень хорошая книга, но доставка была долгой."
#text_de = "Sehr gutes Buch, aber Lieferung war langsam."

score_ru = analyzer.polarity_scores(text_ru)
score_de = analyzer.polarity_scores(text_de)

print(f"Русский: {score_ru}")
print(f"Немецкий: {score_de}")
