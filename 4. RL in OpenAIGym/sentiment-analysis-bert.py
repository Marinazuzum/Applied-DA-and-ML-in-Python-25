
from transformers import pipeline

# Загрузка модели для анализа тональности (поддерживает много языков)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-multilingual-cased")

text_ru = "Очень хорошая книга, но доставка была долгой."
text_de = "Sehr gutes Buch, aber Lieferung war langsam."

result_ru = sentiment_pipeline(text_ru)
result_de = sentiment_pipeline(text_de)

print(f"Русский: {result_ru}")
print(f"Немецкий: {result_de}")