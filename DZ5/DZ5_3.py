from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_key_serg")

# Анализ тональности
sentiment = pipeline(
    "sentiment-analysis",
    model="cointegrated/rubert-tiny-sentiment-balanced",
    token=HF_TOKEN,
)

sentiment1 = pipeline(
    "sentiment-analysis",
    model="ZombitX64/MultiSent-E5-Pro",
    token=HF_TOKEN,
)

sentiment2 = pipeline(
    "sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis",
    token=HF_TOKEN,
)


reviews = [
    "Отличный сервис, молодцы ИТ служба!",
    "Затянули по времени, но сделали нормально",
    "Ничего не сделали, необходимо переделать"
]
for review in reviews:
    res = sentiment(review)[0]
    print(f"  {res['label']:>10s} ({res['score']:.0%}) — {review}")
for review in reviews:
    res = sentiment1(review)[0]
    print(f"  {res['label']:>10s} ({res['score']:.0%}) — {review}")
for review in reviews:
    res = sentiment2(review)[0]
    print(f"  {res['label']:>10s} ({res['score']:.0%}) — {review}")
