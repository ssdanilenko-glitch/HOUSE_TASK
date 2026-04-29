from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_key_S"))

# Поиск моделей для генерации текста, отсортированных по скачиваниям
print("Топ-10 моделей для sentiment text-classification:\n")
models = api.list_models(
    pipeline_tag="text-classification",
    sort="downloads",
    limit=10
)

for model in models:
    print(f"  {model.id}")
    print(f"    Скачиваний: {model.downloads:,}")
    print(f"    Лайков: {model.likes:,}")
    print(f"    Теги: {', '.join(model.tags[:5]) if model.tags else 'нет'}")
    print(f"    Лицензия: {model.card_data.get('license', 'не указана') if model.card_data else 'нет данных'}")
    print()

# Получаем детальную информацию о конкретной модели
print("="*50)
print("Детали модели cardiffnlp/twitter-roberta-base-sentiment-latest:\n")
info = api.model_info("cardiffnlp/twitter-roberta-base-sentiment-latest")
print(f"  ID: {info.id}")
print(f"  Автор: {info.author}")
print(f"  Скачиваний: {info.downloads:,}")
print(f"  Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
print(f"  Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
print(f"  Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")
