#!/usr/bin/env python3
"""
Скрипт для демонстрации работы с LLM API через OpenAI SDK.
Поддерживает два провайдера: Ollama (локально) и OpenRouter.
Требуется установить: openai, python-dotenv.
Для Ollama: запущенный сервер на http://localhost:11434
Для OpenRouter: ключ в .env (OPENROUTER_API_KEY)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Конфигурация провайдеров
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # не требуется, но клиент ожидает непустое значение
        "default_model": "qwen3.5:9b",  # или любая другая установленная модель
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "default_model": "openai/gpt-oss-120b:free",
    }
}

def get_client(provider: str) -> tuple[OpenAI, str]:
    """Возвращает настроенный клиент OpenAI и имя модели по умолчанию для выбранного провайдера."""
    if provider not in PROVIDERS:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}. Доступны: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    if not config["api_key"]:
        raise ValueError(f"API ключ для {provider} не найден. Проверьте .env файл (переменная OPENROUTER_API_KEY)")

    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    return client, config["default_model"]

def basic_completion(client: OpenAI, model: str):
    """Демонстрация базового запроса с system и user сообщениями."""
    print("\n=== Базовый запрос (system + user) ===")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты — полезный ассистент, отвечающий кратко и по делу."},
            {"role": "user", "content": "Что такое API? Ответь в одном предложении."}
        ]
    )
    print(f"Ответ:\n{response.choices[0].message.content}\n")

def temperature_experiment(client: OpenAI, model: str):
    """Эксперимент с разными значениями температуры: 0, 0.7, 1.5.
    Выводит ответы и объясняет разницу."""
    prompt = "Отвечай кратко и на русском языке. Придумай одно оригинальное название для стартапа в сфере ИИ."

    print("\n=== Эксперимент с температурой ===")
    print("Температура контролирует случайность ответа: низкая → детерминированный выбор, высокая → креативность/хаос.\n")

    for temp in [0.0, 0.7, 1.5]:
        print(f"--- Температура = {temp} ---")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        print(f"Ответ: {response.choices[0].message.content}\n")

    print("Ожидаемые различия:")
    print("- При temp=0 ответ будет почти одинаков при каждом запуске (низкая дивергенция).")
    print("- При temp=0.7 баланс креативности и связности, ответы разнообразны, но осмысленны.")
    print("- При temp=1.5 ответы могут быть сюрреалистичными, несвязными или повторяющимися (высокая энтропия).\n")

def streaming_completion(client: OpenAI, model: str):
    """Демонстрация потоковой выдачи (streaming)."""
    print("\n=== Потоковый запрос (stream=True) ===")
    print("Ассистент печатает ответ по частям:")

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Назови три достоинства языка Python, каждое с новой строки."}
        ],
        stream=True
    )

    collected_chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            collected_chunks.append(content)
    print("\n")  # финальный перевод строки
    full_response = "".join(collected_chunks)
    print(f"Полный ответ собран из {len(collected_chunks)} фрагментов.\n")
 

def main():
    # Выберите провайдера по умолчанию: 'ollama' или 'openrouter'
    # Для смены просто измените строку ниже
    DEFAULT_PROVIDER = "ollama" #"openrouter"  # или "openrouter""ollama"


    try:
        client, model = get_client(DEFAULT_PROVIDER)
        print(f"Используется провайдер: {DEFAULT_PROVIDER}, модель: {model}")
    except Exception as e:
        print(f"Не удалось инициализировать клиент: {e}")
        return

    # Демонстрация возможностей
    basic_completion(client, model)
    temperature_experiment(client, model)
    streaming_completion(client, model)
    #switch_provider_example()

if __name__ == "__main__":
    main()
