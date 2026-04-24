# pip install ollama tabulate

import time

import ollama


# Модели для сравнения (скачайте заранее: ollama pull <model>)
models = [
    "qwen3.5:9b",   # Большая модель
    "gemma4:e4b",   # средняя
    "qwen3:8b",# Большая модель
]


# Тестовые задачи
tasks = [
    {
        "name": "Код",
        "prompt": "Напиши Python-функцию, которая находит все дубликаты в списке и делает замену на 1 найденный элемент. "
                  "Верни только код, без объяснений."
    },
    {
        "name": "Русский текст",
        "prompt": "Объясни в 2 предложениях, чем отличаются языки программирования С# от Python."
    },
    {
        "name": "Анализ",
        "prompt": "Клиент написал: 'Заказ пришёл целый, но с опозданием на неделю. Прошу учесть при следующем заказе.' "
                  "Определи тональность (positive/negative/neutral) и срочность (high/medium/low)."
    }
]


# Запуск бенчмарка
for model in models:
    print(f"\n{'=' * 60}")
    print(f"Модель: {model}")
    print(f"{'=' * 60}")

    for task in tasks:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}]
        )
        elapsed = time.time() - start
        tokens = response.get("eval_count", 0)
        speed = tokens / elapsed if elapsed > 0 else 0

        # Первые 100 символов ответа
        answer_preview = response["message"]["content"][:300].replace("\n", " ")
        print(f"\n  [{task['name']}]")
        print(f"  Время: {elapsed:.1f}с | Токены: {tokens} | Скорость: {speed:.1f} tok/s")
        print(f"  Ответ: {answer_preview}...")
