# Установите необходимые библиотеки:
# pip install transformers torch pandas tabulate

import time
import pandas as pd
from transformers import pipeline
from tabulate import tabulate

import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_key_serg")

# ==================== 1. ПОДГОТОВКА ТЕСТОВЫХ ПРИМЕРОВ НА РУССКОМ ЯЗЫКЕ ====================
# Список кортежей (текст, ожидаемая метка в формате POSITIVE/NEGATIVE/NEUTRAL)
test_samples = [
    ("Фильм просто потрясающий! Лучшее, что я видел в этом году.", "POSITIVE"),
    ("Товар сломался на второй день. Полное разочарование.", "NEGATIVE"),
    ("Обычная гостиница, ничего особенного.", "NEUTRAL"),
    ("Обслуживание на высшем уровне, персонал очень вежливый.", "POSITIVE"),
    ("Отвратительное качество, деньги на ветер.", "NEGATIVE"),
    ("Нормально, работает, но без восторга.", "NEUTRAL"),
    ("Превысил все ожидания! Рекомендую каждому.", "POSITIVE"),
    ("Не соответствует цене. Дешевые материалы.", "NEGATIVE"),
    ("Средний ресторан, еда как везде.", "NEUTRAL"),
    ("Лучшая покупка в моей жизни! Счастлив безмерно.", "POSITIVE"),
    ("Персонал был груб и не помогал.", "NEGATIVE"),
    ("Ничего особенного, но свою функцию выполняет.", "NEUTRAL"),
    ("Гениально! Абсолютный восторг!", "POSITIVE"),
    ("Полный мусор. Обходите стороной.", "NEGATIVE"),
    ("Приемлемо, учитывая цену.", "NEUTRAL"),
]


# ==================== 2. ФУНКЦИЯ НОРМАЛИЗАЦИИ МЕТОК ====================
def normalize_label(raw_label: str) -> str:
    """
    Приводит выходной лейбл модели к одному из: POSITIVE, NEGATIVE, NEUTRAL.
    """
    raw_label = raw_label.upper()
    if "POS" in raw_label or "LABEL_1" in raw_label or raw_label == "1":
        return "POSITIVE"
    elif "NEG" in raw_label or "LABEL_0" in raw_label or raw_label == "0":
        return "NEGATIVE"
    elif "NEU" in raw_label or "NEUTRAL" in raw_label:
        return "NEUTRAL"
    else:
        return "UNKNOWN"


# ==================== 3. ТЕСТИРОВАНИЕ ОДНОЙ МОДЕЛИ ====================
def test_model(model_id: str, samples, num_runs: int = 1):
    """
    Загружает pipeline, прогоняет samples, замеряет время и accuracy.
    """
    print(f"\n--- Тестирование модели: {model_id} ---")
    print("Загрузка модели (первый запуск может занять время)...")

    try:
        pipe = pipeline("sentiment-analysis", model=model_id, token=HF_TOKEN,)
    except Exception as e:
        print(f"Ошибка загрузки модели {model_id}: {e}")
        return None

    # Прогрев
    _ = pipe("Разогрев")

    times = []
    predictions = []
    ground_truths = []

    print("Прогон тестовых примеров...")
    for text, true_label in samples:
        start_time = time.perf_counter()
        try:
            result = pipe(text)[0]
            pred_raw = result['label']
            pred_norm = normalize_label(pred_raw)
        except Exception as e:
            print(f"Ошибка при обработке текста '{text[:50]}...': {e}")
            pred_norm = "ERROR"
        end_time = time.perf_counter()

        inference_time = end_time - start_time
        times.append(inference_time)
        predictions.append(pred_norm)
        ground_truths.append(true_label)

    # Подсчёт accuracy (игнорируем ERROR и UNKNOWN)
    correct = 0
    total = 0
    for pred, true in zip(predictions, ground_truths):
        if pred in ("ERROR", "UNKNOWN"):
            continue
        total += 1
        if pred == true:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    results = {
        "model_id": model_id,
        "avg_inference_time_ms": avg_time * 1000,
        "total_inference_time_s": total_time,
        "accuracy": accuracy,
        "num_valid_samples": total,
        "times_per_sample_ms": [t * 1000 for t in times],
        "predictions": predictions,
        "ground_truths": ground_truths
    }
    return results


# ==================== 4. ЗАПУСК ДЛЯ ТРЁХ РУССКОЯЗЫЧНЫХ МОДЕЛЕЙ ====================
def main():
    # Выбираем модели, которые поддерживают русский язык
    models_to_test = [
        "blanchefort/rubert-base-cased-sentiment",  # POSITIVE/NEGATIVE/NEUTRAL
        "cointegrated/rubert-tiny2",  # POSITIVE/NEGATIVE (точность ниже, но быстрая)
        "cointegrated/rubert-tiny-sentiment-balanced"  # POSITIVE/NEGATIVE/NEUTRAL
    ]

    all_results = []

    for model in models_to_test:
        res = test_model(model, test_samples)
        if res:
            all_results.append(res)

    # ==================== 5. ВЫВОД РЕЗУЛЬТАТОВ ====================
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ (русские примеры)")
    print("=" * 80)

    summary_data = []
    for r in all_results:
        summary_data.append([
            r["model_id"],
            f"{r['avg_inference_time_ms']:.2f} мс",
            f"{r['total_inference_time_s']:.3f} с",
            f"{r['accuracy'] * 100:.1f}%",
            r["num_valid_samples"]
        ])

    print(tabulate(summary_data,
                   headers=["Модель", "Среднее время/пример", "Общее время", "Точность (Accuracy)", "Учтено примеров"],
                   tablefmt="grid"))

    # Детальные результаты по каждому примеру
    for r in all_results:
        print(f"\n--- Детали для модели: {r['model_id']} ---")
        detail_data = []
        for i, sample in enumerate(test_samples):
            sample_text, true_label = sample  # распаковываем текст и истинную метку
            pred = r["predictions"][i]
            time_ms = r["times_per_sample_ms"][i]
            detail_data.append([
                i + 1,
                sample_text[:55] + ("..." if len(sample_text) > 55 else ""),
                true_label,
                pred,
                "✓" if pred == true_label else "✗",
                f"{time_ms:.2f} мс"
            ])
        print(tabulate(detail_data,
                       headers=["#", "Текст (первые 55 симв.)", "Истинная метка", "Предсказание", "Совпадение",
                                "Время"],
                       tablefmt="simple"))
    # Победители
    print("\n" + "=" * 80)
    if all_results:
        best_acc = max(all_results, key=lambda x: x["accuracy"])
        fastest = min(all_results, key=lambda x: x["avg_inference_time_ms"])
        print(f"🏆 Лучшая точность: {best_acc['model_id']} ({best_acc['accuracy'] * 100:.1f}%)")
        print(f"⚡ Самый быстрый инференс: {fastest['model_id']} ({fastest['avg_inference_time_ms']:.2f} мс/пример)")
    else:
        print("Не удалось протестировать ни одной модели.")


if __name__ == "__main__":
    main()
