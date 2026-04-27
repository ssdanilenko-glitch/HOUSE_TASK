def calculate_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_m: float,   # Цена за 1M input-токенов
    output_price_per_m: float,  # Цена за 1M output-токенов
    cache_ratio: float = 0.0,   # Доля кэшированных input-токенов (0.0–1.0)
    cache_discount: float = 0.9 # Скидка на кэшированные токены (0.9 = 90%)
) -> dict:
    """Калькулятор стоимости ИИ-проекта"""
    days_in_month = 30

    # Общее количество токенов в месяц
    total_input = requests_per_day * avg_input_tokens * days_in_month
    total_output = requests_per_day * avg_output_tokens * days_in_month

    # Стоимость input с учётом кэширования
    cached_input = total_input * cache_ratio
    uncached_input = total_input * (1 - cache_ratio)
    input_cost = (
        uncached_input * input_price_per_m / 1_000_000 +
        cached_input * input_price_per_m * (1 - cache_discount) / 1_000_000
    )

    # Стоимость output (не кэшируется)
    output_cost = total_output * output_price_per_m / 1_000_000

    total = input_cost + output_cost

    return {
        "requests_month": requests_per_day * days_in_month,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost": round(input_cost, 2),
        "output_cost": round(output_cost, 2),
        "total_monthly": round(total, 2),
        "cost_per_request": round(total / (requests_per_day * days_in_month), 6)
    }


# Сценарий: чат-бот с 1000 запросов/день
chatbot = calculate_cost(
    requests_per_day=100,
    avg_input_tokens=800,   # system prompt + вопрос пользователя
    avg_output_tokens=400,  # ответ модели
    input_price_per_m=0.28, # DeepSeek V3.2
    output_price_per_m=0.42,
    cache_ratio=0.7,        # 70% input — это system prompt (кэшируется)
    cache_discount=0.9      # 90% скидка на кэш у DeepSeek
)


print("Чат-бот на DeepSeek V3.2:")
print(f"  Запросов/месяц: {chatbot['requests_month']:,}")
print(f"  Input: ${chatbot['input_cost']}")
print(f"  Output: ${chatbot['output_cost']}")
print(f"  Итого: ${chatbot['total_monthly']}/месяц")
print(f"  За один запрос: ${chatbot['cost_per_request']}")
# Итого: ~$7.53/месяц за 30 000 запросов
