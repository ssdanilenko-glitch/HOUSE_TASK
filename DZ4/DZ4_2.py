import ollama
import time

from Evel_tasks import EVAL_TASKS



def evaluate_model(model: str) -> dict:
    """Прогоняем модель через набор задач и считаем accuracy"""
    passed = 0
    details = []

    for task in EVAL_TASKS:
        start = time.perf_counter()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            stream=True
        )

        ttft = None
        tokens = 0
        full_answer = ""

        for chunk in response:
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens += 1
            # Предполагаем, что chunk имеет структуру: {"message": {"content": "..."}}
            full_answer += chunk["message"]["content"]

        total = time.perf_counter() - start
        gen_time = total - ttft if ttft else total
        throughput = tokens / gen_time if gen_time > 0 else 0

        success = task["check"](full_answer)

        if success:
            passed += 1
#        details.append(f"  {'✓' if success else '✗'} {task['name']}")

        details.append({
            "name": task["name"],
            "success": success,
            "ttft": ttft,
            "total": total,
            "tokens": tokens,
            "throughput": throughput
        })

    accuracy = passed / len(EVAL_TASKS) * 100
    return {"model": model, "accuracy": accuracy, "details": details}


# Оценка
#result = evaluate_model("qwen3:8b")
#result = evaluate_model("gemma4:e4b")
# Усредняем результаты

# Сравниваем 2 модели
models = ["qwen3:8b", "gemma4:e4b","qwen3.5:9b"]
runs = len(EVAL_TASKS)
#print("Бенчмарк моделей (среднее за 3 прогона):\n")
for model in models:
    results = evaluate_model(model)
    metrics = results["details"]

    print(f"{results['model']:20s} | TTFT: {round(sum(r["ttft"] for r in metrics) / runs, 3):.3f}s | "
    f"Total: {round(sum(r["total"] for r in metrics) / runs, 3):.2f}s | {round(sum(r["throughput"] for r in metrics) / runs, 1):.1f} tok/s")

    print(f"\n{results['model']}: {results['accuracy']:.0f}% ({int(results['accuracy']/20)}/{len(EVAL_TASKS)})")
    #for detail in results["details"]:
    #    print(detail)

