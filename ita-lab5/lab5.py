import re
import json
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# 1. Підготовка вхідних даних (чистий та зашумлений текст)
clean_file = 'ocr_result_clean.txt'
degraded_file = 'ocr_result_degraded.txt'

# Якщо файли ще не створені — підтягуємо з lab 4 або генеруємо
if not os.path.exists(clean_file):
    if os.path.exists('ocr_result.txt'):
        with open('ocr_result.txt', 'r', encoding='utf-8') as src:
            clean_content = src.read()
    else:
        clean_content = (
            "ВЕБ ЧЕК\nФОП ПЕРЕМОГА Б.Д.\nмагазин\n"
            "УКРАЇНА, М.КИЇВ, СВЯТОШИНСЬКИЙ Р-Н проспект Перемоги, 67\n"
            "ПН 385381222511\nІД 38538312\n"
            "1.000 x 2025.04\n2208601100\n4820024210058\nABRO546581\n"
            "Товар з ПДВ та кодом УКТЗЕД\nАкциз(шт.) 2025.04 ГА\n"
            "ГОТІВКА 2030.00 грн\nСУМА 2025.04 грн\nДО СПЛАТИ 2025.00 грн"
        )
    with open(clean_file, 'w', encoding='utf-8') as f:
        f.write(clean_content)

# Деградований текст з аномаліями Варіанту 9 (склеювання кількості, підміна роздільників @)
if not os.path.exists(degraded_file):
    degraded_content = (
        "ВEБ 4EK\nФОП ПEРEMOГA Б.Д.\nмaгaзин\n"
        "УКРА1НА, М.КИ1В, СВЯТ0ШИНСЬКИЙ Р-Н пр-т Перемоги, 67\n"
        "ПН 385381222511\n1Д 38538312\n"
        "1.000@2025.04\n2208601100\n1.000x4820024210058\nABR0546581\n"
        "Товар з ПДВ та к0дом УКТЗЕД\n"
        "Г0Т1ВКА 2030.00 грн\nCYMA 2025.04 грн\nДО СПЛАТИ 2025.00 грн"
    )
    with open(degraded_file, 'w', encoding='utf-8') as f:
        f.write(degraded_content)

with open(clean_file, 'r', encoding='utf-8') as f:
    clean_text = f.read()

with open(degraded_file, 'r', encoding='utf-8') as f:
    noisy_text = f.read()

print("[+] Файли успішно завантажено!")


# КРОК 2. RegEx аналіз для Варіанту 9 (Штрихкоди та Артикули)
print("\n" + "=" * 50)
print("[*] КРОК 2: Дослідження RegEx (Варіант 9)")
print("=" * 50)

# Базовий наївний шаблон (очікує окремий ізольований числовий штрихкод)
naive_barcode_pattern = r'\b\d{10,14}\b'

# Комплексний адаптивний RegEx для варіанту 9:
# 1) Ізолює штрихкод, навіть якщо він склеєний з кількістю через 'x', '*', '@' або пробіли: (?:\d+(?:\.\d+)?[\s*xX@]+)?
# 2) Ловить як чисто цифрові коди (EAN-13, УКТЗЕД: 10-14 цифр), так і літерно-цифрові артикули (наприклад, ABRO546581)
robust_sku_pattern = r'(?:(?:\d+(?:\.\d+)?\s*[*xX@]\s*)|(?:\b))([A-ZА-Я0-9]{2,6}\d{4,10}|\d{10,14})\b'

print("[!] Наївний RegEx на чистому тексті:", re.findall(naive_barcode_pattern, clean_text))
print("[!] Наївний RegEx на зашумленому тексті (збій склеювання):", re.findall(naive_barcode_pattern, noisy_text))

clean_skus = re.findall(robust_sku_pattern, clean_text)
noisy_skus = re.findall(robust_sku_pattern, noisy_text)
print("\n[+] Адаптивний RegEx на чистому тексті:", clean_skus)
print("[+] Адаптивний RegEx на зашумленому тексті:", noisy_skus)

# КРОК 3 & 4. WordPiece токенізація та NER (без агрегації vs з агрегацією)
print("\n" + "=" * 50)
print("[*] КРОК 3 & 4: WordPiece токенізація (Babelscape/wikineural-multilingual-ner)")
print("=" * 50)

primary_model_name = "Babelscape/wikineural-multilingual-ner"

# 1. Сирий вихід без агрегації (демонстрація ## префіксів та BIO розмітки)
ner_raw = pipeline("ner", model=primary_model_name, aggregation_strategy=None)
raw_tokens = ner_raw(clean_text[:120])

print("Приклад сабтокенізації WordPiece (##-префікси та BIO-теги):")
for tok in raw_tokens[:8]:
    print(f"  Токен: {tok['word']:<12} | Тег: {tok['entity']:<7} | Score: {tok['score']:.3f}")

# 2. Вилучення з простою агрегацією
ner_simple = pipeline("ner", model=primary_model_name, aggregation_strategy="simple")
entities_clean_main = ner_simple(clean_text)
entities_noisy_main = ner_simple(noisy_text)

print(f"\nЗнайдено сутностей у чистому тексті: {len(entities_clean_main)}")
for ent in entities_clean_main:
    print(f"  [{ent['entity_group']}] {ent['word']} (впевненість: {ent['score']:.2f})")


# КРОК 5 & 6. Порівняльний аналіз трьох моделей енкодерів
print("\n" + "=" * 50)
print("[*] КРОК 5 & 6: Порівняння 3-х моделей трансформерів")
print("=" * 50)

models_to_test = [
    "Babelscape/wikineural-multilingual-ner",
    "dslim/bert-base-NER",
    "elastic/distilbert-base-cased-finetuned-conll03-english"
]

benchmark_results = []
all_extracted = {}

for model_name in models_to_test:
    print(f"[*] Тестування моделі: {model_name}...")
    pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")

    clean_ents = pipe(clean_text)
    noisy_ents = pipe(noisy_text)

    all_extracted[model_name] = {"clean": clean_ents, "noisy": noisy_ents}

    # 1. Preservation Rate (Коефіцієнт збереження)
    clean_words = {e['word'].strip().lower() for e in clean_ents}
    noisy_words = {e['word'].strip().lower() for e in noisy_ents}
    preserved = clean_words.intersection(noisy_words)
    preservation_rate = (len(preserved) / len(clean_words) * 100) if clean_words else 0.0

    # 2. Confidence Delta (Падіння впевненості)
    avg_conf_clean = np.mean([e['score'] for e in clean_ents]) if clean_ents else 0.0
    avg_conf_noisy = np.mean([e['score'] for e in noisy_ents]) if noisy_ents else 0.0
    conf_delta = max(0.0, avg_conf_clean - avg_conf_noisy)

    # 3. Class Drift (Мутація класів)
    clean_dict = {e['word'].strip().lower(): e['entity_group'] for e in clean_ents}
    drift_count = 0
    for e in noisy_ents:
        w = e['word'].strip().lower()
        if w in clean_dict and clean_dict[w] != e['entity_group']:
            drift_count += 1

    benchmark_results.append({
        "Model": model_name.split("/")[-1],
        "Clean Count": len(clean_ents),
        "Noisy Count": len(noisy_ents),
        "Preservation (%)": round(preservation_rate, 2),
        "Conf Delta": round(conf_delta, 4),
        "Class Drift": drift_count
    })

df_metrics = pd.DataFrame(benchmark_results)
print("\n" + df_metrics.to_string(index=False))

# КРОК 7. Відбір еталонної моделі та експорт у ner_extracted.json
# Критерій: максимальний Preservation Rate -> мінімальний Conf Delta -> мінімальний Class Drift
best_idx = 0
best_model_name = models_to_test[0]
best_score = (-1, 999, 999)

for idx, res in enumerate(benchmark_results):
    score = (res["Preservation (%)"], -res["Conf Delta"], -res["Class Drift"])
    if score > best_score:
        best_score = score
        best_idx = idx
        best_model_name = models_to_test[idx]

print(f"\n[+] Еталонна модель: {best_model_name}")

best_clean_ents = all_extracted[best_model_name]["clean"]

json_output = {
    "document_metadata": {
        "source_file": clean_file,
        "model_name": best_model_name,
        "aggregation_strategy": "simple",
        "total_entities_found": len(best_clean_ents)
    },
    "entities": [
        {
            "entity_group": ent["entity_group"],
            "word": ent["word"],
            "score": round(float(ent["score"]), 4),
            "start": ent["start"],
            "end": ent["end"]
        }
        for ent in best_clean_ents
    ]
}

output_json_path = "ner_extracted.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print(f"[+] Результат успішно записано у файл: {output_json_path}")
print(json.dumps(json_output, ensure_ascii=False, indent=2))