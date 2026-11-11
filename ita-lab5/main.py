import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# 1. Завантаження та попередня обробка
IMAGE_PATH = 'receipt.jpg'
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError("Файл receipt.jpg не знайдено! Покладіть його в папку з проєктом.")

# Переводимо у відтінки сірого для EasyOCR та у RGB для відображення
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Ініціалізація OCR
print("[*] Ініціалізація EasyOCR (uk, en)...")
reader = easyocr.Reader(['uk', 'en'], gpu=False)

print("[*] Розпізнавання оригінального зображення...")
results = reader.readtext(gray_img)
print(f"[+] Знайдено блоків: {len(results)}")

# 3. Візуалізація рамок та експорт тексту
output_img = img_rgb.copy()
extracted_text_lines = []

for bbox, text, confidence in results:
    top_left = tuple([int(val) for val in bbox[0]])
    bottom_right = tuple([int(val) for val in bbox[2]])
    cv2.rectangle(output_img, top_left, bottom_right, (0, 255, 0), 2)

    if confidence > 0.3:
        extracted_text_lines.append(text)
        print(f"Текст: {text} | Впевненість: {confidence:.2f}")

# Збереження сирого тексту для Лаби 5
full_raw_text = "\n".join(extracted_text_lines)
with open("ocr_result.txt", "w", encoding="utf-8") as file:
    file.write(full_raw_text)
print("\n[+] Сирий текст збережено у файл 'ocr_result.txt'")

# Відображення оригіналу з рамками
plt.figure(figsize=(9, 9))
plt.imshow(output_img)
plt.axis('off')
plt.title("Результат сегментації OCR (Оригінал)")
plt.show()

# 4. ВАРІАНТ 9 (Гаусівський шум)
print("\n" + "=" * 45)
print("Початок дослідження Гаусівського шуму (Варіант 9)")
print("=" * 45)

initial_word_count = len(results)
initial_confidence = sum([conf for _, _, conf in results]) / initial_word_count if initial_word_count > 0 else 0
print(f"Базові показники: слів = {initial_word_count}, сер. впевненість = {initial_confidence:.2f}")


def apply_gaussian_noise(image, sigma):
    """Накладає гаусівський шум на чорно-білу матрицю"""
    mean = 0
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Параметри згідно з методичкою для 9 варіанту
sigma = 10
step = 10
threshold_conf = 0.40
threshold_words = initial_word_count * 0.50

history = []

while True:
    noisy_img = apply_gaussian_noise(gray_img, sigma)
    res_noisy = reader.readtext(noisy_img)

    words_count = len(res_noisy)
    avg_conf = sum([conf for _, _, conf in res_noisy]) / words_count if words_count > 0 else 0.0

    history.append((sigma, words_count, avg_conf, noisy_img))
    print(f"Параметр sigma: {sigma} | Знайдено слів: {words_count} | Сер. впевненість: {avg_conf:.2f}")

    # Критерій зупинки: впевненість < 0.4 або кількість слів зменшилась на 50%
    if avg_conf < threshold_conf or words_count <= threshold_words or sigma >= 100:
        print(f"\n[!] Досягнуто критерію зупинки при sigma = {sigma}")
        break

    sigma += step

# Візуалізація фінального зашумленого зображення
last_sigma, last_words, last_conf, last_img = history[-1]
plt.figure(figsize=(8, 8))
plt.imshow(last_img, cmap='gray')
plt.title(f"Гаусівський шум (sigma={last_sigma})\nСлів: {last_words}, Впевненість: {last_conf:.2f}")
plt.axis('off')
plt.show()