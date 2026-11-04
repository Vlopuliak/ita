import re
import pandas as pd
from playwright.sync_api import sync_playwright, Response, Route, TimeoutError

# Глобальний масив для накопичення зібраних автозапчастин
collected_items = []


def extract_oem_and_compat(title: str, item_id: str = "", brand: str = ""):
    """Виділяє OEM/артикул деталі та марки сумісних авто з назви товару."""
    paren_matches = re.findall(r"\(([^()]+)\)", title)
    oem = paren_matches[-1].strip() if paren_matches else str(item_id)
    if not oem or len(oem) < 3:
        oem = str(item_id) if item_id else "OEM-STD"

    car_brands = re.findall(
        r"\b(VW|Volkswagen|Audi|Skoda|Seat|BMW|Mercedes|Ford|Renault|Dacia|Toyota|"
        r"Lexus|Opel|Chevrolet|Daewoo|Lanos|Sens|ВАЗ|ЗАЗ|Нива|Таврія|Honda|Mazda|"
        r"Nissan|Hyundai|Kia|Peugeot|Citroen|Fiat|Volvo|Subaru|Mitsubishi)\b",
        title,
        re.IGNORECASE,
    )
    if car_brands:
        compat = ", ".join(
            dict.fromkeys(
                b.upper() if len(b) <= 3 else b.capitalize() for b in car_brands
            )
        )
    elif brand:
        compat = f"Універсальна ({brand} OEM)"
    else:
        compat = "Універсальна (за каталогом OEM)"

    return oem, compat


# Крок 3. Оптимізація ресурсів та блокування медіа-трафіку (Route Abort)
def route_filter(route: Route):
    blocked_types = ["image", "media", "font"]
    if route.request.resource_type in blocked_types:
        route.abort()
    else:
        route.continue_()


# Крок 4. Асинхронне перехоплення API-відповідей (Network Interception)
def handle_network_response(response: Response):
    target_patterns = ["api/product/details", "goods/get", "api/v", "graphql"]
    if any(p in response.url for p in target_patterns) and response.status == 200:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = response.json()
                raw_items = payload.get("data", [])
                if isinstance(raw_items, dict):
                    raw_items = raw_items.get("goods", []) or raw_items.get("items", [])

                if isinstance(raw_items, list) and len(raw_items) > 0:
                    print(f"[+] Перехоплено чергову пачку: {len(raw_items)} товарів з API")
                    for item in raw_items:
                        if not isinstance(item, dict):
                            continue
                        title = item.get("title") or item.get("name") or ""
                        item_id = str(item.get("id") or "")
                        brand = str(item.get("brand") or "")
                        price = item.get("price")
                        if title and price is not None:
                            oem, compat = extract_oem_and_compat(title, item_id, brand)
                            collected_items.append({
                                "id": item_id or oem,
                                "Назва деталі": title.strip(),
                                "OEM/Артикул запчастини": oem,
                                "Сумісність": compat,
                                "Ціна": f"{price} грн",
                            })
            except Exception as err:
                print(f"[-] Помилка парсингу JSON: {err}")


# Резервне вилучення карток через семантично стійкі локатори Playwright
def collect_visible_cards(page):
    tiles = page.locator("rz-catalog-tile, .catalog-grid__cell").all()
    added = 0
    for tile in tiles:
        try:
            title_el = tile.locator(".goods-tile__title, a.goods-tile__heading").first
            price_el = tile.locator(".goods-tile__price-value").first
            if not title_el.is_visible():
                continue
            title = title_el.inner_text().strip()
            price_raw = price_el.inner_text().strip() if price_el.is_visible() else ""
            price_num = re.sub(r"[^\d]", "", price_raw)
            price_str = f"{price_num} грн" if price_num else "Ціну уточнюйте"

            goods_id_el = tile.locator("[data-goods-id]").first
            goods_id = (
                goods_id_el.get_attribute("data-goods-id")
                if goods_id_el.count() > 0
                else ""
            )
            oem, compat = extract_oem_and_compat(title, goods_id or "")

            collected_items.append({
                "id": goods_id or oem,
                "Назва деталі": title,
                "OEM/Артикул запчастини": oem,
                "Сумісність": compat,
                "Ціна": price_str,
            })
            added += 1
        except Exception:
            continue
    if added > 0:
        print(f"[+] Перехоплено чергову пачку: {added} товарів з каталогу")


def main():
    start_url = "https://auto.rozetka.com.ua/ua/svechi-zagiganiya/c4642656/"
    target_csv = "autoparts.csv"

    with sync_playwright() as p:
        # Крок 5. Налаштування контексту браузера та маскування автоматизації
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        page.route("**/*", route_filter)
        page.on("response", handle_network_response)

        # Крок 6. Ініціалізація навігації з обробкою винятків TimeoutError
        print(f"[*] Відкриваємо каталог: {start_url}")
        try:
            page.goto(start_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
        except TimeoutError as err:
            print(f"[!] Системний виняток TimeoutError під час відкриття сторінки: {err}")

        # Крок 7. Емуляція плавного скролу для тригера Lazy Loading
        print("[*] Поступовий скрол сторінки вниз для виклику XHR-запитів...")
        for _ in range(6):
            page.evaluate("window.scrollBy(0, 900)")
            page.wait_for_timeout(1500)

        page.evaluate("window.scrollTo(0, document.body.scrollHeight - 800)")
        page.wait_for_timeout(2000)
        collect_visible_cards(page)

        # Крок 8. Взаємодія з пагінацією через механізм Actionability Checks
        show_more_btn = page.locator(
            "button:has-text('Показати ще'), a:has-text('Показати ще'), .show-more"
        ).first

        if show_more_btn.is_visible():
            print("[*] Клікаємо на 'Показати ще' для довантаження наступної сторінки...")
            show_more_btn.click()
            page.wait_for_timeout(3000)
            for _ in range(5):
                page.evaluate("window.scrollBy(0, 900)")
                page.wait_for_timeout(1500)
            collect_visible_cards(page)
        else:
            print("[!] Кнопку 'Показати ще' не знайдено, перевіряємо перемикач сторінок...")
            page_2_btn = (
                page.locator("a[href*='page=2'], a[href*='page-2']")
                .or_(page.get_by_role("link", name="2"))
                .first
            )
            if page_2_btn.is_visible():
                print("[*] Клікаємо на сторінку 2...")
                page_2_btn.click()
                page.wait_for_timeout(4000)
                for _ in range(5):
                    page.evaluate("window.scrollBy(0, 900)")
                    page.wait_for_timeout(1500)
                collect_visible_cards(page)

        browser.close()

    # Крок 9. Усунення дублікатів та експорт даних у структурований CSV
    if collected_items:
        df = pd.DataFrame(collected_items)
        df.dropna(subset=["Назва деталі"], inplace=True)
        df.drop_duplicates(subset=["id"], inplace=True)

        columns_order = [
            "Назва деталі",
            "OEM/Артикул запчастини",
            "Сумісність",
            "Ціна",
        ]
        df = df[columns_order]
        df.to_csv(target_csv, index=False, encoding="utf-8-sig")

        print(f"\n[✓] Успішно зібрано унікальних записів: {len(df)}")
        print(f"[✓] Артефакт збережено у файл: {target_csv}")


if __name__ == "__main__":
    main()