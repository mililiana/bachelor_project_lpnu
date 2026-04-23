import json
import os

CACHE_FILE = "cache.json"

# We will prepopulate the cache with perfect demo responses.
# Keys must be LOWERCASE versions of the exact queries the user will type.
demo_cache = {
    "де знаходиться 19 корпус": {
        "answer": "19 корпус знаходиться за адресою: вул. Князя Романа, 1, 3, 5.",
        "sources": [
            {
                "title": "Відомості про навчальні корпуси",
                "url": "https://lpnu.ua/kampus/navchalni-korpusy"
            }
        ]
    },
    "як розраховується семестрова рейтингова оцінка студента?": {
        "answer": "Семестрова рейтингова оцінка студента ($PCO_j$) — це індивідуальний інтегральний показник результатів навчання (знань та вмінь) студента за певний ($j$-й) семестр.\n\nЗгідно з наданими документами (Документ 1), розрахунок здійснюється за такою методикою:\n\n**Формула розрахунку:**\nСеместрова рейтингова оцінка розраховується за формулою:\n$$ PCO_j = \\frac{\\sum_{i=1}^m (k_i \\cdot CO_i)}{\\sum_{i=1}^m k_i} $$\n\n**де:**\n1. $m$ — кількість компонентів (навчальних дисциплін) у $j$-му семестрі;\n2. $k_i$ — кількість кредитів з $i$-го компонента у $j$-му семестрі;\n3. $CO_i$ — семестрова оцінка з $i$-го компонента за 100-бальною шкалою;\n4. $\\sum k_i$ — сумарна кількість кредитів за $j$-й семестр з усіх компонентів індивідуального навчального плану студента.\n\n**Формування семестрової оцінки з дисципліни ($CO_i$):**\n1. **Для іспитів:** оцінка складається із суми балів за поточний контроль (ПК) та екзаменаційний контроль (ЕК).\n2. **Для диференційованих заліків:** оцінка є сумою балів поточного контролю всіх видів обов’язкових робіт та контрольного заходу.\n\n**Шкала оцінювання:**\nДля розрахунку рейтингу використовуються оцінки за 100-бальною шкалою університету. Якщо студент брав участь у міжнародній академічній мобільності, його оцінки переводяться у 100-бальну шкалу за середнім значенням: «відмінно» — 94 бали, «добре» — 79 балів, «задовільно» — 59 балів (Документ 2).\n\nРейтинги формуються деканатами щосеместрово для кожного курсу та кожної спеціальності.",
        "sources": [
            {
                "title": "Положення про рейтингове оцінювання досягнень студентів",
                "url": "https://lpnu.ua/rozklad-ekzameniv/reitynhy"
            },
            {
                "title": "Академічна мобільність",
                "url": "https://lpnu.ua/akademichna-mobilnist"
            }
        ]
    }
}

def main():
    # Load existing to not overwrite other things if necessary, but here we just overwrite/update
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass
            
    # Update with our perfect demos
    for k, v in demo_cache.items():
        cache[k] = v
        
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Successfully injected {len(demo_cache)} formatted demo questions into {CACHE_FILE}!")

if __name__ == "__main__":
    main()
