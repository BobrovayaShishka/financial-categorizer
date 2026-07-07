# Bank Service — категоризация расходов по выписке

Сервис автоматической категоризации банковских операций для **счёта физического лица**.  
Часть транзакций приходит без категории — сервис распределяет их по расходным категориям верхнего уровня, минимизируя попадание в «Прочее».

## Возможности

- **Трёхуровневый пайплайн** с экономией токенов:
  1. База мерчантов (`config/merchants.yaml`) — 0 токенов
  2. Правила по ключевым словам (`config/categories.yaml`) — 0 токенов
  3. LLM через Ollama — только для неопознанных операций, батчами
- Понимание бизнес-расходов: Tilda, Figma, хостинг, реклама и т.д.
- Генератор **синтетических выписок** для тестирования без реального датасета
- Оценка точности: accuracy, F1 по категориям, доля «Прочее», учёт токенов
- FastAPI + веб-демо
- Docker Compose: Ollama + API
- Заготовка экспорта данных для **fine-tune**, если accuracy < 95%

## Быстрый старт (Docker)

```bash
cp .env.example .env
docker compose up --build
```

После запуска:
- **Демо UI:** http://localhost:8000
- **Ollama:** http://localhost:11434

При первом запуске контейнер `ollama-init` автоматически скачает модель `qwen2.5:3b`.

## Категории расходов

| ID | Категория |
|----|-----------|
| `groceries` | Продукты и быт |
| `dining` | Рестораны и кафе |
| `transport` | Транспорт |
| `subscriptions` | Подписки и связь |
| `health` | Здоровье и красота |
| `shopping` | Покупки и одежда |
| `entertainment` | Развлечения |
| `housing` | Жильё и коммунальные |
| `education` | Образование |
| `finance` | Финансы и переводы |
| `business` | Бизнес и самозанятость |
| `other` | Прочее |

Категории и мерчанты настраиваются в YAML без изменения кода.

## Структура проекта

```
bank-service/
├── docker-compose.yml      # Ollama + API
├── Dockerfile
├── requirements.txt
├── .env.example
├── config/
│   ├── categories.yaml     # категории и ключевые слова
│   └── merchants.yaml      # мерчанты → категория (Tilda → business)
├── data/
│   ├── raw/                # выписка без разметки
│   ├── labeled/            # выписка + ground_truth
│   └── reports/            # JSON-отчёты оценки
├── src/
│   ├── domain/             # модели, таксономия
│   ├── engines/            # merchant KB, rules, LLM
│   ├── pipeline/           # оркестратор категоризации
│   ├── synth/              # генератор синтетики
│   ├── metrics/            # метрики качества
│   ├── training/           # экспорт для fine-tune
│   └── api/                # FastAPI + демо UI
├── scripts/
│   ├── generate_data.py    # генерация синтетической выписки
│   └── run_eval.py         # оценка accuracy
└── tests/
```

## Скрипты

### Генерация синтетических данных

```bash
python scripts/generate_data.py
# или в Docker:
docker compose exec api python scripts/generate_data.py
```

Создаёт 250 транзакций:
- `data/raw/statement.csv` — как из банка
- `data/labeled/statement_labeled.csv` — с эталонными категориями

### Оценка точности

```bash
# только правила + база мерчантов (без Ollama)
python scripts/run_eval.py --no-llm

# с LLM
python scripts/run_eval.py
```

Отчёт сохраняется в `data/reports/latest.json`.

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Демо-интерфейс |
| `GET` | `/health` | Статус сервиса |
| `GET` | `/categories` | Список категорий |
| `POST` | `/categorize` | Категоризация JSON |
| `POST` | `/categorize/upload` | Загрузка CSV |
| `POST` | `/demo/evaluate` | Оценка на синтетике |

### Формат CSV

```csv
id,date,amount,description,mcc,bank_category
tx_0001,2026-05-09,-1500.00,PYATEROCHKA 1234,5411,
```

- `amount < 0` — расход (категоризируется)
- `amount >= 0` — доход (пропускается)
- `bank_category` — необязательная категория от банка

## Конфигурация

Скопируйте `.env.example` → `.env`:

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OLLAMA_HOST` | `http://ollama:11434` | Адрес Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Модель |
| `LLM_BATCH_SIZE` | `15` | Размер батча для LLM |
| `ENABLE_LLM_CACHE` | `true` | Кэш ответов LLM |

## Пайплайн

```
Выписка CSV
    │
    ▼
amount < 0? ──нет──► Доход (skip)
    │
   да
    ▼
Есть bank_category? ──да──► Маппинг
    │
   нет
    ▼
Merchant KB ──match──► Категория (0 токенов)
    │
   нет
    ▼
Keyword rules ──match──► Категория (0 токенов)
    │
   нет
    ▼
LLM (Ollama, batch) ──► Категория + учёт токенов
```

## Fine-tune (если accuracy < 95%)

```python
from pathlib import Path
from src.training.export_dataset import export_training_pairs

export_training_pairs(
    Path("data/labeled/statement_labeled.csv"),
    Path("data/training/dataset.jsonl"),
)
```

Экспортированный JSONL можно использовать для LoRA fine-tune (Unsloth, Axolotl и т.д.), затем заменить `OLLAMA_MODEL` в `.env`.

## Тесты

```bash
docker compose exec api python -m pytest tests/ -q
```

## Локальная разработка

> Рекомендуется Python 3.11–3.12. На Python 3.14 pydantic может не собраться — используйте Docker.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

## Лицензия

MIT (или укажите свою)
