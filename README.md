# 🗂️ Ingestion Script

> **Умная индексация кодовой базы в векторный поиск Qdrant.**
> Превратите ваш код в знания для RAG-агентов за считанные секунды. Поддержка `.gitignore`, инкрементальное обновление и совместимость с OpenAI API (Local LLM/Ollama).

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

# ✨ Возможности (Features)

- 🧠 **Инкрементальная индексация:** Вычисляет MD5-хеши файлов и пропускает неизмененные. Экономит время и токены.
- 🛡️ **Умная фильтрация:** Автоматически учитывает правила `.gitignore`.
- 🔌 **LLM Agnostic:** Работает с любым провайдером, совместимым с OpenAI API (OpenAI, LM Studio, Ollama, vLLM).
- 🧩 **Мультиязычность:** Поддержка разбиения (chunking) для 30+ языков программирования (Python, JS/TS, Go, Rust, C++, и др.).
- 📊 **Визуализация:** Наглядный UX: прогресс-бары и спиннеры (Halo, Tqdm).

# 📒 Поддерживаемые языки

<details>
<summary>📦 Список поддерживаемых расширений</summary>

| Расширение                  | Язык (LangChain) |
| --------------------------- | ---------------- |
| `.py`                       | Python           |
| `.js`, `.jsx`               | JavaScript       |
| `.ts`, `.tsx`               | TypeScript       |
| `.c`, `.cpp`, `.hpp`, `.cc` | C / C++          |
| `.cs`                       | C#               |
| `.go`                       | Go               |
| `.java`                     | Java             |
| `.kt`                       | Kotlin           |
| `.scala`                    | Scala            |
| `.rb`                       | Ruby             |
| `.rs`                       | Rust             |
| `.swift`                    | Swift            |
| `.ex`, `.exs`               | Elixir           |
| `.php`                      | PHP              |
| `.lua`                      | Lua              |
| `.pl`                       | Perl             |
| `.ps1`                      | PowerShell       |
| `.r`                        | R                |
| `.hs`                       | Haskell          |
| `.cbl`                      | Cobol            |
| `.vb`                       | Visual Basic 6   |
| `.html`                     | HTML             |
| `.md`                       | Markdown         |
| `.rst`                      | RST              |
| `.tex`                      | LaTeX            |
| `.sol`                      | Solidity         |
| `.proto`                    | Protocol Buffers |

</details>

---

# 🛠 Требования (Prerequisites)

- **Docker** v29.1.2+
- **Python** 3.14+
- **Менеджер пакетов:** `uv` (рекомендуется) или `pip`

# ⚙️ Установка и Настройка

## 1. Запуск инфраструктуры

Запустите Qdrant одной командой используя Docker:

```bash
# Запуск векторной базы данных Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

## 2. Клонирование репозитория

```bash
git clone https://github.com/your-username/ingestion-script.git
cd ingestion-script
```

## 3. Настройка переменных окружения

Скопируйте файл `.env.template` и переименуйте его в `.env`:

```bash
# Windows (PowerShell):
copy-item .env.template .env
# Windows (CMD):
copy .env.template .env
# macOS/Linux:
cp .env.template .env
```

## 4. Установите зависимости

### Вариант А: Через `uv` (Рекомендуемый)

```bash
# 1. Создание venv и установка зависимостей (одной командой)
uv venv && uv sync

# 2. Запуск
uv run main.py '/path/to/your/project'
```

### Вариант Б: Через стандартный `pip`

Классический способ.

```bash
# 1. Создание venv
python -m venv .venv

# 2. Активация
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Установка
pip install -r requirements.txt

# 4. Запуск
python main.py '/path/to/your/project'
```

**и отредактируйте при необходимости постоянно использовать конкретный провайдер и модель**

# ⏭️ Примеры запуска с конфигурацией провайдеров

Вы можете задать эти параметры в файле `.env` или передать их через аргументы командной строки при запуске скрипта.

## 1. Использование OpenAI (оригинал)

В `.env` файле:

```env
BASE_LLM_PROVIDER_URL=https://api.openai.com/v1
LLM_PROVIDER_API_KEY=sk-proj-XXXXXX...
EMBD_MODEL=text-embedding-3-small
EMBD_VECTOR_SIZE=1536
```

Или через CLI:

```bash
uv run main.py '/path/to/your/project' --base_llm_provider_url "https://api.openai.com/v1" --llm_provider_api_key "sk-proj-XXXXXX..." --embd_model "text-embedding-3-small" --embd_vector_size 1536
```

## 2. Использование локального сервера (LM Studio или Ollama)

В `.env` файле:

```env
BASE_LLM_PROVIDER_URL=http://localhost:1234/v1
LLM_PROVIDER_API_KEY=not-needed
EMBD_MODEL=nomic-ai/nomic-embed-text-v1.5-GGUF
EMBD_VECTOR_SIZE=768
```

Или через CLI:

```bash
uv run main.py '/path/to/your/project' --base_llm_provider_url "http://localhost:1234/v1" --llm_provider_api_key "not-needed" --embd_model "nomic-ai/nomic-embed-text-v1.5-GGUF" --embd_vector_size 768
```

## 3. Использование облачных провайдеров (Together AI)

В `.env` файле:

```env
BASE_LLM_PROVIDER_URL=https://api.together.xyz/v1
LLM_PROVIDER_API_KEY=your_together_api_key
EMBD_MODEL=togethercomputer/m2-bert-80M-8k-retrieval
EMBD_VECTOR_SIZE=768
```

Или через CLI:

```bash
uv run main.py '/path/to/your/project' --base_llm_provider_url "https://api.together.xyz/v1" --llm_provider_api_key "your_together_api_key" --embd_model "togethercomputer/m2-bert-80M-8k-retrieval" --embd_vector_size 768
```

# 🏗️ Технические детали и Архитектура

## Схема процесса (Workflow)

```mermaid
graph TD
    Start((Начало)) --> Scan[Сканирование файлов (с учетом .gitignore)]
    Scan --> Loop{Для каждого файла}

    Loop --> Hash[Вычисление MD5 хеша]
    Hash --> Check{Хеш есть в Qdrant?}

    Check -- Да --> Skip[Пропустить файл]
    Check -- Нет --> Split[Разбиение на чанки (RecursiveCharacterTextSplitter)]

    Split --> Embed[Генерация эмбеддингов (OpenAI API / Local LLM)]
    Embed --> Upsert[Сохранение в Qdrant (Vector + Metadata)]

    Upsert --> Next[Следующий файл]
    Skip --> Next
    Next --> Loop
    Loop -- Все файлы обработаны --> End((Конец))

    style Check fill:#f9f,stroke:#333,stroke-width:2px
    style Skip fill:#fff4dd,stroke:#d4a017
    style Upsert fill:#d4edda,stroke:#28a745
```

## Структура данных в Qdrant:

- **path**: Полный путь к файлу (используется для фильтрации).
- **content**: Текст чанка с заголовком в виде имени файла.
- **language**: Язык программирования для фильтрации запросов.
- **hash**: MD5-хеш версии файла на момент индексации.

# 🪲 Решение проблем (Troubleshooting)

## Типичные проблемы:

- **Connection Error**: Убедитесь, что Qdrant доступен по адресу http://localhost:6333. Если вы запускаете скрипт внутри другого контейнера, используйте имя сервиса.
- **Dimension Mismatch**: Если вы сменили модель эмбеддингов, нужно либо создать новую коллекцию, либо удалить старую, так как размер вектора (EMBD_VECTOR_SIZE) фиксируется при создании.

# 📄 Лицензия и Вклад

Этот проект распространяется под лицензией **MIT**.
