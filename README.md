# RAG-MISIS CHARIOT

# Описание проекта

В этом проекте реализованы два подхода для построения RAG (Retrieval-Augmented Generation) решения с использованием  **Severless VLLM** и **YandexGPT**. Оба подхода представлены в виде отдельных модулей и используют уникальные возможности каждой модели для обработки запросов.

## Структура файлов

- **main_vllm.py** и **vectorestore_vllm.py** — реализация с использованием Severless VLLM на базе модели **Meta-Llama-3.1-70B-Instruct-AWQ-INT4**.
  - Эти файлы содержат код для интеграции с Severless VLLM API, что позволяет эффективно использовать мощные языковые модели в облаке.

- **main_yandex.py** и **vectorestore_yandex.py** — реализация с использованием **API YandexGPT**.
  - Эти файлы используют API YandexGPT для создания эмбеддингов и генерации ответов, что особенно полезно для задач, требующих высокой релевантности ответов и поддержки русского языка.

## Описание каждой реализации

### Severless VLLM с Meta-Llama-3.1-70B-Instruct-AWQ-INT4

- **Файлы**: `main_vllm.py`, `vectorestore_vllm.py`
- **Модель**: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- **Использование**: Severless VLLM API позволяет выполнять генерацию текста на базе модели Meta-Llama, которая ориентирована на ответы с высокой степенью точности. Это решение оптимально для задач, где требуется детальная обработка текста, мощные вычислительные ресурсы и масштабируемость в облаке.
- **Особенности**: поддержка крупной модели, высокая точность ответов и гибкость использования облачных ресурсов.

### YandexGPT

- **Файлы**: `main_yandex.py`, `vectorestore_yandex.py`
- **Модель**: YandexGPT
- **Использование**: YandexGPT API предоставляет доступ к модели, адаптированной для русского языка, что делает её особенно полезной для задач, требующих работы с русскоязычными текстами и обработки запросов на русском языке.
- **Особенности**: встроенные эмбеддинги и генерация на русском языке, что обеспечивает высокую релевантность для специфических бизнес-задач и взаимодействия с локальными данными.

## Как запустить проект

1. **Установка зависимостей**: Убедитесь, что Python 3.8+ установлен, и установите зависимости с помощью команды:
   ```bash
   pip install -r requirements.txt



