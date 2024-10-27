# RAG-MISIS CHARIOT

Проект реализует классический RAG (Retrieval-Augmented Generation) пайплайн с ключевыми модификациями в предобработке текста и его разбиении на смысловые блоки. В отличие от традиционного подхода, где текст делится на части ограниченного размера, в данном решении текст делится на смысловые блоки, связанные с конкретными пунктами документа. Этот подход улучшает точность цитирования исходного текста и повышает качество ответов.

## Ключевые особенности

1. **Создание смысловых блоков**:
   - Вместо стандартного разбиения по количеству символов или токенов текст делится на смысловые блоки, основываясь на контексте.
   - Каждый блок привязан к конкретному пункту документа, что позволяет модели давать более точные и релевантные цитаты, повышая надежность и прозрачность ответов.

2. **Улучшенная точность цитирования**:
   - Привязка каждого блока к конкретному разделу документа позволяет точно указывать на источник. Это делает ответы более понятными и повышает их достоверность.

3. **Оптимальный выбор языковой модели**:
   - Проведено мини-исследование для выбора оптимальной модели, соответствующей задачам бизнеса, что обеспечивает баланс между производительностью, точностью и эффективностью.

## Обзор пайплайна

Пайплайн включает следующие этапы:

1. **Предобработка текста**:
   - Текст документа проходит предварительную обработку, ориентированную на смысловое разбиение, а не на ограниченное по размеру.

2. **Смысловое разбиение на блоки**:
   - Текст делится на блоки, исходя из его контекста и содержания, причем каждый блок привязывается к соответствующему пункту документа.

3. **Генерация с поддержкой извлечения (RAG)**:
   - При генерации ответов наиболее релевантные блоки извлекаются и передаются в модель для создания осмысленных и основанных на контексте ответов с точными ссылками.

## Преимущества подхода

- **Повышенное качество ответов**: Смысловое разбиение позволяет модели давать точные и релевантные ответы.
- **Надежное цитирование источников**: Привязка блоков к пунктам документа позволяет точно указывать источники, что повышает доверие и удобство.
- **Эффективность модели**: Выбранная модель оптимизирована под нужды бизнеса, обеспечивая баланс точности и эффективности.

main_vllm.py, vectorestore_vllm.py - код с использованием Severless VLLM c Meta-Llama-3.1-70B-Instruct-AWQ-INT4
main_yandex.py, vectorestore_yandex.py - код с использованием API YandexGPT

  Установите зависимости командой:
  ```bash
  pip install -r requirements.txt


