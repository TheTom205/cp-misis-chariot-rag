import json
import clickhouse_connect
import networkx as nx
from docx import Document as DocxDocument
import openai
from langchain_community.embeddings.yandex import YandexGPTEmbeddings

from langchain_community.llms import YandexGPT

import re
import requests
class Vectorestore:
    def __init__(self, docx_path: str, ch_size: int = 1024):
        self.ch_size = ch_size
        # openai_api_key =  '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'
        
        # Initialize Chat model for answering and Embeddings model for embedding generation
        # self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)
        # self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-large')
        # self.llm = YandexGPT(
        #     api_key='AQVNwwO6r_Ko914j7zAW3H5lPga-sVKoXCipbr8_',
        #     folder_id='b1gp8gibtjpdhh1rk46d',
        #     model='pro-latest',
        #     temperature=0.1,
        #     top_p=0.85,
        #     max_tokens=250,
        #     frequency_penalty=0.2,
        #     presence_penalty=0.3
        # )
        self.embeddings = YandexGPTEmbeddings(api_key='AQVNwwO6r_Ko914j7zAW3H5lPga-sVKoXCipbr8_', folder_id='b1gp8gibtjpdhh1rk46d')

        # Подключение к ClickHouse
        self.client = clickhouse_connect.get_client(
            host='0.0.0.0',
            port=8123,
            username='default',
            password=''
        )

        # Создаём таблицу user_memory в базе rag_system, если её ещё нет
        self.client.command('''
            CREATE TABLE IF NOT EXISTS rag_system.user_memory (
                user_id String,
                chat_id String,
                message String,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (user_id, chat_id, timestamp)
        ''')

    def save_to_memory(self, user_id: str, chat_id: str, message: str):
        """Сохраняет сообщение в память ClickHouse в таблице rag_system.user_memory."""
        query = '''
            INSERT INTO rag_system.user_memory (user_id, chat_id, message) VALUES (%s, %s, %s)
        '''
        self.client.command(query, (user_id, chat_id, message))

    def get_relevant_memory(self, query_embedding, user_id: str, chat_id: str, limit: int = 10, relevance_threshold: float = 0.55):
        """Извлекает только релевантные сообщения пользователя из таблицы rag_system.user_memory на основе порога релевантности."""
        
        # Извлекаем последние сообщения пользователя
        query = f'''
            SELECT message FROM rag_system.user_memory
            WHERE user_id = '{user_id}' AND chat_id = '{chat_id}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        results = self.client.query(query).result_rows
        
        # Фильтруем сообщения на основе порога схожести
        relevant_messages = []
        for row in results:
            message = row[0]
            message_embedding = self.embeddings.embed_query(message)  # Получаем embedding для каждого сообщения
            similarity = sum([a * b for a, b in zip(query_embedding, message_embedding)])  # Считаем dot product для определения релевантности

            if similarity >= relevance_threshold:
                relevant_messages.append(message)
                
        return "\n".join(relevant_messages)

    def read_docx(self, file_path):
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text

    def extract_structure(self, text):
        pattern = re.compile(r'(\d+(\.\d+)+)\s+(.*)')
        structure = []
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                number = match.group(1)
                title = match.group(3)
                structure.append((number, title))
        return structure


    def similarity_search(self, query_embedding, k=2, metric="dotProduct", threshold=0.35):
        embedding_str = ', '.join(map(str, query_embedding))
        
        similarity_function = "dotProduct" if metric == "dotProduct" else "euclideanDistance"

        if threshold is not None:
            if metric == "dotProduct":
                threshold_condition = f"{similarity_function}(embedding, [{embedding_str}]) >= {threshold}"
            else:
                threshold_condition = f"{similarity_function}(embedding, [{embedding_str}]) <= {threshold}"
        else:
            threshold_condition = "1"

        query = f'''
            SELECT metadata, content, embedding,
            {similarity_function}(embedding, [{embedding_str}]) AS similarity
            FROM rag_system.docs_ya
            WHERE {threshold_condition}
            ORDER BY similarity {'DESC' if metric == "dotProduct" else 'ASC'}
            LIMIT {k}
        '''

        result = self.client.query(query).result_rows
        return result
    
    def get_last_messages_by_chat(self):
        """Получает все уникальные chat_id и последнее сообщение в каждом чате, отсортированные по времени последнего сообщения и chat_id."""
        query = '''
            SELECT chat_id, user_id, argMax(message, timestamp) AS last_message, max(timestamp) AS last_timestamp
            FROM rag_system.user_memory
            GROUP BY chat_id, user_id
            ORDER BY last_timestamp DESC, chat_id DESC
        '''
        results = self.client.query(query).result_rows
        return [
            {
                "chat_id": row[0],
                "user_id": row[1],
                "last_message": row[2],
                "last_timestamp": row[3].isoformat()  # Преобразуем timestamp в ISO формат
            } for row in results
        ]    
    def create_chat(self, user_id: str, document_id: str):
        """Создает новый чат с указанными user_id и document_id, генерируя уникальный chat_id."""
        # Генерация уникального chat_id (можно использовать UUID или timestamp)
        import uuid
        chat_id = str(uuid.uuid4())  # Генерируем уникальный chat_id
        
        # Вставка данных в ClickHouse
        query = '''
            INSERT INTO rag_system.user_memory (user_id, chat_id, message) VALUES (%s, %s, %s)
        '''
        initial_message = f"Чат создан для документа {document_id}"
        self.client.command(query, (user_id, chat_id, initial_message))
        
        return {"chat_id": chat_id, "user_id": user_id, "document_id": document_id}


    
    def async_get_answer(self, query: str = None, user_id: str = None, chat_id: str = None):
        def get_model_answer(messages):
            VLLM_ENDPOINT = "https://api.runpod.ai/v2/vllm-02fton87gguw5v/openai/v1"
            RUNPOD_TOKEN = "ZK57MDEASEH3KC8WJDX51144FXJ8TWY37X8YQICP"
            openai.api_base = VLLM_ENDPOINT
            openai.api_key = RUNPOD_TOKEN
            completion = openai.ChatCompletion.create(
                model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                messages=messages,
                temperature=0.3,
                max_tokens=3000,
                n=1
            )
            return completion['choices'][0]['message']['content']
        def preprocess_query(query):
            synonyms = {"КК": "система"}
            for abbr, full in synonyms.items():
                query = query.replace(abbr, full)
            return query
        def send_telegram_message(text: str):
            bot_token = "7659961567:AAEM5ZMDYb0Ia7UDuf1SvjPEenWBu0AZo1Y"
            chat_id = "-1002388226666"

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': text
            }
            response = requests.post(url, data=payload)
            
            if response.status_code == 200:
                print("Сообщение успешно отправлено.")
            else:
                print(f"Ошибка при отправке сообщения: {response.status_code} - {response.text}")

        query_embedding = self.embeddings.embed_query(preprocess_query(query))
        docs = self.similarity_search(query_embedding)

        enriched_docs = [metadata + content for metadata, content, _, _ in docs] if docs else []
        # if enriched_docs:
        message_content = "\n".join(enriched_docs)
        # else:
        #     message_content = 'Нет релевантной информации'
        #     send_telegram_message(f'chat_id: {chat_id}\nmessage: "{query}"')


        # Получаем релевантные сообщения из истории на основе схожести
        relevant_history = self.get_relevant_memory(query_embedding, user_id, chat_id)

        # Создаём input для модели с добавлением только релевантной истории

        system_prompt = """
            Ваша задача — предоставить поддержку по вопросам работы с приложением, используя исключительно информацию из контекста. Отвечайте кратко и по существу, разделяя шаги и инструкции на отдельные строки для удобства чтения.

            Начинайте ответ сразу с сути, избегая вводных слов, таких как "Ответ". Если ваш ответ включает несколько шагов или действий, перечисляйте их построчно с переносом строки. В конце каждого ответа указывайте все соответствующие пункты и ссылку на документ в формате: «(см. Документ 1, пункт 2.8.8)».

            Используйте только те пункты, которые явно указаны в контексте, и не добавляйте несуществующие пункты. Если информация не найдена в контексте, вежливо сообщите: «К сожалению, я не могу помочь с этим вопросом».

            Ваш приоритет — давать точный, чёткий ответ, разбивая шаги на отдельные строки и всегда указывая ссылку на документ и номер пункта в конце сообщения, если информация найдена в контексте.
        """



        combined_input = f"{message_content}{relevant_history}"
        print(combined_input)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": combined_input},
            {"role": "user", "content": query}
            ]

        # Get the response
        response = get_model_answer(messages)
        if response=='К сожалению, я не могу помочь с этим вопросом.':
            send_telegram_message(f'chat_id: {chat_id}\nmessage: "{query}"')

        # Сохранение текущего запроса и ответа в ClickHouse
        self.save_to_memory(user_id, chat_id, f"Вопрос: {query}\nОтвет: {response}")
        return response
