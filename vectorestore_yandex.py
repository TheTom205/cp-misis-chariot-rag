import json
import clickhouse_connect
import networkx as nx
from docx import Document as DocxDocument
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
import re
import requests
class Vectorestore:
    def __init__(self, docx_path: str, ch_size: int = 1024):
        self.ch_size = ch_size
        openai_api_key =  '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'
        
        # Initialize Chat model for answering and Embeddings model for embedding generation
        # self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)
        # self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-large')
        self.llm = YandexGPT(
            api_key='AQVNwwO6r_Ko914j7zAW3H5lPga-sVKoXCipbr8_',
            folder_id='b1gp8gibtjpdhh1rk46d',
            model='pro-latest',
            temperature=0.1,
            top_p=0.85,
            max_tokens=250,
            frequency_penalty=0.2,
            presence_penalty=0.3
        )
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

    def get_relevant_memory(self, query_embedding, user_id: str, chat_id: str, limit: int = 10, relevance_threshold: float = 0.7):
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


    def similarity_search(self, query_embedding, k=2, metric="dotProduct", threshold=0.6):
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
        """Получает все уникальные chat_id и последнее сообщение в каждом чате."""
        query = '''
            SELECT chat_id, message 
            FROM rag_system.user_memory AS t1
            JOIN (
                SELECT chat_id, max(timestamp) AS last_timestamp
                FROM rag_system.user_memory
                GROUP BY chat_id
            ) AS t2 ON t1.chat_id = t2.chat_id AND t1.timestamp = t2.last_timestamp
            ORDER BY chat_id
        '''
        results = self.client.query(query).result_rows
        return [{"chat_id": row[0], "last_message": row[1]} for row in results]
    
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
        if enriched_docs:
            message_content = "\n".join(enriched_docs)
        else:
            message_content = 'Нет релевантной информации'
            send_telegram_message(f'chat_id: {chat_id}\nmessage: "{query}"')


        # Получаем релевантные сообщения из истории на основе схожести
        relevant_history = self.get_relevant_memory(query_embedding, user_id, chat_id)

        # Создаём input для модели с добавлением только релевантной истории
        system_prompt = """
                        Ваша задача — техническая поддержка: помогите пользователю понять, как работает приложение и предложите конкретные шаги для решения его вопроса.
                        
                        Начинайте ответ сразу с сути, без вводных слов, таких как "Ответ". Давайте чёткие и краткие ответы, избегая лишних деталей. Если информация найдена в контексте, укажите все пункты, в которых содержится информация, в конце ответа: «(см. пункты 2.3, 2.8.8)».
                        
                        Если проблема требует действий, предложите конкретные шаги для её решения в формате списка. Например:
                        
                        1. Перейдите в раздел «Профили».
                        2. Выберите профиль, который хотите активировать.
                        3. Измените статус профиля на «Активный».
                        
                        Используйте только те пункты, которые явно указаны в контексте. Не добавляйте несуществующие пункты и не делайте предположений о наличии пунктов.

                        Если в контексте нет полезных данных, вежливо сообщите: «К сожалению, я не могу помочь с этим вопросом».
                        
                        Ваш приоритет — дать краткий, точный ответ, избегая предположений и добавляя пункты только при их наличии в контексте.
                        """

        combined_input = f"Текущий вопрос:\n{query}\n\nКонтекст:\n{message_content}\n\nРелевантная история:\n{relevant_history}"
        print(combined_input)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        output_parser = StrOutputParser()
        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=output_parser)

        # Получаем ответ от модели
        answer = chain.invoke({"input": combined_input})
        answer_text = answer['text'] if isinstance(answer, dict) and 'text' in answer else str(answer)
        trimmed_answer_text = answer_text

        # Сохранение текущего запроса и ответа в ClickHouse
        self.save_to_memory(user_id, chat_id, f"Вопрос: {query}\nОтвет: {trimmed_answer_text}")
        return trimmed_answer_text
