from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, SparseVectorParams, Modifier, Prefetch, SparseVector, FusionQuery, Fusion, Range, MatchAny
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import torch
import re
import uuid
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from natasha.extractors import DatesExtractor

class HistoryQdrantManager:
    def __init__(self, collection_name="history_russia"):
        if torch.backends.mps.is_available():
            device = "mps"  # для моего мака
            print("Выбран Apple Silicon(MPS)!")
        elif torch.cuda.is_available():
            device = 'cuda' # для не маков с ГПУ
            print("Выбрана CUDA")
        else:
            device = "cpu"
            print("Выбрано ЦПУ")
        
        self.device = device
        self.collection_name = collection_name
        
        self.client = QdrantClient(url="http://localhost:6333")

        print("Загрузка NLP моделей Natasha...")
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
        
        print("Загрузка модели BERTA")
        self.dense_model = SentenceTransformer("sergeyzh/BERTA", device=self.device)
        self.vector_size = self.dense_model.get_sentence_embedding_dimension() # Будет 768
        
        print("Загрузка модели BM25...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        #print("768? ->", self.vector_size)
        self.setup_collection()

    def lemmatize_text(self, text: str) -> str:
        clean_text = re.sub(r'[^\w\s]', ' ', text)

        doc = Doc(clean_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        lemmas = [token.lemma for token in doc.tokens if token.lemma]
        return " ".join(lemmas)

    def setup_collection(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Создаем коллекцию '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                sparse_vectors_config={  # для BM-25
                    "texts_BM25": SparseVectorParams(modifier=Modifier.IDF)
                },
                vectors_config={  # для векторного поиска
                    "texts_transformer": VectorParams(size=self.vector_size, distance=Distance.COSINE)
                }
            )
            print("Создаем индекс для быстрого поиска по годам...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="years",
                field_schema=PayloadSchemaType.INTEGER
            )
            print("Создаем индекс для быстрого поиска по айди документов...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.INTEGER
            )
            print("Создаем индекс для быстрого поиска по айди чанков...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema=PayloadSchemaType.INTEGER
            )
        else:
            print("Коллекция уже существует")


    def insert_facts(self, facts: list, batch_size: int = 32):

        dense_vectors = self.dense_model.encode([fact["text"] for fact in facts], batch_size=batch_size).tolist()

        lemm_text = [self.lemmatize_text(fact["text"]) for fact in facts]

        sparse_vectors = list(self.sparse_model.embed(lemm_text, batch_size=batch_size))

        points = [0 for _ in range(len(facts))]
        for i in range(len(facts)):            
            point = PointStruct(
                id=i, # если хочется рандом то есть uuid64 а если хэш то uuid65 я же рассчитываю что за 1 раз бд сделаем
                vector={
                    "texts_transformer": dense_vectors[i],
                    "texts_BM25": SparseVector(
                        indices=sparse_vectors[i].indices.tolist(), 
                        values=sparse_vectors[i].values.tolist()
                    )
                },
                payload={
                    "text": facts[i]["text"],
                    "lemm_text": lemm_text[i],
                    "years": facts[i].get("years", None),
                    "document_id": facts[i]["document_id"],
                    "chunk_id": facts[i]["chunk_id"]
                }
            )
            points[i] = point
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Успешно загружено {len(facts)} фактов")

    def find_answer(self, question: str, limit: int = 7, batch_size: int = 32, exact_years: list = None):
        # limit должен быть такой чтобы у модельки предсказателя осталось не меньше нескольких тысяч токенов для пресказания ответв
        # для примера у одной гугловской модели 32 768 а у нас контекст эмбедера 512 => примерно такой же размер чанка =>  можно хоть 50, но едва ли больше 10 надо
        RADIUS = 1
        question_dense = self.dense_model.encode(question, batch_size=batch_size).tolist()

        question_sparse_raw = list(self.sparse_model.embed([self.lemmatize_text(question)]))[0]
        
        question_sparse = SparseVector(
            indices=question_sparse_raw.indices.tolist(),
            values=question_sparse_raw.values.tolist()
        )

        if not exact_years:
            print("Год не передан для точного поиска, payload не будет использован")
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=question_dense,
                        using="texts_transformer",
                        limit=limit * 2
                    ),
                    Prefetch(
                        query=question_sparse,
                        using="texts_BM25",
                        limit=limit * 2
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=True  # для дебага поможет
            ).points

        else:
            print(f"Точный поиск по переданным годам: {exact_years}")
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=question_dense,
                        using="texts_transformer",
                        filter=Filter(
                            must=[FieldCondition(key="years", match=MatchAny(any=exact_years))]
                        ),
                        limit=limit
                    ),
                    Prefetch(
                        query=question_sparse,
                        using="texts_BM25",
                        filter=Filter(
                            must=[FieldCondition(key="years", match=MatchAny(any=exact_years))]
                        ),
                        limit=limit
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=True  # для дебага поможет
            ).points

            # когда ищем по датам можем найти кучу чанков с датой но надо поглядеть в соседей для доп инфы если мы разрезали на два чанка сплошной текст-описание 
            if len(search_result) > 0:
                print("Нашлось мало документов с нужной датой, попробуем поискать в соседних чанках доп информацию")
                # окрестность вокруг чанка RADIUS
                # MAX_CHUNKS = 21 <- # 7 * 3

                full_text_map = {}
                lemm_text_map = {}
            
                for hit in search_result:
                    cur_doc_id = hit.payload['document_id']
                    cur_chunk_id = hit.payload['chunk_id']
                    
                    neighbors = self.client.scroll(
                        collection_name=self.collection_name, 
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="document_id", match=MatchValue(value=cur_doc_id)),
                                FieldCondition(key="chunk_id", range=Range(gte=max(cur_chunk_id - RADIUS, 0), lte=cur_chunk_id + RADIUS))
                            ]
                        ), 
                        limit=2*RADIUS+1,
                        with_payload=True
                    )[0]

                    for neighbor in neighbors:
                        key = (neighbor.payload['document_id'], neighbor.payload['chunk_id'])
                        full_text_map[key] = neighbor.payload['text']
                        lemm_text_map[key] = neighbor.payload['lemm_text']

                return [(full_text_map[key], lemm_text_map[key]) for key, _ in sorted(full_text_map.items())]
            
        return [(hit.payload["text"], hit.payload["lemm_text"]) for hit in search_result]

    def query_to_db(self, question: str, limit: int = 10, batch_size: int = 32):
        matches = list(self.dates_extractor(question))
        exact_years = None

        if len(matches) > 0:
            exact_years = [match.fact.year for match in matches]
        
        # подстраховка-эвристика на случай вопроса вида "что произошло в 1942" - наташа не поймет что это год
        # да могут быть случайные совпадения вида 1942 солдата умерло.... Но там или мы найдем что нужно или попросим пользователя переформулировать
        # в переформулировке лучше прямо написать что должны быть акуратные даты или чет такое

        if not exact_years:
            raw_years = re.findall(r'\b[1-9]\d{2,3}\b', question)
            for y_str in raw_years:
                y_int = int(y_str)
                if 100 <= y_int <= 2026:
                    if not exact_years:
                        exact_years = []
                    exact_years.append(y_int)

        search_result = self.find_answer(question, limit, batch_size, exact_years)

        if len(search_result) == 0 or  (exact_years and max(exact_years) >= 2014):
            return "НИЧЕГО НЕ НАШЕЛ НЕ НАДО ТЫКАТЬ ПО API ЛИШНИЙ РАЗ НЕЙРОНКУ!!!"
        else:
            if exact_years:
                lemm_result = ' '.join([lemm_text for _, lemm_text in search_result])
                for match in matches:
                    # проверка если человек спросил конкретную дату а мы нашли год но не про дату - тоже галлюцинация будет
                    if match.fact.day:
                        if str(match.fact.day) not in lemm_result:
                            return "НИЧЕГО НЕ НАШЕЛ НЕ НАДО ТЫКАТЬ ПО API ЛИШНИЙ РАЗ НЕЙРОНКУ!!!"

            # так как ожидается в боте список текстов то не буду склеивать - TO DO - Add metadata
            # return ' '.join([text for text, _ in search_result]) 
            return [text for text, _ in search_result]
