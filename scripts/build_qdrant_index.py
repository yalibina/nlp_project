import json
import sys
from pathlib import Path

# импорт фэковый потому что почти наверное я это просто запущу у себя локально
from app.core.qdrant_manager import HistoryQdrantManager

def main():
    # заглушка
    json_path = ""
    """
    вот такой джсон я ожидаю по типу
    data = [
        {"text": "В 1812 году Наполеон вторгся в Россию, началась Отечественная война.", "years": [1812], "document_id" : 0, "chunk_id": 0},
        {"text": "В 1961 году Юрий Гагарин совершил первый полет человека в космос.", "years": [1961], "document_id" : 1, "chunk_id": 0},
        {"text": "В 1861 году Александр II подписал манифест об отмене крепостного права.", "years": [1861], "document_id" : 2, "chunk_id": 0},
        {"text": "Однако после этого еще много лет крестьяне не имели полной свободы","document_id" : 2, "chunk_id": 1},
        {"text": "В 1941 году началась Великая Отечественная Война, она закончилась в 1945, унеся жизни миллионов людей", "years": [1941, 1945], "document_id" : 3, "chunk_id": 0}
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        # list из dict'ов с полями (text, years, document_id, chunk_id)
        facts_data = json.load(f)

    db = HistoryQdrantManager(collection_name="history_russia")

    db.insert_facts(facts=facts_data, batch_size=128)
    
    print("БД полностью готова")

if __name__ == "__main__":
    main()
