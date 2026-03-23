import csv
import json
from qdrant_exp import HistoryQdrantManager

def evaluate_rag(csv_file_path):
    db = HistoryQdrantManager("history_russia_clean")
    
    results = []
    metrics = {
        'total_questions': 0,
        'exact_match': 0,
        'document_match': 0,
        'retrieved_any': 0,
        'retrieved_none': 0,
        'mrr_score': 0.0,
        'hit_rate_1': 0,
        'hit_rate_3': 0,
        'hit_rate_5': 0,
        'avg_precision': 0.0
    }
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            metrics['total_questions'] += 1
            
            query_id = row['query_id']
            query = row['query']
            expected_doc_id = int(row['gold_document_id'])
            expected_chunk_id = int(row['gold_chunk_id'])
            
            search_result = db.query_to_db(question=query, limit=5)
            
            if search_result == "НИЧЕГО НЕ НАШЕЛ НЕ НАДО ТЫКАТЬ ПО API ЛИШНИЙ РАЗ НЕЙРОНКУ!!!":
                metrics['retrieved_none'] += 1
                results.append({
                    'query_id': query_id,
                    'query': query,
                    'expected_doc': expected_doc_id,
                    'expected_chunk': expected_chunk_id,
                    'found_docs': [],
                    'exact_match': False,
                    'document_match': False
                })
                continue

            found_items = []
            exact_found = False
            doc_found = False
            
            for text, metadata in search_result:
                doc_id = metadata.get('document_id')
                chunk_id = metadata.get('chunk_id')
                
                found_items.append({
                    'document_id': doc_id,
                    'chunk_id': chunk_id
                })
                
                if doc_id == expected_doc_id and chunk_id == expected_chunk_id:
                    exact_found = True
                
                if doc_id == expected_doc_id:
                    doc_found = True

            
            # MRR
            rank = None
            for i, item in enumerate(found_items):
                if item['document_id'] == expected_doc_id and item['chunk_id'] == expected_chunk_id:
                    rank = i + 1
                    break
            if rank:
                metrics['mrr_score'] += 1.0 / rank

            # Hit@K
            found_in_top_1 = len(found_items) > 0 and (
                found_items[0]['document_id'] == expected_doc_id and 
                found_items[0]['chunk_id'] == expected_chunk_id
            )
            if found_in_top_1:
                metrics['hit_rate_1'] += 1

            found_in_top_3 = any(
                item['document_id'] == expected_doc_id and item['chunk_id'] == expected_chunk_id
                for item in found_items[:3]
            )
            if found_in_top_3:
                metrics['hit_rate_3'] += 1

            found_in_top_5 = any(
                item['document_id'] == expected_doc_id and item['chunk_id'] == expected_chunk_id
                for item in found_items[:5]
            )
            if found_in_top_5:
                metrics['hit_rate_5'] += 1

            # Precision@5
            relevant_in_top_5 = sum(
                1 for item in found_items[:5]
                if item['document_id'] == expected_doc_id and item['chunk_id'] == expected_chunk_id
            )
            precision_at_5 = relevant_in_top_5 / min(5, len(found_items)) if found_items else 0
            metrics['avg_precision'] += precision_at_5

            if exact_found:
                metrics['exact_match'] += 1
            if doc_found:
                metrics['document_match'] += 1
            if found_items:
                metrics['retrieved_any'] += 1
            
            results.append({
                'query_id': query_id,
                'query': query,
                'expected_doc': expected_doc_id,
                'expected_chunk': expected_chunk_id,
                'found_docs': found_items,
                'exact_match': exact_found,
                'document_match': doc_found
            })
    
    if metrics['total_questions'] > 0:
        metrics['exact_match_rate'] = metrics['exact_match'] / metrics['total_questions']
        metrics['document_match_rate'] = metrics['document_match'] / metrics['total_questions']
        metrics['recall_any'] = metrics['retrieved_any'] / metrics['total_questions']
        metrics['mrr_score'] /= metrics['total_questions']
        metrics['hit_rate_1'] /= metrics['total_questions']
        metrics['hit_rate_3'] /= metrics['total_questions']
        metrics['hit_rate_5'] /= metrics['total_questions']
        metrics['avg_precision'] /= metrics['total_questions']
    
    return results, metrics

if __name__ == "__main__":
    results, metrics = evaluate_rag("full_bench.csv")

    print("\n" + "="*50)
    print("МЕТРИКИ КАЧЕСТВА RAG")
    print("="*50)
    print(f"Всего вопросов: {metrics['total_questions']}")
    print(f"Точных совпадений (doc+chunk): {metrics['exact_match']} ({metrics.get('exact_match_rate', 0):.2%})")
    print(f"Совпадений по документу: {metrics['document_match']} ({metrics.get('document_match_rate', 0):.2%})")
    print(f"Нашёл что-то: {metrics['retrieved_any']} ({metrics.get('recall_any', 0):.2%})")
    print(f"Ничего не нашёл: {metrics['retrieved_none']}")
    print(f"MRR@5: {metrics['mrr_score']:.3f}")
    print(f"Hit@1: {metrics['hit_rate_1']:.2%}")
    print(f"Hit@3: {metrics['hit_rate_3']:.2%}")
    print(f"Hit@5: {metrics['hit_rate_5']:.2%}")
    print(f"Average Precision@5: {metrics['avg_precision']:.3f}")
    
    with open("rag_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
