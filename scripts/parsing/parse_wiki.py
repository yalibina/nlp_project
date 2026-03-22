import mwclient
import mwparserfromhell
from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger
)
from natasha.extractors import DatesExtractor
import re
import json
from transformers import AutoTokenizer
from tqdm import tqdm

import warnings

# полностью скрыть все DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''
Extracting metadata
'''

year_pattern = r"\b([1-9]\d{2,3})\b\s*(?:г\.|года|год|гг\.)"
CENTURY_ROMAN = r'(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)'
century_pattern = rf'\b({CENTURY_ROMAN})\s*(?:век(?:а)?|в\.?)'
range_pattern = rf'\b({CENTURY_ROMAN})[–-]({CENTURY_ROMAN})\s*(?:век(?:а)?|вв?\.?)'

def roman_to_int(s):
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    prev = 0
    total = 0
    for ch in reversed(s):
        val = roman[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)
dates_extractor = DatesExtractor(morph_vocab)

def extract_dates(text):
    years = set()
    centuries = set()

    for y in re.findall(year_pattern, text):
        years.add(int(y))
    try:
        dates_natasha = dates_extractor(text)
        for date in dates_natasha:
            if date.fact.year:
                years.add(int(date.fact.year))
    except Exception as e:
        # можно логировать, но не падать
        pass

    for c in re.findall(century_pattern, text):
        centuries.add(roman_to_int(c))

    for c1, c2 in re.findall(range_pattern, text):
        start = roman_to_int(c1)
        end = roman_to_int(c2)
        for c in range(start, end + 1):
            centuries.add(c)

    for year in years:
        centuries.add(year // 100 + 1)

    return {
        "years": sorted(years),
        "centuries": sorted(centuries)
    }


def extract_entities(text):
    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)

    if doc.spans is None:
        return []

    locations = set()
    people = set()
    orgs = set()

    for span in doc.spans:
        if span.type == 'LOC':
            locations.add(span.text)
        elif span.type == 'PER':
            people.add(span.text)
        elif span.type == 'ORG':
            orgs.add(span.text)

    return {'locs': list(locations),
            'people': list(people),
            'orgs': list(orgs)}

def collect_metadata(text):
    dates = extract_dates(text)
    entities = extract_entities(text)
    return (dates | entities)


'''
Parsing
'''

def remove_all_refs_from_wikicode(wikicode):
    """
    Убирает все <ref> теги и self-closing refs прямо из Wikicode.
    Возвращает текст без референсов.
    """
    text = str(wikicode)

    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

    text = re.sub(r'<ref[^>]*/>', '', text)

    text = re.sub(r'\[\d+(?:–\d+)?\]', '', text)

    return text


def get_sections(page_text):
    wikicode = mwparserfromhell.parse(page_text)

    clean_text = remove_all_refs_from_wikicode(wikicode)

    # парсим уже очищенный текст заново
    wikicode = mwparserfromhell.parse(clean_text)

    sections = []
    for section in wikicode.get_sections(include_lead=True, include_headings=True):
        heading = section.filter_headings()
        heading_text = heading[0].title if heading else "Lead"
        text = section.strip_code().strip()
        if text:
            sections.append({"heading": heading_text, "text": text})
    return sections


def crawl_category(site, cat_names, max_depth=2):
    visited = set()
    articles = set()

    def _crawl(cat_name, depth):
        if depth > max_depth or cat_name in visited:
            return
        visited.add(cat_name)

        cat = site.categories[cat_name]

        for page in cat.members(namespace=0):  # только статьи
            articles.add(page.name)

        for subcat in cat.members(namespace=14):  # только категории
            _crawl(subcat.name[10:], depth + 1)

    for root_cat in cat_names:
        _crawl(root_cat, 0)

    return list(articles)


'''
Chunking
'''

def chunk_text_by_tokens(text, tokenizer, max_tokens=512, overlap_tokens=50):
    """
    Разбивает текст на чанки по токенам токенизатора.
    Сохраняет смысловые границы (по абзацам).
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.tokenize(para))

        if current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                current_chunk_tokens = tokenizer.tokenize(chunk_text)
                overlap_text_tokens = current_chunk_tokens[-overlap_tokens:]
                overlap_text = tokenizer.convert_tokens_to_string(overlap_text_tokens)

                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(overlap_text_tokens)

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def collect_json():
    site = mwclient.Site('ru.wikipedia.org')

    seed_categories = [
        "История России",
        "История СССР",
        "История России до VIII века",
        "Средневековая Россия",
        "Русское государство",
        "Российская империя",
        "История России 1917—1991",
        "Гражданская война в России",
        "Древняя Русь"
    ]

    articles = crawl_category(site, seed_categories, max_depth=1)
    print(f"Найдено статей: {len(articles)}")

    tokenizer = AutoTokenizer.from_pretrained("sergeyzh/BERTA")

    rag_chunks = []

    doc_id = 0

    for title in tqdm(articles):
        try:
            page = site.pages[title]
            page_url = f"https://{site.host}/wiki/{page.name.replace(' ', '_')}"

            sections = get_sections(page.text())
            chunk_id = 0
            for sec in sections:
                if sec['heading'] == ' См. также ' or sec['heading'] == ' Источники ' or sec['heading'] == ' Примечания ':
                    continue
                
                text_chunks = chunk_text_by_tokens(sec["text"], tokenizer, max_tokens=512, overlap_tokens=50)

                for chunk_text_content in text_chunks:
                    meta_chunk = collect_metadata(chunk_text_content)
                    if meta_chunk['centuries']:
                        if meta_chunk['centuries'][-1] == 21:
                            continue
                    chunk = ({
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "page_url": page_url,
                        "text": chunk_text_content
                    } | meta_chunk)
                    rag_chunks.append(chunk)
                    chunk_id += 1

            doc_id += 1

        except Exception as e:
            print(f"Ошибка при обработке {title}: {e}")

    with open("rag_history_russia.jsonl", "w", encoding="utf-8") as f:
        for chunk in rag_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Pipeline завершен. Файл rag_history_russia.jsonl готов.")

if __name__ == "__main__":
    collect_json()