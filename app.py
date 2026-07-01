#!/usr/bin/env python3
"""
Srpski STT + lokalna vektorska pretraga projektnih statusa i HR podataka

Aplikacija koristi OpenAI API za speech-to-text i embeddings,
a lokalnu ChromaDB bazu za semantičku pretragu:
- Projektni statusi vozila (project_statuses kolekcija)
- HR zapisi zaposlenih (hr_records kolekcija)
"""

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from benchmark_tests import HR_BENCHMARK_TESTS, PROJECT_BENCHMARK_TESTS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PROJECT_STATUS_JSONL = os.getenv("PROJECT_STATUS_JSONL", "./status_projekata_vector_records.jsonl")
HR_RECORDS_JSONL = os.getenv("HR_RECORDS_JSONL", "./hr_records_vector_records.jsonl")
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

PROJECT_COLLECTION_NAME = "project_statuses"
HR_COLLECTION_NAME = "hr_records"
OLD_COLLECTION_NAME = "people"

PROJECT_INTENT = "project_status"
HR_INTENT = "hr"
UNKNOWN_INTENT = "unknown"
MIXED_INTENT = "mixed"

PROJECT_KEYWORDS = [
    "projekat", "project", "pasars", "mrap", "lot", "vozilo", "vozila",
    "realizacija", "procenat", "status projekta", "zavrseno",
    "koliko je zavrseno", "procenat realizacije",
]

HR_KEYWORDS = [
    "zaposleni", "radnik", "ugovor", "istek ugovora", "sektor",
    "radno mesto", "prestanak", "otkaz", "opomena", "destimulacija",
    "ljudski resursi", "hr", "kadrovi", "sledeca aktivnost", "sledeća aktivnost",
    "ko je", "ko radi", "kome istice", "na odredjeno", "na neodredjeno",
    "na određeno", "na neodređeno", "sistem administrator", "elektromehanicar",
    "bravar", "inzenjer",
    "disciplinska", "disciplinski", "disciplinsku", "napomena", "napomenu",
    "izostanak", "izostanci", "izostancima", "odsutnost",
    "probni rad", "probnom radu", "probnog rada",
]

CYR_TO_LAT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj",
    "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S",
    "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č",
    "Џ": "Dž", "Ш": "Š",
}

LAT_TO_CIR_MAP = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "đ": "ђ", "e": "е",
    "ž": "ж", "z": "з", "i": "и", "j": "ј", "k": "к", "l": "л", "lj": "љ",
    "m": "м", "n": "н", "nj": "њ", "o": "о", "p": "п", "r": "р", "s": "с",
    "t": "т", "ć": "ћ", "u": "у", "f": "ф", "h": "х", "c": "ц", "č": "ч",
    "dž": "џ", "š": "ш",
    "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "Đ": "Ђ", "E": "Е",
    "Ž": "Ж", "Z": "З", "I": "И", "J": "Ј", "K": "К", "L": "Л", "Lj": "Љ",
    "M": "М", "N": "Н", "Nj": "Њ", "O": "О", "P": "П", "R": "Р", "S": "С",
    "T": "Т", "Ć": "Ћ", "U": "У", "F": "Ф", "H": "Х", "C": "Ц", "Č": "Ч",
    "Dž": "Џ", "Ш": "Ш",
}

SERBIAN_WORD_TO_NUMBER = {
    "jedan": 1, "dva": 2, "tri": 3, "četiri": 4, "cetiri": 4, "pet": 5,
    "šest": 6, "sest": 6, "sedam": 7, "osam": 8, "devet": 9, "deset": 10,
    "један": 1, "два": 2, "три": 3, "четири": 4, "пет": 5,
    "шест": 6, "седам": 7, "осам": 8, "девет": 9, "десет": 10,
    "prvi": 1, "drugi": 2, "treci": 3, "treći": 3, "cetvrti": 4, "četvrti": 4,
    "peti": 5, "šesti": 6, "sesti": 6, "sedmi": 7, "osmi": 8, "deveti": 9, "deseti": 10,
}

SERBIAN_MONTHS = {
    "januar": 1, "januaru": 1, "januara": 1,
    "februar": 2, "februaru": 2, "februara": 2,
    "mart": 3, "martu": 3, "marta": 3,
    "april": 4, "aprilu": 4, "aprila": 4,
    "maj": 5, "maju": 5, "maja": 5,
    "jun": 6, "junu": 6, "juna": 6,
    "jul": 7, "julu": 7, "jula": 7,
    "avgust": 8, "avgustu": 8, "avgusta": 8,
    "septembar": 9, "septembru": 9, "septembra": 9,
    "oktobar": 10, "oktobru": 10, "oktobra": 10,
    "novembar": 11, "novembru": 11, "novembra": 11,
    "decembar": 12, "decembru": 12, "decembra": 12,
}

client: OpenAI = None
chroma_client: chromadb.PersistentClient = None
project_collection: chromadb.Collection = None
hr_collection: chromadb.Collection = None

HR_EMPLOYEE_CACHE: list[dict] = []


@dataclass
class RetrievalResult:
    """Rezultat retrieval operacije."""
    status: str
    results: list[dict] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    query: str = ""
    rewritten_query: str = ""
    collection_used: str = ""
    intent: str = ""
    intent_confidence: str = ""


def check_api_key():
    """Provera da li je OPENAI_API_KEY podešen."""
    if not OPENAI_API_KEY:
        print("=" * 60)
        print("GREŠKA: OPENAI_API_KEY nije podešen!")
        print()
        print("Koraci za podešavanje:")
        print("  1. Kopiraj .env.example u .env:")
        print("     cp .env.example .env")
        print()
        print("  2. Otvori .env i dodaj svoj OpenAI API ključ:")
        print("     OPENAI_API_KEY=sk-...")
        print("=" * 60)
        sys.exit(1)


def cyr_to_lat(text: str) -> str:
    """Konvertuje srpsku ćirilicu u latinicu."""
    result = []
    for char in text:
        result.append(CYR_TO_LAT_MAP.get(char, char))
    return "".join(result)


def lat_to_cyr(text: str) -> str:
    """Konvertuje latinicu u cirilicu."""
    result = []
    for char in text:
        result.append(LAT_TO_CIR_MAP.get(char, char))
    return "".join(result)

def remove_diacritics(text: str) -> str:
    """Uklanja dijakritike (č→c, š→s, ž→z, ć→c, đ→d)."""
    replacements = {
        "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d",
        "Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "D",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def normalize_text(value: str) -> str:
    """
    Normalizuje tekst za poređenje:
    - trimuje
    - lowercase
    - ćirilica → latinica
    - uklanja višak razmaka
    - uklanja dijakritike
    """
    if not value:
        return ""
    text = value.strip()
    text = cyr_to_lat(text)
    text = text.lower()
    text = " ".join(text.split())
    text = remove_diacritics(text)
    return text


def strip_serbian_case_ending(name: str) -> str:
    """
    Uklanja srpske padežne nastavke sa imena.
    npr. "jelenu" -> "jelen", "goranom" -> "goran"
    """
    case_endings = ["om", "em", "oj", "im", "u", "a", "i", "e"]
    name_lower = name.lower()
    for ending in case_endings:
        if name_lower.endswith(ending) and len(name_lower) > len(ending) + 2:
            return name_lower[:-len(ending)]
    return name_lower


def names_match_fuzzy(query_name: str, stored_name: str) -> bool:
    """
    Proverava da li se imena podudaraju uzimajući u obzir padeze.
    Zahteva minimum 4 karaktera za fuzzy matching da izbegne lažne pozitive.
    """
    query_norm = normalize_text(query_name)
    stored_norm = normalize_text(stored_name)
    
    if len(query_norm) < 4:
        return query_norm == stored_norm
    
    if query_norm == stored_norm:
        return True
    
    if len(query_norm) >= 4 and (query_norm in stored_norm or stored_norm in query_norm):
        return True
    
    query_stem = strip_serbian_case_ending(query_norm)
    stored_stem = strip_serbian_case_ending(stored_norm)
    
    if len(query_stem) >= 4 and query_stem == stored_stem:
        return True
    
    if len(query_stem) >= 5 and len(stored_stem) >= 5:
        if stored_stem.startswith(query_stem) or query_stem.startswith(stored_stem):
            return True
    
    return False


def parse_serbian_number(text: str) -> int | None:
    """Parsira broj iz teksta (cifra ili srpska reč)."""
    text = text.strip().lower()
    text_lat = cyr_to_lat(text)
    text_no_dia = remove_diacritics(text_lat)
    
    if text.isdigit():
        return int(text)
    
    if text_lat in SERBIAN_WORD_TO_NUMBER:
        return SERBIAN_WORD_TO_NUMBER[text_lat]
    if text_no_dia in SERBIAN_WORD_TO_NUMBER:
        return SERBIAN_WORD_TO_NUMBER[text_no_dia]
    if text in SERBIAN_WORD_TO_NUMBER:
        return SERBIAN_WORD_TO_NUMBER[text]
    
    return None


def delete_old_collections():
    """Briše staru kolekciju 'people' ako postoji."""
    global chroma_client
    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if OLD_COLLECTION_NAME in existing_collections:
            chroma_client.delete_collection(OLD_COLLECTION_NAME)
            print(f"Obrisana stara kolekcija: {OLD_COLLECTION_NAME}")
    except Exception as e:
        print(f"Upozorenje pri brisanju stare kolekcije: {e}")


def reset_collection(collection_name: str):
    """Briše i ponovo kreira kolekciju."""
    global chroma_client
    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if collection_name in existing_collections:
            chroma_client.delete_collection(collection_name)
            print(f"Obrisana kolekcija za reset: {collection_name}")
    except Exception as e:
        print(f"Upozorenje pri resetovanju kolekcije: {e}")


def load_project_status_records() -> list[dict]:
    """
    Učitava projektne statuse iz JSONL fajla.
    Validira obavezna polja i preskače nevalidne zapise.
    """
    if not os.path.exists(PROJECT_STATUS_JSONL):
        print(f"GREŠKA: Fajl {PROJECT_STATUS_JSONL} ne postoji!")
        sys.exit(1)
    
    required_fields = ["id", "text", "project", "lot", "vehicle_number", "completion_percent"]
    records = []
    
    with open(PROJECT_STATUS_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"UPOZORENJE: Red {line_num} nije validan JSON: {e}")
                continue
            
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                print(f"UPOZORENJE: Red {line_num} nema obavezna polja: {missing_fields}")
                continue
            
            records.append(record)
    
    return records


def load_hr_records() -> list[dict]:
    """
    Učitava HR zapise iz JSONL fajla.
    Validira obavezna polja i preskače nevalidne zapise.
    """
    if not os.path.exists(HR_RECORDS_JSONL):
        print(f"UPOZORENJE: Fajl {HR_RECORDS_JSONL} ne postoji!")
        return []
    
    required_fields = [
        "id",
        "record_type",
        "employee_name",
        "employee_name_normalized",
        "department",
        "employment_status",
        "position",
        "text",
    ]
    records = []
    
    with open(HR_RECORDS_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"UPOZORENJE: HR red {line_num} nije validan JSON: {e}")
                continue
            
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                print(f"UPOZORENJE: HR red {line_num} nema obavezna polja: {missing_fields}")
                continue
            
            records.append(record)
    
    return records


def build_hr_document(record: dict) -> str:
    """
    Pravi semantički bogat tekst za embedding od HR zapisa.
    Uključuje ime na ćirilici i latinici, radno mesto, sektor, status, datume.
    """
    parts = [
        f"HR zapis za zaposlenog {record.get('employee_name', '')} ",
        f"({record.get('employee_name_latin', '')}).",
    ]
    
    if record.get("department"):
        parts.append(f" Zaposleni radi u sektoru {record['department']}.")
    
    if record.get("position"):
        parts.append(f" Radno mesto je {record['position']}.")
    
    if record.get("employment_status"):
        parts.append(f" Status zaposlenja ili ugovora je {record['employment_status']}.")
    
    contract_start = record.get("contract_start_date_original", record.get("contract_start_date", ""))
    if contract_start:
        parts.append(f" Datum početka ugovora je {contract_start}.")
    
    contract_end = record.get("contract_end_date_original", record.get("contract_end_date", ""))
    if contract_end:
        parts.append(f" Datum isteka ugovora je {contract_end}.")
    
    if record.get("termination_basis"):
        parts.append(f" Osnov prestanka radnog odnosa ili HR napomena: {record['termination_basis']}.")
    
    if record.get("next_action"):
        parts.append(f" Sledeća HR aktivnost: {record['next_action']}.")
    
    if record.get("text"):
        parts.append(f" {record['text']}")
    
    return "".join(parts)


def build_hr_metadata(record: dict) -> dict:
    """Pravi metadata za Chroma HR zapis (samo scalar vrednosti)."""
    status = record.get("employment_status", "")
    status_normalized = normalize_text(status)
    
    department = record.get("department", "")
    department_normalized = normalize_text(department)
    
    position = record.get("position", "")
    position_normalized = normalize_text(position)
    
    return {
        "record_type": record.get("record_type", "hr_employee_contract"),
        "employee_name": record.get("employee_name", ""),
        "employee_name_latin": record.get("employee_name_latin", ""),
        "employee_name_normalized": record.get("employee_name_normalized", ""),
        "department": department,
        "department_normalized": department_normalized,
        "employment_status": status,
        "employment_status_normalized": status_normalized,
        "position": position,
        "position_normalized": position_normalized,
        "contract_start_date": record.get("contract_start_date") or "",
        "contract_end_date": record.get("contract_end_date") or "",
        "termination_basis": record.get("termination_basis", ""),
        "next_action": record.get("next_action", ""),
        "source_file": record.get("source_file", ""),
        "source_sheet": record.get("source_sheet", ""),
        "source_range": record.get("source_range", ""),
    }


def build_project_status_document(record: dict) -> str:
    """
    Pravi tekstualni dokument za embedding od zapisa projektnog statusa.
    Uključuje semantičke varijante za bolje pronalaženje.
    """
    project = record["project"]
    lot = record["lot"]
    vehicle = record["vehicle_number"]
    percent = record["completion_percent"]
    
    aliases = record.get("project_aliases", [])
    aliases_text = ", ".join(aliases) if aliases else project
    
    doc_parts = [
        f"Projekat {project}.",
        f"Alias nazivi projekta: {aliases_text}.",
        f"Lot {lot}.",
        f"Vozilo broj {vehicle}.",
        f"Vozilo {vehicle} u lotu {lot} je na {percent}% realizacije.",
        f"Status vozila broj {vehicle} za projekat {project} je {percent} procenata.",
        f"Projekat {project} lot {lot} vozilo {vehicle} procenat {percent}.",
        f"Realizacija vozila {vehicle} na projektu {project} u lotu {lot} iznosi {percent}%.",
    ]
    
    if record.get("text"):
        doc_parts.append(record["text"])
    
    return " ".join(doc_parts)


def build_metadata(record: dict) -> dict:
    """Pravi metadata za Chroma zapis."""
    aliases = record.get("project_aliases", [])
    aliases_str = ",".join(aliases) if aliases else ""
    
    metadata = {
        "project": record["project"],
        "project_normalized": normalize_text(record["project"]),
        "lot": int(record["lot"]),
        "vehicle_number": int(record["vehicle_number"]),
        "completion_percent": int(record["completion_percent"]),
        "record_type": "project_vehicle_status",
    }
    
    if aliases_str:
        metadata["project_aliases"] = aliases_str
    if record.get("source_file"):
        metadata["source_file"] = record["source_file"]
    if record.get("source_sheet"):
        metadata["source_sheet"] = record["source_sheet"]
    if record.get("source_range"):
        metadata["source_range"] = record["source_range"]
    
    return metadata


def get_embedding(text: str) -> list[float]:
    """Dobija embedding za jedan tekst."""
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Dobija embeddings za listu tekstova u jednom batch pozivu."""
    if not texts:
        return []
    
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def seed_project_database(reset: bool = True):
    """
    Inicijalizuje ChromaDB bazu sa projektnim statusima iz JSONL fajla.
    
    Args:
        reset: Ako je True, briše postojeću kolekciju pre seedovanja.
    """
    global project_collection, chroma_client
    
    delete_old_collections()
    
    if reset:
        reset_collection(PROJECT_COLLECTION_NAME)
    
    project_collection = chroma_client.get_or_create_collection(
        name=PROJECT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    records = load_project_status_records()
    print(f"Učitano {len(records)} projektnih zapisa iz {PROJECT_STATUS_JSONL}")
    
    if not records:
        print("Nema projektnih zapisa za učitavanje!")
        return
    
    ids = []
    documents = []
    metadatas = []
    
    for record in records:
        ids.append(record["id"])
        documents.append(build_project_status_document(record))
        metadatas.append(build_metadata(record))
    
    print(f"Generisanje embeddings za {len(documents)} projektnih dokumenata...")
    
    batch_size = 100
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}...")
        batch_embeddings = get_embeddings(batch)
        embeddings.extend(batch_embeddings)
    
    project_collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Upisano {len(ids)} zapisa u ChromaDB kolekciju '{PROJECT_COLLECTION_NAME}'")


def seed_hr_database(reset: bool = True):
    """
    Inicijalizuje ChromaDB bazu sa HR zapisima iz JSONL fajla.
    
    Args:
        reset: Ako je True, briše postojeću kolekciju pre seedovanja.
    """
    global hr_collection, chroma_client, HR_EMPLOYEE_CACHE
    
    if reset:
        reset_collection(HR_COLLECTION_NAME)
    
    hr_collection = chroma_client.get_or_create_collection(
        name=HR_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    records = load_hr_records()
    print(f"Učitano {len(records)} HR zapisa iz {HR_RECORDS_JSONL}")
    
    if not records:
        print("Nema HR zapisa za učitavanje!")
        return
    
    HR_EMPLOYEE_CACHE = [
        {
            "employee_name": r.get("employee_name", ""),
            "employee_name_latin": r.get("employee_name_latin", ""),
            "employee_name_normalized": r.get("employee_name_normalized", ""),
        }
        for r in records
    ]
    
    ids = []
    documents = []
    metadatas = []
    
    for record in records:
        ids.append(record["id"])
        documents.append(build_hr_document(record))
        metadatas.append(build_hr_metadata(record))
    
    print(f"Generisanje embeddings za {len(documents)} HR dokumenata...")
    
    batch_size = 100
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}...")
        batch_embeddings = get_embeddings(batch)
        embeddings.extend(batch_embeddings)
    
    hr_collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Upisano {len(ids)} zapisa u ChromaDB kolekciju '{HR_COLLECTION_NAME}'")


def seed_all_databases(reset: bool = True):
    """
    Seeduje obe kolekcije: projektne statuse i HR zapise.
    
    Args:
        reset: Ako je True, briše postojeće kolekcije pre seedovanja.
    """
    seed_project_database(reset=reset)
    seed_hr_database(reset=reset)


def init_hr_employee_cache():
    """Inicijalizuje cache zaposlenih iz HR kolekcije."""
    global HR_EMPLOYEE_CACHE, hr_collection
    
    if hr_collection is None:
        return
    
    try:
        all_hr = hr_collection.get(include=["metadatas"])
        HR_EMPLOYEE_CACHE = [
            {
                "employee_name": meta.get("employee_name", ""),
                "employee_name_latin": meta.get("employee_name_latin", ""),
                "employee_name_normalized": meta.get("employee_name_normalized", ""),
            }
            for meta in all_hr.get("metadatas", [])
        ]
        print(f"Inicijalizovan HR cache sa {len(HR_EMPLOYEE_CACHE)} zaposlenih")
    except Exception as e:
        print(f"Greška pri inicijalizaciji HR cache: {e}")


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Broji koliko ključnih reči se nalazi u tekstu."""
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        if keyword.lower() in text_lower:
            count += 1
    return count


def extract_hr_query_entities(question: str) -> dict:
    """
    Izvlači HR entitete iz pitanja:
    - ime zaposlenog
    - sektor
    - status ugovora
    - pitanja o isteku/prestanku
    - specifični termini (opomena, probni rad, disciplinska, izostanci)
    """
    entities = {
        "employee_name": None,
        "employee_name_normalized": None,
        "department_query": None,
        "position_query": None,
        "contract_query": None,
        "termination_query": None,
        "termination_keyword": None,
        "status_query": None,
        "date_month": None,
        "date_year": None,
    }
    
    q_normalized = normalize_text(question)
    q_words = q_normalized.split()
    
    for emp in HR_EMPLOYEE_CACHE:
        emp_norm = emp.get("employee_name_normalized", "")
        if emp_norm and emp_norm in q_normalized:
            entities["employee_name"] = emp.get("employee_name", "")
            entities["employee_name_normalized"] = emp_norm
            break
        
        emp_latin = normalize_text(emp.get("employee_name_latin", ""))
        if emp_latin and emp_latin in q_normalized:
            entities["employee_name"] = emp.get("employee_name", "")
            entities["employee_name_normalized"] = emp.get("employee_name_normalized", "")
            break
        
        name_parts = emp_norm.split()
        if len(name_parts) >= 2:
            if all(part in q_normalized for part in name_parts):
                entities["employee_name"] = emp.get("employee_name", "")
                entities["employee_name_normalized"] = emp_norm
                break
            
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            first_match = any(names_match_fuzzy(word, first_name) for word in q_words)
            last_match = any(names_match_fuzzy(word, last_name) for word in q_words) if last_name else False
            
            if first_match and last_match:
                entities["employee_name"] = emp.get("employee_name", "")
                entities["employee_name_normalized"] = emp_norm
                break
    
    department_patterns = [
        r"(?:u |iz )?(?:sektoru?|sluzb[aei]|direkcij[aei])\s+(\w+(?:\s+\w+)*)",
        r"(?:radi u|zaposlen u)\s+(.+?)(?:\s+kao|\s+na|\?|$)",
        r"(?:u |iz )?it\s*sektor",
        r"(?:u |iz )?(?:sluzbi?)\s+(\w+)",
    ]
    for pattern in department_patterns:
        match = re.search(pattern, q_normalized, re.IGNORECASE)
        if match:
            dept = match.group(1).strip() if match.lastindex else match.group(0).strip()
            dept = re.sub(r"^(u|iz)\s+", "", dept)
            dept = re.sub(r"u$", "", dept).strip()
            entities["department_query"] = dept
            break
    
    if "it sektor" in q_normalized or "it sektoru" in q_normalized:
        entities["department_query"] = "it sektor"
    
    if "bezbednost" in q_normalized or "bzr" in q_normalized or "zdravlje na radu" in q_normalized:
        entities["department_query"] = "bezbednost"
    
    if "korisnicka podrska" in q_normalized or "korisnickoj podrsci" in q_normalized:
        entities["department_query"] = "korisnicka podrska"
    
    position_keywords = {
        "sistem administrator": ["систем администратор", "sistem administrator"],
        "vodja smene": ["вођа смене", "vodja smene", "voda smene"],
        "inzenjer bzr": ["инжењер бзр", "inzenjer bzr", "bzr"],
        "hr specijalista": ["hr специјалиста", "hr specijalista"],
        "finansijski analiticar": ["финансијски аналитичар", "finansijski analiticar"],
    }
    
    for pos_key, pos_variants in position_keywords.items():
        if pos_key in q_normalized:
            entities["position_query"] = pos_key
            break
        for variant in pos_variants:
            if variant.lower() in q_normalized:
                entities["position_query"] = pos_key
                break
        if entities["position_query"]:
            break
    
    contract_patterns = [
        r"(?:kome |ko ima |ciji )(?:istice|istekao|zavrsava se) ugovor",
        r"istek(?:ao)? ugovor",
        r"ugovor istice",
        r"kada istice ugovor",
    ]
    for pattern in contract_patterns:
        if re.search(pattern, q_normalized, re.IGNORECASE):
            entities["contract_query"] = "expiring"
            break
    
    status_odredjeno_patterns = [
        r"na odre[dđ]eno", r"na određeno", r"status.*odre[dđ]eno",
        r"ko je na odre[dđ]eno", r"koji su na odre[dđ]eno",
        r"odredeno", r"odredjeno",
    ]
    status_neodredjeno_patterns = [
        r"na neodre[dđ]eno", r"na neodređeno", r"status.*neodre[dđ]eno",
        r"ko je na neodre[dđ]eno", r"koji su na neodre[dđ]eno",
        r"neodredeno", r"neodredjeno",
    ]
    
    for pattern in status_odredjeno_patterns:
        if re.search(pattern, q_normalized, re.IGNORECASE):
            entities["status_query"] = "odredjeno"
            break
    
    if not entities["status_query"]:
        for pattern in status_neodredjeno_patterns:
            if re.search(pattern, q_normalized, re.IGNORECASE):
                entities["status_query"] = "neodredjeno"
                break
    
    termination_patterns = [
        r"(?:ima |ko ima |ima li )(?:opomenu|destimulacij[au]|otkaz)",
        r"opomen[au] pred otkaz",
        r"destimulacij[au]",
        r"osnov prestanka",
    ]
    for pattern in termination_patterns:
        if re.search(pattern, q_normalized, re.IGNORECASE):
            entities["termination_query"] = "has_termination_basis"
            break
    
    specific_termination_keywords = {
        "opomena pred otkaz": ["опомена пред отказ", "opomena pred otkaz"],
        "probni rad": ["пробни рад", "probni rad", "пробног рада", "probnom radu", "probnog rada"],
        "disciplinska": ["дисциплинска", "disciplinska", "дисциплинску", "disciplinsku", "disciplinskom"],
        "izostanak": ["изостана", "izostana", "izostanci", "изостанци", "izostancima"],
    }
    
    for key, keywords in specific_termination_keywords.items():
        if key in q_normalized:
            entities["termination_keyword"] = key
            entities["termination_query"] = "has_termination_basis"
            break
        for kw in keywords:
            if kw.lower() in q_normalized:
                entities["termination_keyword"] = key
                entities["termination_query"] = "has_termination_basis"
                break
        if entities["termination_keyword"]:
            break
    
    for month_name, month_num in SERBIAN_MONTHS.items():
        if month_name in q_normalized:
            entities["date_month"] = month_num
            break
    
    year_match = re.search(r"20\d{2}", question)
    if year_match:
        entities["date_year"] = int(year_match.group())
    
    return entities


def detect_intent(question: str) -> dict:
    """
    Određuje intent pitanja: project_status, hr, mixed ili unknown.
    Koristi kombinaciju keyword matching i entity extraction.
    """
    q_normalized = normalize_text(question)
    q_lat = cyr_to_lat(question.lower())
    
    project_score = count_keyword_matches(q_normalized, PROJECT_KEYWORDS)
    project_score += count_keyword_matches(q_lat, PROJECT_KEYWORDS)
    
    hr_score = count_keyword_matches(q_normalized, HR_KEYWORDS)
    hr_score += count_keyword_matches(q_lat, HR_KEYWORDS)
    
    project_entities = extract_project_query_entities(question)
    hr_entities = extract_hr_query_entities(question)
    
    if project_entities.get("project") or project_entities.get("vehicle_number") or project_entities.get("lot"):
        project_score += 3
    
    if hr_entities.get("employee_name") or hr_entities.get("department_query") or hr_entities.get("contract_query"):
        hr_score += 3
    
    if hr_entities.get("status_query") or hr_entities.get("termination_query"):
        hr_score += 2
    
    if project_score > 0 and hr_score == 0:
        return {"intent": PROJECT_INTENT, "confidence": "high", "project_score": project_score, "hr_score": hr_score}
    
    if hr_score > 0 and project_score == 0:
        return {"intent": HR_INTENT, "confidence": "high", "project_score": project_score, "hr_score": hr_score}
    
    if project_score > hr_score:
        confidence = "high" if project_score > hr_score + 2 else "medium"
        return {"intent": PROJECT_INTENT, "confidence": confidence, "project_score": project_score, "hr_score": hr_score}
    
    if hr_score > project_score:
        confidence = "high" if hr_score > project_score + 2 else "medium"
        return {"intent": HR_INTENT, "confidence": confidence, "project_score": project_score, "hr_score": hr_score}
    
    if project_score > 0 and hr_score > 0:
        return {"intent": MIXED_INTENT, "confidence": "low", "project_score": project_score, "hr_score": hr_score}
    
    return {"intent": UNKNOWN_INTENT, "confidence": "low", "project_score": project_score, "hr_score": hr_score}


def extract_project_query_entities(question: str) -> dict:
    """
    Izvlači entitete iz pitanja: project, lot, vehicle_number.
    Podržava cifre i srpske reči (latinica i ćirilica).
    """
    entities = {
        "project": None,
        "project_normalized": None,
        "lot": None,
        "vehicle_number": None,
    }
    
    q_lower = question.lower()
    q_lat = cyr_to_lat(q_lower)
    q_normalized = normalize_text(question)
    
    known_projects = ["pasars", "mrap", "ntv"]
    for proj in known_projects:
        if proj in q_normalized:
            entities["project"] = proj.upper()
            entities["project_normalized"] = proj
            break
    
    lot_patterns = [
        r"lot[ua]?\s*(?:broj[a]?)?\s*(\d+|jedan|dva|tri|četiri|cetiri|pet|šest|sest|sedam|osam|devet|deset)",
        r"лот[уа]?\s*(?:број[а]?)?\s*(\d+|један|два|три|четири|пет|шест|седам|осам|девет|десет)",
        r"lot\s+(\w+)",
    ]
    
    for pattern in lot_patterns:
        match = re.search(pattern, q_lat, re.IGNORECASE)
        if not match:
            match = re.search(pattern, question, re.IGNORECASE)
        if match:
            lot_val = parse_serbian_number(match.group(1))
            if lot_val:
                entities["lot"] = lot_val
                break
    
    vehicle_patterns = [
        r"vozil[oa]\s*(?:broj[a]?)?\s*(\d+|jedan|dva|tri|četiri|cetiri|pet|šest|sest|sedam|osam|devet|deset)",
        r"возил[оа]\s*(?:број[а]?)?\s*(\d+|један|два|три|четири|пет|шест|седам|осам|девет|десет)",
        r"v(?:ozilo)?\s*(\d+)",
        r"vozilo\s+(\w+)",
    ]
    
    for pattern in vehicle_patterns:
        match = re.search(pattern, q_lat, re.IGNORECASE)
        if not match:
            match = re.search(pattern, question, re.IGNORECASE)
        if match:
            vehicle_val = parse_serbian_number(match.group(1))
            if vehicle_val:
                entities["vehicle_number"] = vehicle_val
                break
    
    return entities


def rewrite_query_for_retrieval(question: str, intent: str, entities: dict) -> str:
    """
    Prepisuje upit za bolji retrieval.
    Ne izmišlja entitete - samo obogaćuje kontekst.
    """
    if intent == PROJECT_INTENT:
        parts = []
        if entities.get("project"):
            parts.append(f"Projekat {entities['project']}")
        if entities.get("lot"):
            parts.append(f"lot {entities['lot']}")
        if entities.get("vehicle_number"):
            parts.append(f"vozilo broj {entities['vehicle_number']}")
        
        if parts:
            parts.append("procenat realizacije, status vozila")
            return ", ".join(parts) + "."
        
        return f"Status projekta, lot, vozilo, procenat realizacije. {question}"
    
    if intent == HR_INTENT:
        parts = ["HR zapis"]
        
        if entities.get("employee_name"):
            parts.append(f"zaposlenog {entities['employee_name']}")
        
        if entities.get("contract_query") == "expiring":
            parts.append("datum isteka ugovora, status ugovora, sledeća HR aktivnost")
        
        if entities.get("status_query") == "odredjeno":
            parts.append("zaposleni na određeno, status ugovora određeno")
        elif entities.get("status_query") == "neodredjeno":
            parts.append("zaposleni na neodređeno, status ugovora neodređeno")
        
        if entities.get("termination_query"):
            parts.append("osnov prestanka radnog odnosa, opomena pred otkaz, destimulacija, disciplinske mere")
        
        if entities.get("department_query"):
            parts.append(f"sektor {entities['department_query']}, radno mesto, zaposleni u sektoru")
        
        if len(parts) == 1:
            return f"HR zapisi zaposlenih, {question}"
        
        return ", ".join(parts) + "."
    
    return question


def retrieve_project_status(question: str) -> RetrievalResult:
    """
    Hibridni retrieval za projektne statuse: metadata lookup pa vector search.
    
    Vraća RetrievalResult sa:
    - status: exact_match | vehicle_found_in_different_lot | project_only | vector_fallback | no_result
    - results: lista pronađenih zapisa
    - entities: izvučeni entiteti iz pitanja
    """
    entities = extract_project_query_entities(question)
    rewritten = rewrite_query_for_retrieval(question, PROJECT_INTENT, entities)
    
    result = RetrievalResult(
        status="no_result",
        results=[],
        entities=entities,
        query=question,
        rewritten_query=rewritten,
        collection_used=PROJECT_COLLECTION_NAME,
        intent=PROJECT_INTENT,
        intent_confidence="high",
    )
    
    project_norm = entities.get("project_normalized")
    lot = entities.get("lot")
    vehicle = entities.get("vehicle_number")
    
    if project_norm and lot is not None and vehicle is not None:
        try:
            exact_results = project_collection.get(
                where={
                    "$and": [
                        {"project_normalized": {"$eq": project_norm}},
                        {"lot": {"$eq": lot}},
                        {"vehicle_number": {"$eq": vehicle}},
                    ]
                },
                include=["metadatas", "documents"],
            )
            
            if exact_results["ids"]:
                result.status = "exact_match"
                result.results = [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(
                        exact_results["ids"],
                        exact_results["metadatas"],
                        exact_results["documents"],
                    )
                ]
                return result
        except Exception as e:
            print(f"Greška pri exact lookup: {e}")
    
    if project_norm and vehicle is not None:
        try:
            vehicle_results = project_collection.get(
                where={
                    "$and": [
                        {"project_normalized": {"$eq": project_norm}},
                        {"vehicle_number": {"$eq": vehicle}},
                    ]
                },
                include=["metadatas", "documents"],
            )
            
            if vehicle_results["ids"]:
                result.status = "vehicle_found_in_different_lot"
                result.results = [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(
                        vehicle_results["ids"],
                        vehicle_results["metadatas"],
                        vehicle_results["documents"],
                    )
                ]
                return result
        except Exception as e:
            print(f"Greška pri vehicle lookup: {e}")
    
    if project_norm:
        try:
            project_results = project_collection.get(
                where={"project_normalized": {"$eq": project_norm}},
                include=["metadatas", "documents"],
            )
            
            if project_results["ids"]:
                result.status = "project_only"
                result.results = [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(
                        project_results["ids"],
                        project_results["metadatas"],
                        project_results["documents"],
                    )
                ]
                return result
        except Exception as e:
            print(f"Greška pri project lookup: {e}")
    
    try:
        query_embedding = get_embedding(rewritten)
        vector_results = project_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas", "documents", "distances"],
        )
        
        if vector_results["ids"] and vector_results["ids"][0]:
            result.status = "vector_fallback"
            result.results = [
                {
                    "id": id_,
                    "metadata": meta,
                    "document": doc,
                    "distance": dist,
                }
                for id_, meta, doc, dist in zip(
                    vector_results["ids"][0],
                    vector_results["metadatas"][0],
                    vector_results["documents"][0],
                    vector_results["distances"][0],
                )
            ]
            return result
    except Exception as e:
        print(f"Greška pri vector search: {e}")
    
    return result


def retrieve_hr_record(question: str) -> RetrievalResult:
    """
    Hibridni retrieval za HR zapise: metadata lookup pa vector search.
    
    Strategije:
    A) Exact employee match po employee_name_normalized
    B) Filter po contract_end_date za pitanja o isteku ugovora
    C) Filter po employment_status_normalized za pitanja o statusu
    D) Filter po termination_basis za pitanja o prestanku
    E) Filter po department_normalized za pitanja o sektoru
    F) Vector search fallback
    """
    hr_entities = extract_hr_query_entities(question)
    rewritten = rewrite_query_for_retrieval(question, HR_INTENT, hr_entities)
    
    result = RetrievalResult(
        status="no_result",
        results=[],
        entities=hr_entities,
        query=question,
        rewritten_query=rewritten,
        collection_used=HR_COLLECTION_NAME,
        intent=HR_INTENT,
        intent_confidence="high",
    )
    
    employee_norm = hr_entities.get("employee_name_normalized")
    if employee_norm:
        try:
            exact_results = hr_collection.get(
                where={"employee_name_normalized": {"$eq": employee_norm}},
                include=["metadatas", "documents"],
            )
            
            if exact_results["ids"]:
                result.status = "exact_employee_match"
                result.results = [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(
                        exact_results["ids"],
                        exact_results["metadatas"],
                        exact_results["documents"],
                    )
                ]
                return result
        except Exception as e:
            print(f"Greška pri exact employee lookup: {e}")
    
    status_query = hr_entities.get("status_query")
    if status_query:
        try:
            status_results = hr_collection.get(
                where={"employment_status_normalized": {"$eq": status_query}},
                include=["metadatas", "documents"],
            )
            
            if status_results["ids"]:
                result.status = "status_filter_match"
                result.results = [
                    {"id": id_, "metadata": meta, "document": doc}
                    for id_, meta, doc in zip(
                        status_results["ids"],
                        status_results["metadatas"],
                        status_results["documents"],
                    )
                ]
                return result
        except Exception as e:
            print(f"Greška pri status filter lookup: {e}")
    
    position_query = hr_entities.get("position_query")
    if position_query:
        try:
            all_hr = hr_collection.get(include=["metadatas", "documents"])
            
            position_filters = {
                "sistem administrator": ["систем администратор", "sistem administrator"],
                "vodja smene": ["вођа смене", "vodja smene"],
                "inzenjer bzr": ["инжењер бзр", "inzenjer bzr"],
            }
            
            filter_keywords = position_filters.get(position_query, [position_query])
            
            matching = []
            for id_, meta, doc in zip(all_hr["ids"], all_hr["metadatas"], all_hr["documents"]):
                position = meta.get("position", "")
                position_norm = normalize_text(position)
                if any(kw.lower() in position_norm or kw.lower() in position.lower() for kw in filter_keywords):
                    matching.append({"id": id_, "metadata": meta, "document": doc})
            
            if matching:
                result.status = "position_filter_match"
                result.results = matching
                return result
        except Exception as e:
            print(f"Greška pri position lookup: {e}")
    
    if hr_entities.get("contract_query") == "expiring" or hr_entities.get("date_month"):
        try:
            all_hr = hr_collection.get(include=["metadatas", "documents"])
            
            target_month = hr_entities.get("date_month")
            target_year = hr_entities.get("date_year")
            
            expiring = []
            for id_, meta, doc in zip(all_hr["ids"], all_hr["metadatas"], all_hr["documents"]):
                end_date = meta.get("contract_end_date", "")
                if end_date and end_date != "Нема истека":
                    if target_month or target_year:
                        date_match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", end_date)
                        if date_match:
                            day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                            month_matches = (target_month is None or month == target_month)
                            year_matches = (target_year is None or year == target_year)
                            if month_matches and year_matches:
                                expiring.append({"id": id_, "metadata": meta, "document": doc})
                    else:
                        expiring.append({"id": id_, "metadata": meta, "document": doc})
            
            expiring.sort(key=lambda x: x["metadata"].get("contract_end_date", "9999-99-99"))
            
            if expiring:
                result.status = "contract_expiring_filter"
                result.results = expiring
                return result
        except Exception as e:
            print(f"Greška pri contract expiring lookup: {e}")
    
    if hr_entities.get("termination_query"):
        try:
            all_hr = hr_collection.get(include=["metadatas", "documents"])
            
            specific_keyword = hr_entities.get("termination_keyword")
            keyword_filters = {
                "opomena pred otkaz": ["опомена пред отказ", "opomena pred otkaz"],
                "probni rad": ["пробни рад", "probni rad", "пробног рада"],
                "disciplinska": ["дисциплинска", "disciplinska"],
                "izostanak": ["изостана", "izostana", "изостанака"],
            }
            
            filter_keywords = keyword_filters.get(specific_keyword, []) if specific_keyword else []
            
            with_termination = []
            for id_, meta, doc in zip(all_hr["ids"], all_hr["metadatas"], all_hr["documents"]):
                termination_basis = meta.get("termination_basis", "")
                if termination_basis:
                    if filter_keywords:
                        termination_lower = termination_basis.lower()
                        if any(kw.lower() in termination_lower for kw in filter_keywords):
                            with_termination.append({"id": id_, "metadata": meta, "document": doc})
                    else:
                        with_termination.append({"id": id_, "metadata": meta, "document": doc})
            
            if with_termination:
                result.status = "termination_basis_filter"
                result.results = with_termination
                return result
        except Exception as e:
            print(f"Greška pri termination lookup: {e}")
    
    dept_query = hr_entities.get("department_query")
    if dept_query:
        dept_norm = normalize_text(dept_query)
        dept_words = dept_norm.split()
        
        dept_aliases = {
            "korisnicka podrska": ["корисничка подршка", "korisnicka podrska"],
            "bezbednost": ["безбедност и здравље на раду", "bezbednost", "bzr"],
            "it": ["it служба", "it sluzba", "it sektor"],
            "finansije": ["финансије", "finansije", "racunovodstvo"],
            "pravni": ["правни сектор", "pravni sektor"],
        }
        
        search_terms = [dept_norm] + dept_words
        for alias_key, alias_values in dept_aliases.items():
            if alias_key in dept_norm:
                search_terms.extend(alias_values)
        
        try:
            all_hr = hr_collection.get(include=["metadatas", "documents"])
            
            in_department = []
            for id_, meta, doc in zip(all_hr["ids"], all_hr["metadatas"], all_hr["documents"]):
                dept = meta.get("department", "")
                dept_normalized = normalize_text(dept)
                
                matched = False
                for term in search_terms:
                    term_lower = term.lower()
                    if term_lower in dept_normalized or term_lower in dept.lower():
                        matched = True
                        break
                
                if not matched and all(word in dept_normalized for word in dept_words):
                    matched = True
                
                if matched:
                    in_department.append({"id": id_, "metadata": meta, "document": doc})
            
            if in_department:
                result.status = "department_filter_match"
                result.results = in_department
                return result
        except Exception as e:
            print(f"Greška pri department lookup: {e}")
    
    try:
        query_embedding = get_embedding(rewritten)
        vector_results = hr_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas", "documents", "distances"],
        )
        
        if vector_results["ids"] and vector_results["ids"][0]:
            result.status = "vector_fallback"
            result.results = [
                {
                    "id": id_,
                    "metadata": meta,
                    "document": doc,
                    "distance": dist,
                }
                for id_, meta, doc, dist in zip(
                    vector_results["ids"][0],
                    vector_results["metadatas"][0],
                    vector_results["documents"][0],
                    vector_results["distances"][0],
                )
            ]
            return result
    except Exception as e:
        print(f"Greška pri HR vector search: {e}")
    
    return result


def generate_project_status_response(retrieval_result: RetrievalResult) -> str:
    """
    Generiše odgovor na osnovu rezultata pretrage projektnih statusa.
    Ne halucinira - koristi samo pronađene podatke.
    """
    status = retrieval_result.status
    results = retrieval_result.results
    entities = retrieval_result.entities
    
    if status == "no_result":
        parts = []
        if entities.get("project"):
            parts.append(f"projekat {entities['project']}")
        if entities.get("lot"):
            parts.append(f"lot {entities['lot']}")
        if entities.get("vehicle_number"):
            parts.append(f"vozilo {entities['vehicle_number']}")
        
        if parts:
            return f"Nemam podataka za {', '.join(parts)}."
        return "Nisam pronašao relevantne podatke u bazi."
    
    if status == "exact_match":
        r = results[0]
        meta = r["metadata"]
        return (
            f"Za projekat {meta['project']}, vozilo broj {meta['vehicle_number']} "
            f"u lotu {meta['lot']} je na {meta['completion_percent']}% realizacije."
        )
    
    if status == "vehicle_found_in_different_lot":
        r = results[0]
        meta = r["metadata"]
        requested_lot = entities.get("lot")
        actual_lot = meta["lot"]
        
        if requested_lot and requested_lot != actual_lot:
            return (
                f"Za projekat {meta['project']}, vozilo broj {meta['vehicle_number']} "
                f"nije u lotu {requested_lot}, nego u lotu {actual_lot}. "
                f"Njegova realizacija je {meta['completion_percent']}%."
            )
        return (
            f"Za projekat {meta['project']}, vozilo broj {meta['vehicle_number']} "
            f"je u lotu {actual_lot} i na {meta['completion_percent']}% realizacije."
        )
    
    if status == "project_only":
        project = entities.get("project", "")
        lines = [f"Pronađena vozila za projekat {project}:"]
        
        sorted_results = sorted(results, key=lambda x: (x["metadata"]["lot"], x["metadata"]["vehicle_number"]))
        
        for r in sorted_results:
            meta = r["metadata"]
            lines.append(
                f"• Lot {meta['lot']}, vozilo {meta['vehicle_number']}: {meta['completion_percent']}%"
            )
        
        return "\n".join(lines)
    
    if status == "vector_fallback":
        lines = ["Na osnovu semantičke pretrage, pronađeni su sledeći rezultati:"]
        for r in results:
            meta = r["metadata"]
            distance = r.get("distance", "N/A")
            lines.append(
                f"• {meta['project']} lot {meta['lot']} vozilo {meta['vehicle_number']}: "
                f"{meta['completion_percent']}% (sličnost: {1 - distance:.2f})"
            )
        return "\n".join(lines)
    
    return "Došlo je do neočekivane greške pri generisanju odgovora."


def generate_hr_answer(retrieval_result: RetrievalResult) -> str:
    """
    Generiše odgovor na osnovu rezultata pretrage HR zapisa.
    Ne halucinira - koristi samo pronađene podatke.
    """
    status = retrieval_result.status
    results = retrieval_result.results
    entities = retrieval_result.entities
    
    if status == "no_result":
        parts = []
        if entities.get("employee_name"):
            parts.append(f"zaposlenog {entities['employee_name']}")
        if entities.get("department_query"):
            parts.append(f"sektora {entities['department_query']}")
        
        if parts:
            return f"Nemam HR podataka za {', '.join(parts)}."
        return "Nisam pronašao relevantne HR podatke u bazi."
    
    if status == "exact_employee_match":
        r = results[0]
        meta = r["metadata"]
        
        parts = [f"{meta.get('employee_name', '')}"]
        
        if meta.get("position"):
            parts.append(f" radi kao {meta['position']}")
        
        if meta.get("department"):
            parts.append(f" u sektoru {meta['department']}")
        
        parts.append(".")
        
        if meta.get("employment_status"):
            parts.append(f" Status ugovora: {meta['employment_status']}.")
        
        if meta.get("contract_end_date"):
            parts.append(f" Ugovor ističe: {meta['contract_end_date']}.")
        
        if meta.get("termination_basis"):
            parts.append(f" Napomena: {meta['termination_basis']}.")
        
        if meta.get("next_action"):
            parts.append(f" Sledeća aktivnost: {meta['next_action']}.")
        
        return "".join(parts)
    
    if status == "status_filter_match":
        status_name = entities.get("status_query", "")
        status_display = "određeno" if status_name == "odredjeno" else "neodređeno" if status_name == "neodredjeno" else status_name
        
        names = [r["metadata"].get("employee_name", "") for r in results]
        
        if len(names) == 1:
            return f"Na {status_display} je: {names[0]}."
        
        return f"Na {status_display} su: {', '.join(names)}."
    
    if status == "contract_expiring_filter":
        lines = ["Zaposleni kojima ističe ugovor:"]
        for r in results[:5]:
            meta = r["metadata"]
            name = meta.get("employee_name", "")
            end_date = meta.get("contract_end_date", "")
            lines.append(f"• {name} - ugovor ističe {end_date}")
        
        if len(results) > 5:
            lines.append(f"... i još {len(results) - 5} zaposlenih")
        
        return "\n".join(lines)
    
    if status == "termination_basis_filter":
        lines = ["Zaposleni sa napomenom o prestanku/disciplinskom merom:"]
        for r in results:
            meta = r["metadata"]
            name = meta.get("employee_name", "")
            basis = meta.get("termination_basis", "")
            lines.append(f"• {name}: {basis}")
        
        return "\n".join(lines)
    
    if status == "position_filter_match":
        r = results[0]
        meta = r["metadata"]
        name = meta.get("employee_name", "")
        position = meta.get("position", "")
        dept = meta.get("department", "")
        
        parts = [f"{name} radi kao {position}"]
        if dept:
            parts.append(f" u sektoru {dept}")
        parts.append(".")
        
        if meta.get("employment_status"):
            parts.append(f" Status ugovora: {meta['employment_status']}.")
        
        if meta.get("contract_end_date"):
            parts.append(f" Ugovor ističe: {meta['contract_end_date']}.")
        
        return "".join(parts)
    
    if status == "department_filter_match":
        dept = results[0]["metadata"].get("department", "") if results else ""
        
        lines = [f"Zaposleni u sektoru {dept}:"]
        for r in results:
            meta = r["metadata"]
            name = meta.get("employee_name", "")
            pos = meta.get("position", "")
            end_date = meta.get("contract_end_date", "")
            if end_date and end_date != "Нема истека":
                lines.append(f"• {name} - {pos}, ugovor ističe: {end_date}")
            else:
                lines.append(f"• {name} - {pos}, ugovor na neodređeno")
        
        return "\n".join(lines)
    
    if status == "vector_fallback":
        lines = ["Na osnovu pretrage HR baze, pronađeni su sledeći rezultati:"]
        for r in results:
            meta = r["metadata"]
            name = meta.get("employee_name", "")
            position = meta.get("position", "")
            dept = meta.get("department", "")
            distance = r.get("distance", "N/A")
            lines.append(f"• {name} - {position}, {dept} (sličnost: {1 - distance:.2f})")
        
        return "\n".join(lines)
    
    return "Došlo je do neočekivane greške pri generisanju HR odgovora."


def route_and_answer(question: str) -> tuple[str, dict]:
    """
    Centralni router koji određuje intent i poziva odgovarajući retriever.
    
    Vraća:
    - odgovor: string sa odgovorom
    - debug: dict sa debug informacijama
    """
    intent_result = detect_intent(question)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    
    debug = {
        "original_question": question,
        "intent": intent,
        "intent_confidence": confidence,
        "project_score": intent_result.get("project_score", 0),
        "hr_score": intent_result.get("hr_score", 0),
    }
    
    if intent == PROJECT_INTENT:
        result = retrieve_project_status(question)
        answer = generate_project_status_response(result)
        
        debug.update({
            "rewritten_query": result.rewritten_query,
            "retrieval_strategy": result.status,
            "collection_used": result.collection_used,
            "result_count": len(result.results),
            "entities": result.entities,
        })
        
        return answer, debug
    
    if intent == HR_INTENT:
        result = retrieve_hr_record(question)
        answer = generate_hr_answer(result)
        
        debug.update({
            "rewritten_query": result.rewritten_query,
            "retrieval_strategy": result.status,
            "collection_used": result.collection_used,
            "result_count": len(result.results),
            "entities": result.entities,
        })
        
        return answer, debug
    
    if intent == MIXED_INTENT:
        answer = (
            "Pitanje meša projektne i HR podatke. "
            "Molim vas da precizirate da li pitanje o statusu projekta/vozila "
            "ili o ljudskim resursima (zaposlenima, ugovorima, sektorima)."
        )
        debug["retrieval_strategy"] = "mixed_intent_clarification"
        return answer, debug
    
    project_result = retrieve_project_status(question)
    hr_result = retrieve_hr_record(question)
    
    project_has_results = project_result.status != "no_result"
    hr_has_results = hr_result.status != "no_result"
    
    if project_has_results and not hr_has_results:
        answer = generate_project_status_response(project_result)
        debug.update({
            "rewritten_query": project_result.rewritten_query,
            "retrieval_strategy": f"fallback_project_{project_result.status}",
            "collection_used": project_result.collection_used,
            "result_count": len(project_result.results),
            "entities": project_result.entities,
        })
        return answer, debug
    
    if hr_has_results and not project_has_results:
        answer = generate_hr_answer(hr_result)
        debug.update({
            "rewritten_query": hr_result.rewritten_query,
            "retrieval_strategy": f"fallback_hr_{hr_result.status}",
            "collection_used": hr_result.collection_used,
            "result_count": len(hr_result.results),
            "entities": hr_result.entities,
        })
        return answer, debug
    
    if project_has_results and hr_has_results:
        answer = (
            "Pronađeni su rezultati i u projektnoj i u HR bazi. "
            "Molim vas da precizirate pitanje."
        )
        debug["retrieval_strategy"] = "ambiguous_both_results"
        return answer, debug
    
    answer = (
        "Nisam pronašao relevantne podatke ni u projektnoj ni u HR bazi. "
        "Pokušajte da precizirate pitanje ili proverite da li podaci postoje u sistemu."
    )
    debug["retrieval_strategy"] = "no_results_anywhere"
    return answer, debug


def transcribe_audio(audio_path: str) -> str:
    """Transkribuje audio fajl preko OpenAI STT API-ja. Zatim prikazuje tekst na cirilici"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=audio_file,
            language="sr",
            response_format="text",
        )
    return transcript.strip() if isinstance(transcript, str) else transcript.text.strip()




def format_debug_output(debug: dict, transcribed: str = None) -> list[str]:
    """Formatira debug informacije za prikaz u UI."""
    parts = ["---"]
    
    if transcribed:
        parts.append(f"🎤 **Transkribovano:** {transcribed}")
    
    intent_emoji = {
        PROJECT_INTENT: "🚗",
        HR_INTENT: "👥",
        MIXED_INTENT: "🔀",
        UNKNOWN_INTENT: "❓",
    }
    intent = debug.get("intent", UNKNOWN_INTENT)
    parts.append(f"{intent_emoji.get(intent, '❓')} **Intent:** {intent} (pouzdanost: {debug.get('intent_confidence', 'N/A')})")
    
    entities = debug.get("entities", {})
    entity_parts = []
    
    if entities.get("project"):
        entity_parts.append(f"projekat={entities['project']}")
    if entities.get("lot"):
        entity_parts.append(f"lot={entities['lot']}")
    if entities.get("vehicle_number"):
        entity_parts.append(f"vozilo={entities['vehicle_number']}")
    if entities.get("employee_name"):
        entity_parts.append(f"zaposleni={entities['employee_name']}")
    if entities.get("department_query"):
        entity_parts.append(f"sektor={entities['department_query']}")
    if entities.get("status_query"):
        entity_parts.append(f"status={entities['status_query']}")
    if entities.get("contract_query"):
        entity_parts.append(f"ugovor={entities['contract_query']}")
    if entities.get("termination_query"):
        entity_parts.append(f"prestanak={entities['termination_query']}")
    
    if entity_parts:
        parts.append(f"🔍 **Entiteti:** {', '.join(entity_parts)}")
    else:
        parts.append("🔍 **Entiteti:** (nisu pronađeni)")
    
    status_map = {
        "exact_match": "✅ Tačan pogodak (projekat)",
        "vehicle_found_in_different_lot": "⚠️ Vozilo u drugom lotu",
        "project_only": "📋 Samo projekat",
        "exact_employee_match": "✅ Tačan pogodak (zaposleni)",
        "status_filter_match": "👥 Filter po statusu",
        "contract_expiring_filter": "📅 Filter po isteku ugovora",
        "termination_basis_filter": "⚠️ Filter po osnovu prestanka",
        "department_filter_match": "🏢 Filter po sektoru",
        "vector_fallback": "🔎 Semantička pretraga",
        "no_result": "❌ Nema rezultata",
        "mixed_intent_clarification": "🔀 Mešovit intent",
        "ambiguous_both_results": "❓ Rezultati u obe baze",
        "no_results_anywhere": "❌ Nema rezultata nigde",
    }
    strategy = debug.get("retrieval_strategy", "")
    parts.append(f"📊 **Strategija:** {status_map.get(strategy, strategy)}")
    
    if debug.get("collection_used"):
        parts.append(f"📁 **Kolekcija:** {debug['collection_used']}")
    
    if debug.get("result_count") is not None:
        parts.append(f"📈 **Broj rezultata:** {debug['result_count']}")
    
    if debug.get("rewritten_query"):
        parts.append(f"✏️ **Rewritten query:** {debug['rewritten_query'][:100]}...")
    
    return parts


def answer_question(
    audio_path: str | None,
    history: list[dict],
    show_debug: bool = True,
) -> tuple[list[dict], None]:
    """
    Obrađuje audio pitanje i vraća odgovor o projektnim statusima ili HR podacima.
    """
    try:
        if not audio_path:
            return history, None
        
        try:
            transcribed = transcribe_audio(audio_path)
            question = transcribed
        except Exception as e:
            error_msg = f"Greška pri transkripciji: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            return history, None
        
        if not question:
            history.append({"role": "assistant", "content": "Nisam uspeo da transkribujem audio. Pokušajte ponovo."})
            return history, None
        
        history.append({"role": "user", "content": lat_to_cyr(question)})
        
        try:
            response_text, debug = route_and_answer(question)
        except Exception as e:
            error_msg = f"Greška pri pretrazi: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            return history, None
        
        response_parts = [f"💬 {lat_to_cyr(response_text)}"]
        
        if show_debug:
            response_parts.extend(format_debug_output(debug, transcribed))
        
        response = "\n\n".join(response_parts)
        history.append({"role": "assistant", "content": response})
        
        return history, None
    
    except Exception as e:
        error_msg = f"Neočekivana greška: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        return history, None


def answer_text_question(
    text_input: str,
    history: list[dict],
    show_debug: bool = True,
) -> tuple[list[dict], str]:
    """
    Obrađuje tekstualno pitanje (za debug/testiranje).
    Podržava i projektna i HR pitanja.
    """
    try:
        if not text_input or not text_input.strip():
            return history, ""
        
        question = text_input.strip()
        question_lat = cyr_to_lat(question)
        
        history.append({"role": "user", "content": question})
        
        try:
            response_text, debug = route_and_answer(question_lat)
        except Exception as e:
            error_msg = f"Greška pri pretrazi: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        response_parts = [f"💬 {response_text}"]
        
        if show_debug:
            response_parts.extend(format_debug_output(debug))
        
        response = "\n\n".join(response_parts)
        history.append({"role": "assistant", "content": response})
        
        return history, ""
    
    except Exception as e:
        error_msg = f"Neočekivana greška: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def run_test_queries():
    """Pokreće test pitanja za verifikaciju oba domena."""
    
    project_test_questions = [
        "pasars lot 2 vozilo 5",
        "ПАСАРС лот 2 возило 5",
        "status pasars vozilo broj 5",
        "pasars lot 2 vozilo 2",
        "koliko je završeno vozilo 2 u pasarsu",
        "koja vozila postoje za pasars",
        "metro lot 1 vozilo 1",
    ]
    
    hr_test_questions = [
        "kada ističe ugovor Lazi Lazareviću",
        "ko je na određeno",
        "ko ima opomenu pred otkaz",
        "ko radi u službi održavanja",
        "šta je sledeća aktivnost za Lazu Lazarevića",
        "ko je sistem administrator",
        "ko radi u IT sektoru",
    ]
    
    print("\n" + "=" * 60)
    print("TEST PITANJA - PROJEKTI")
    print("=" * 60)
    
    for q in project_test_questions:
        print(f"\n📝 Pitanje: {q}")
        response, debug = route_and_answer(q)
        print(f"   Intent: {debug['intent']} ({debug['intent_confidence']})")
        print(f"   Strategija: {debug.get('retrieval_strategy', 'N/A')}")
        print(f"   Entiteti: {debug.get('entities', {})}")
        print(f"   Odgovor: {response[:200]}...")
    
    print("\n" + "=" * 60)
    print("TEST PITANJA - HR")
    print("=" * 60)
    
    for q in hr_test_questions:
        print(f"\n📝 Pitanje: {q}")
        response, debug = route_and_answer(q)
        print(f"   Intent: {debug['intent']} ({debug['intent_confidence']})")
        print(f"   Strategija: {debug.get('retrieval_strategy', 'N/A')}")
        print(f"   Entiteti: {debug.get('entities', {})}")
        print(f"   Odgovor: {response[:200]}...")
    
    print("\n" + "=" * 60)


# =============================================================================
# BENCHMARK TEST - PROJEKTI
# Test pitanja su u benchmark_tests.py (importovano na vrhu fajla)
# =============================================================================


def evaluate_project_answer(response: str, debug: dict, expected: dict) -> dict:
    """
    Evaluira odgovor sistema u odnosu na očekivane vrednosti.
    
    Vraća dict sa rezultatima provere za svaku metriku.
    """
    results = {
        "intent_correct": False,
        "project_correct": False,
        "lot_correct": False,
        "vehicle_correct": False,
        "percent_correct": False,
        "percent_mentioned": False,
    }
    
    # 1. Provera intenta
    actual_intent = debug.get("intent", "")
    results["intent_correct"] = actual_intent == expected.get("intent", "")
    
    # 2. Provera entiteta
    entities = debug.get("entities", {})
    
    # Projekat
    actual_project = entities.get("project_normalized") or entities.get("project", "")
    expected_project = expected.get("project", "")
    results["project_correct"] = actual_project.upper() == expected_project.upper()
    
    # Lot
    actual_lot = entities.get("lot")
    expected_lot = expected.get("lot")
    if actual_lot is not None and expected_lot is not None:
        results["lot_correct"] = int(actual_lot) == int(expected_lot)
    
    # Vozilo
    actual_vehicle = entities.get("vehicle_number")
    expected_vehicle = expected.get("vehicle")
    if actual_vehicle is not None and expected_vehicle is not None:
        results["vehicle_correct"] = int(actual_vehicle) == int(expected_vehicle)
    
    # 3. Provera procenta u odgovoru
    expected_percent = expected.get("completion_percent")
    if expected_percent is not None:
        percent_str = str(expected_percent)
        # Tražimo procenat u odgovoru (npr. "25%", "25 %", "25 procenata")
        results["percent_mentioned"] = (
            f"{percent_str}%" in response or
            f"{percent_str} %" in response or
            f"{percent_str} procen" in response.lower() or
            f"na {percent_str}" in response.lower() or
            f"je {percent_str}" in response.lower()
        )
        # Strožija provera - da li je tačan procenat
        results["percent_correct"] = results["percent_mentioned"]
    
    return results


def run_project_benchmark() -> dict:
    """
    Pokreće benchmark test za projektna pitanja.
    
    Vraća summary dict sa svim metrikama.
    """
    print("\n" + "=" * 70)
    print("🧪 BENCHMARK TEST - PROJEKTI")
    print("=" * 70)
    print(f"Broj test pitanja: {len(PROJECT_BENCHMARK_TESTS)}")
    print("-" * 70)
    
    all_results = []
    
    for test in PROJECT_BENCHMARK_TESTS:
        test_id = test["id"]
        question = test["question"]
        expected = test["expected"]
        description = test["description"]
        
        print(f"\n[{test_id}] {description}")
        print(f"    Pitanje: \"{question}\"")
        
        # Pozovi sistem
        response, debug = route_and_answer(question)
        
        # Evaluiraj
        eval_results = evaluate_project_answer(response, debug, expected)
        
        # Izračunaj score za ovaj test (0-100%)
        checks = [
            eval_results["intent_correct"],
            eval_results["project_correct"],
            eval_results["lot_correct"],
            eval_results["vehicle_correct"],
            eval_results["percent_mentioned"],
        ]
        test_score = sum(checks) / len(checks) * 100
        
        # Prikaži rezultate
        status_icons = {True: "✅", False: "❌"}
        print(f"    Intent:   {status_icons[eval_results['intent_correct']]} (očekivan: {expected.get('intent')}, dobijen: {debug.get('intent')})")
        print(f"    Projekat: {status_icons[eval_results['project_correct']]} (očekivan: {expected.get('project')}, dobijen: {debug.get('entities', {}).get('project_normalized', 'N/A')})")
        print(f"    Lot:      {status_icons[eval_results['lot_correct']]} (očekivan: {expected.get('lot')}, dobijen: {debug.get('entities', {}).get('lot', 'N/A')})")
        print(f"    Vozilo:   {status_icons[eval_results['vehicle_correct']]} (očekivan: {expected.get('vehicle')}, dobijen: {debug.get('entities', {}).get('vehicle_number', 'N/A')})")
        print(f"    Procenat: {status_icons[eval_results['percent_mentioned']]} (očekivan: {expected.get('completion_percent')}% u odgovoru)")
        print(f"    📊 Score: {test_score:.0f}%")
        print(f"    Odgovor: {response[:150]}...")
        
        all_results.append({
            "test_id": test_id,
            "question": question,
            "expected": expected,
            "eval_results": eval_results,
            "test_score": test_score,
            "response": response,
            "debug": debug,
        })
    
    # Agregatne metrike
    print("\n" + "=" * 70)
    print("📈 AGREGATNE METRIKE")
    print("=" * 70)
    
    total_tests = len(all_results)
    
    intent_accuracy = sum(1 for r in all_results if r["eval_results"]["intent_correct"]) / total_tests * 100
    project_accuracy = sum(1 for r in all_results if r["eval_results"]["project_correct"]) / total_tests * 100
    lot_accuracy = sum(1 for r in all_results if r["eval_results"]["lot_correct"]) / total_tests * 100
    vehicle_accuracy = sum(1 for r in all_results if r["eval_results"]["vehicle_correct"]) / total_tests * 100
    percent_accuracy = sum(1 for r in all_results if r["eval_results"]["percent_mentioned"]) / total_tests * 100
    
    avg_score = sum(r["test_score"] for r in all_results) / total_tests
    perfect_tests = sum(1 for r in all_results if r["test_score"] == 100)
    failed_tests = sum(1 for r in all_results if r["test_score"] < 60)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  METRIKA                          │  REZULTAT                       │
├───────────────────────────────────┼─────────────────────────────────┤
│  Intent tačnost                   │  {intent_accuracy:5.1f}%  ({int(intent_accuracy/100*total_tests)}/{total_tests})                   │
│  Projekat tačnost                 │  {project_accuracy:5.1f}%  ({int(project_accuracy/100*total_tests)}/{total_tests})                   │
│  Lot tačnost                      │  {lot_accuracy:5.1f}%  ({int(lot_accuracy/100*total_tests)}/{total_tests})                   │
│  Vozilo tačnost                   │  {vehicle_accuracy:5.1f}%  ({int(vehicle_accuracy/100*total_tests)}/{total_tests})                   │
│  Procenat u odgovoru              │  {percent_accuracy:5.1f}%  ({int(percent_accuracy/100*total_tests)}/{total_tests})                   │
├───────────────────────────────────┼─────────────────────────────────┤
│  ⭐ PROSEČAN SCORE                │  {avg_score:5.1f}%                           │
│  ✅ Perfektni testovi (100%)      │  {perfect_tests}/{total_tests}                              │
│  ❌ Neuspeli testovi (<60%)       │  {failed_tests}/{total_tests}                              │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Lista neuspelih testova
    if failed_tests > 0:
        print("\n⚠️  NEUSPELI TESTOVI (score < 60%):")
        for r in all_results:
            if r["test_score"] < 60:
                print(f"    - [{r['test_id']}] {r['question']} (score: {r['test_score']:.0f}%)")
    
    print("\n" + "=" * 70)
    
    return {
        "total_tests": total_tests,
        "intent_accuracy": intent_accuracy,
        "project_accuracy": project_accuracy,
        "lot_accuracy": lot_accuracy,
        "vehicle_accuracy": vehicle_accuracy,
        "percent_accuracy": percent_accuracy,
        "avg_score": avg_score,
        "perfect_tests": perfect_tests,
        "failed_tests": failed_tests,
        "all_results": all_results,
    }


# =============================================================================
# BENCHMARK TEST - HR
# Test pitanja su u benchmark_tests.py (importovano na vrhu fajla)
# =============================================================================


def evaluate_hr_answer(response: str, debug: dict, expected: dict) -> dict:
    """
    Evaluira HR odgovor sistema u odnosu na očekivane vrednosti.
    
    Vraća dict sa rezultatima provere za svaku metriku.
    """
    results = {
        "intent_correct": False,
        "employee_found": False,
        "department_mentioned": False,
        "status_mentioned": False,
        "date_mentioned": False,
        "position_mentioned": False,
    }
    
    response_lower = response.lower()
    response_normalized = normalize_text(response)
    
    # 1. Provera intenta
    actual_intent = debug.get("intent", "")
    results["intent_correct"] = actual_intent == expected.get("intent", "")
    
    # 2. Provera entiteta - zaposleni
    entities = debug.get("entities", {})
    expected_employee = expected.get("employee_name", "")
    
    if expected_employee:
        actual_employee = entities.get("employee_name_normalized") or ""
        expected_norm = normalize_text(expected_employee)
        results["employee_found"] = (
            expected_norm in actual_employee.lower() or
            expected_norm in response_normalized
        )
    
    expected_employees = expected.get("expected_employees", [])
    if expected_employees:
        found_count = sum(1 for emp in expected_employees if normalize_text(emp) in response_normalized)
        results["employee_found"] = found_count >= 1
    
    # 3. Provera sektora/departmana u odgovoru
    expected_dept = expected.get("department", "")
    if expected_dept:
        expected_dept_norm = normalize_text(expected_dept)
        results["department_mentioned"] = (
            expected_dept_norm in response_normalized or
            expected_dept in response
        )
    
    # 4. Provera statusa zaposlenja
    expected_status = expected.get("employment_status", "")
    if expected_status:
        results["status_mentioned"] = (
            expected_status.lower() in response_lower or
            expected_status in response or
            "одређено" in response_lower or
            "неодређено" in response_lower or
            "određeno" in response_lower or
            "neodređeno" in response_lower
        )
    
    # 5. Provera datuma (početak ili kraj ugovora)
    expected_end_date = expected.get("contract_end_date", "")
    expected_start_date = expected.get("contract_start_date", "")
    
    if expected_end_date and expected_end_date != "Нема истека":
        results["date_mentioned"] = expected_end_date in response
    elif expected_start_date:
        results["date_mentioned"] = expected_start_date in response
    else:
        results["date_mentioned"] = True  # Nije očekivan datum
    
    # 6. Provera pozicije
    expected_position = expected.get("position", "")
    if expected_position:
        results["position_mentioned"] = (
            expected_position.lower() in response_lower or
            expected_position in response
        )
    else:
        results["position_mentioned"] = True  # Nije očekivana pozicija
    
    # 7. Provera upozorenja/opomene
    if expected.get("has_warning"):
        warning_keywords = expected.get("warning_keywords", [])
        results["warning_found"] = any(kw.lower() in response_lower for kw in warning_keywords)
    
    return results


def run_hr_benchmark() -> dict:
    """
    Pokreće benchmark test za HR pitanja.
    
    Vraća summary dict sa svim metrikama.
    """
    print("\n" + "=" * 70)
    print("🧪 BENCHMARK TEST - HR (Ljudski resursi)")
    print("=" * 70)
    print(f"Broj test pitanja: {len(HR_BENCHMARK_TESTS)}")
    print("-" * 70)
    
    all_results = []
    
    for test in HR_BENCHMARK_TESTS:
        test_id = test["id"]
        question = test["question"]
        expected = test["expected"]
        description = test["description"]
        
        print(f"\n[{test_id}] {description}")
        print(f"    Pitanje: \"{question}\"")
        
        # Pozovi sistem
        response, debug = route_and_answer(question)
        
        # Evaluiraj
        eval_results = evaluate_hr_answer(response, debug, expected)
        
        # Izračunaj score za ovaj test
        checks = [
            eval_results["intent_correct"],
            eval_results["employee_found"],
            eval_results["department_mentioned"],
            eval_results["status_mentioned"],
            eval_results["date_mentioned"],
        ]
        test_score = sum(checks) / len(checks) * 100
        
        # Prikaži rezultate
        status_icons = {True: "✅", False: "❌"}
        print(f"    Intent:     {status_icons[eval_results['intent_correct']]} (očekivan: {expected.get('intent')}, dobijen: {debug.get('intent')})")
        print(f"    Zaposleni:  {status_icons[eval_results['employee_found']]} (očekivan: {expected.get('employee_name', expected.get('expected_employees', 'N/A'))})")
        print(f"    Sektor:     {status_icons[eval_results['department_mentioned']]} (očekivan: {expected.get('department', 'N/A')})")
        print(f"    Status:     {status_icons[eval_results['status_mentioned']]} (očekivan: {expected.get('employment_status', 'N/A')})")
        print(f"    Datum:      {status_icons[eval_results['date_mentioned']]} (očekivan: {expected.get('contract_end_date', expected.get('contract_start_date', 'N/A'))})")
        print(f"    📊 Score: {test_score:.0f}%")
        print(f"    Odgovor: {response[:150]}...")
        
        all_results.append({
            "test_id": test_id,
            "question": question,
            "expected": expected,
            "eval_results": eval_results,
            "test_score": test_score,
            "response": response,
            "debug": debug,
        })
    
    # Agregatne metrike
    print("\n" + "=" * 70)
    print("📈 AGREGATNE METRIKE - HR")
    print("=" * 70)
    
    total_tests = len(all_results)
    
    intent_accuracy = sum(1 for r in all_results if r["eval_results"]["intent_correct"]) / total_tests * 100
    employee_accuracy = sum(1 for r in all_results if r["eval_results"]["employee_found"]) / total_tests * 100
    department_accuracy = sum(1 for r in all_results if r["eval_results"]["department_mentioned"]) / total_tests * 100
    status_accuracy = sum(1 for r in all_results if r["eval_results"]["status_mentioned"]) / total_tests * 100
    date_accuracy = sum(1 for r in all_results if r["eval_results"]["date_mentioned"]) / total_tests * 100
    
    avg_score = sum(r["test_score"] for r in all_results) / total_tests
    perfect_tests = sum(1 for r in all_results if r["test_score"] == 100)
    failed_tests = sum(1 for r in all_results if r["test_score"] < 60)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  METRIKA                          │  REZULTAT                       │
├───────────────────────────────────┼─────────────────────────────────┤
│  Intent tačnost                   │  {intent_accuracy:5.1f}%  ({int(intent_accuracy/100*total_tests)}/{total_tests})                   │
│  Zaposleni pronađen               │  {employee_accuracy:5.1f}%  ({int(employee_accuracy/100*total_tests)}/{total_tests})                   │
│  Sektor u odgovoru                │  {department_accuracy:5.1f}%  ({int(department_accuracy/100*total_tests)}/{total_tests})                   │
│  Status zaposlenja                │  {status_accuracy:5.1f}%  ({int(status_accuracy/100*total_tests)}/{total_tests})                   │
│  Datum u odgovoru                 │  {date_accuracy:5.1f}%  ({int(date_accuracy/100*total_tests)}/{total_tests})                   │
├───────────────────────────────────┼─────────────────────────────────┤
│  ⭐ PROSEČAN SCORE                │  {avg_score:5.1f}%                           │
│  ✅ Perfektni testovi (100%)      │  {perfect_tests}/{total_tests}                              │
│  ❌ Neuspeli testovi (<60%)       │  {failed_tests}/{total_tests}                              │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Lista neuspelih testova
    if failed_tests > 0:
        print("\n⚠️  NEUSPELI TESTOVI (score < 60%):")
        for r in all_results:
            if r["test_score"] < 60:
                print(f"    - [{r['test_id']}] {r['question']} (score: {r['test_score']:.0f}%)")
    
    print("\n" + "=" * 70)
    
    return {
        "total_tests": total_tests,
        "intent_accuracy": intent_accuracy,
        "employee_accuracy": employee_accuracy,
        "department_accuracy": department_accuracy,
        "status_accuracy": status_accuracy,
        "date_accuracy": date_accuracy,
        "avg_score": avg_score,
        "perfect_tests": perfect_tests,
        "failed_tests": failed_tests,
        "all_results": all_results,
    }


CUSTOM_CSS = """
.audio-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}
.recording-info {
    text-align: center;
    color: #666;
    font-size: 14px;
    margin-top: 10px;
}
body.push-to-talk-recording .recording-info {
    color: #c62828;
    font-weight: 600;
}
body.push-to-talk-recording #voice-input {
    outline: 2px solid #c62828;
    outline-offset: 4px;
    border-radius: 8px;
}
.domain-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin-right: 5px;
}
.project-badge {
    background-color: #e3f2fd;
    color: #1565c0;
}
.hr-badge {
    background-color: #f3e5f5;
    color: #7b1fa2;
}
"""

PUSH_TO_TALK_JS = """
(() => {
    const COMBO_KEYS = new Set(["y", "t"]);
    const pressedKeys = new Set();
    let shortcutActive = false;

    const getAudioRoot = () => document.getElementById("voice-input");

    const isEditableTarget = () => {
        const el = document.activeElement;
        if (!el) return false;
        const tag = el.tagName;
        return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
    };

    const getRecordButton = () => {
        const root = getAudioRoot();
        if (!root) return null;
        return root.querySelector("button.record-button");
    };

    const getStopButton = () => {
        const root = getAudioRoot();
        if (!root) return null;
        return root.querySelector("button.stop-button, button.stop-button-paused");
    };

    const isRecordingVisible = () => {
        const stopBtn = getStopButton();
        return !!(stopBtn && stopBtn.offsetParent !== null);
    };

    const setRecordingVisual = (active) => {
        document.body.classList.toggle("push-to-talk-recording", active);
    };

    const startRecording = () => {
        if (shortcutActive || isRecordingVisible()) return;
        const btn = getRecordButton();
        if (!btn || btn.offsetParent === null) return;
        btn.click();
        shortcutActive = true;
        requestAnimationFrame(() => {
            if (isRecordingVisible()) {
                setRecordingVisual(true);
            } else {
                shortcutActive = false;
            }
        });
    };

    const stopRecording = () => {
        if (!shortcutActive) return;
        const stopBtn = getStopButton();
        if (stopBtn && stopBtn.offsetParent !== null) {
            stopBtn.click();
        }
        shortcutActive = false;
        pressedKeys.clear();
        setRecordingVisual(false);
    };

    const comboHeld = () => pressedKeys.has("y") && pressedKeys.has("t");

    document.addEventListener("keydown", (event) => {
        const key = event.key.toLowerCase();
        if (!COMBO_KEYS.has(key) || event.repeat) return;
        if (isEditableTarget() && !shortcutActive) return;

        pressedKeys.add(key);
        if (comboHeld()) {
            event.preventDefault();
            startRecording();
        }
    }, true);

    document.addEventListener("keyup", (event) => {
        const key = event.key.toLowerCase();
        if (!COMBO_KEYS.has(key)) return;

        pressedKeys.delete(key);
        if (shortcutActive) {
            event.preventDefault();
            stopRecording();
        }
    }, true);

    window.addEventListener("blur", () => {
        if (!shortcutActive) return;
        stopRecording();
    });
})();
"""


def create_ui() -> gr.Blocks:
    """Kreira Gradio UI sa podrškom za projektne i HR upite."""
    with gr.Blocks(title="Статус пројеката и HR - STT + RAG") as demo:
        gr.Markdown("# TeрвoБот")
        
        
        show_debug = gr.Checkbox(label="Прикажи debug информације", value=True)
        
        chatbot = gr.Chatbot(
            label="Разговор",
            height=450,
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="📝 Текстуално питање",
                    placeholder="нпр. 'ПАСАРС лот 2 возило 5' или 'ко је на одређено'",
                )
            with gr.Column(scale=1):
                send_btn = gr.Button("Пошаљи", variant="primary")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Кликните за снимање",
                    elem_id="voice-input",
                    elem_classes=["audio-container"],
                )
                gr.Markdown(
                    "<p class='recording-info'>🔴 Држите <kbd>Y</kbd> + <kbd>T</kbd> за снимање → "
                    "пустите да пошаљете питање (или кликните микрофон)</p>",
                    elem_classes=["recording-info"],
                )
        
        audio_input.stop_recording(
            fn=answer_question,
            inputs=[audio_input, chatbot, show_debug],
            outputs=[chatbot, audio_input],
        )
        
        send_btn.click(
            fn=answer_text_question,
            inputs=[text_input, chatbot, show_debug],
            outputs=[chatbot, text_input],
        )
        
        text_input.submit(
            fn=answer_text_question,
            inputs=[text_input, chatbot, show_debug],
            outputs=[chatbot, text_input],
        )
        
        gr.Markdown("---")
        
       
    
    return demo


def main():
    global client, chroma_client, project_collection, hr_collection
    
    print("=" * 60, flush=True)
    print("Status projekata i HR - STT + RAG", flush=True)
    print("=" * 60, flush=True)
    
    check_api_key()
    
    print(f"STT model: {OPENAI_STT_MODEL}", flush=True)
    print(f"Embedding model: {OPENAI_EMBEDDING_MODEL}", flush=True)
    print(f"Chat model: {OPENAI_CHAT_MODEL}", flush=True)
    print(f"ChromaDB folder: {CHROMA_DIR}", flush=True)
    print(f"Project JSONL: {PROJECT_STATUS_JSONL}", flush=True)
    print(f"HR JSONL: {HR_RECORDS_JSONL}", flush=True)
    print(flush=True)
    
    print("Inicijalizacija OpenAI klijenta...", flush=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print("Inicijalizacija ChromaDB...", flush=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    print("Brisanje starih kolekcija...", flush=True)
    delete_old_collections()
    
    print("Kreiranje project kolekcije...", flush=True)
    project_collection = chroma_client.get_or_create_collection(
        name=PROJECT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    print("Kreiranje HR kolekcije...", flush=True)
    hr_collection = chroma_client.get_or_create_collection(
        name=HR_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    project_count = project_collection.count()
    hr_count = hr_collection.count()
    
    print(f"Trenutni broj projektnih zapisa: {project_count}", flush=True)
    print(f"Trenutni broj HR zapisa: {hr_count}", flush=True)
    
    if project_count == 0:
        print("Project baza je prazna, učitavam projektne statuse...", flush=True)
        seed_project_database(reset=False)
    else:
        print(f"Project baza već sadrži {project_count} zapisa", flush=True)
    
    if hr_count == 0:
        print("HR baza je prazna, učitavam HR zapise...", flush=True)
        seed_hr_database(reset=False)
    else:
        print(f"HR baza već sadrži {hr_count} zapisa", flush=True)
        init_hr_employee_cache()
    
    if "--seed" in sys.argv:
        print("\nForsiran seed svih baza...", flush=True)
        seed_all_databases(reset=True)
        print("Seed završen.", flush=True)
        return
    
    if "--test" in sys.argv:
        run_test_queries()
        return
    
    if "--benchmark" in sys.argv or "--bench" in sys.argv:
        run_project_benchmark()
        return
    
    if "--benchmark-hr" in sys.argv or "--bench-hr" in sys.argv:
        run_hr_benchmark()
        return
    
    if "--benchmark-all" in sys.argv or "--bench-all" in sys.argv:
        print("\n🚀 Pokrećem sve benchmark testove...\n")
        project_results = run_project_benchmark()
        hr_results = run_hr_benchmark()
        
        print("\n" + "=" * 70)
        print("📊 UKUPNI REZULTATI")
        print("=" * 70)
        total_tests = project_results["total_tests"] + hr_results["total_tests"]
        total_avg = (project_results["avg_score"] + hr_results["avg_score"]) / 2
        print(f"   Projekti:  {project_results['avg_score']:.1f}% prosek ({project_results['perfect_tests']}/{project_results['total_tests']} perfektno)")
        print(f"   HR:        {hr_results['avg_score']:.1f}% prosek ({hr_results['perfect_tests']}/{hr_results['total_tests']} perfektno)")
        print(f"   ────────────────────────────")
        print(f"   UKUPNO:    {total_avg:.1f}% prosek ({total_tests} testova)")
        print("=" * 70)
        return
    
    print(flush=True)
    print(f"Pokrećem Gradio UI na http://{GRADIO_HOST}:{GRADIO_PORT}", flush=True)
    print("=" * 60, flush=True)
    
    demo = create_ui()
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        css=CUSTOM_CSS,
        js=PUSH_TO_TALK_JS,
    )


if __name__ == "__main__":
    main()
