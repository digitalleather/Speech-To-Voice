# Srpski STT + lokalna vektorska pretraga

Python aplikacija za srpski speech-to-text sa lokalnom vektorskom pretragom osoba.

## Šta radi aplikacija?

Korisnik može kroz web UI da:

1. **Ukuca pitanje tekstualno**, ili
2. **Snimi pitanje preko mikrofona**

Aplikacija zatim:

1. Ako je pitanje glasovno, šalje audio na **OpenAI Speech-to-Text API**
2. Dobija transkribovan tekst na srpskom jeziku
3. Pravi **embedding pitanja** preko OpenAI Embeddings API-ja
4. Pretražuje lokalnu **ChromaDB** vektorsku bazu osoba
5. Pronalazi osobu po imenu/prezimenu iz pitanja
6. Vraća **datum rođenja** te osobe u chat odgovoru

## Arhitektura

```
Gradio UI
  ↓
tekst ili mikrofon
  ↓
OpenAI gpt-4o-transcribe za STT
  ↓
OpenAI text-embedding-3-large za embedding pitanja
  ↓
lokalni ChromaDB
  ↓
metadata osobe
  ↓
OpenAI gpt-4o-mini za generisanje odgovora
  ↓
chatbot odgovor + izvor u chatu
```

## Gde se šta izvršava?

| Komponenta | Lokacija |
|------------|----------|
| Audio transkripcija | OpenAI API (cloud) |
| Embedding pitanja | OpenAI API (cloud) |
| Generisanje odgovora | OpenAI API (cloud) |
| Vektorska baza | Lokalno (ChromaDB) |
| Demo podaci | Lokalno (`people.json`) |
| Web UI | Lokalno (Gradio) |
| API ključ | Lokalno (`.env`) |

## Instalacija

### 1. Sistemske zavisnosti

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv
```

### 2. Python okruženje

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Python paketi

```bash
pip install -r requirements.txt
```

### 4. Konfiguracija

```bash
cp .env.example .env
nano .env
```

U `.env` fajlu **obavezno** dodaj svoj OpenAI API ključ:

```env
OPENAI_API_KEY=sk-tvoj-api-kljuc-ovde
```

Ostale opcije su opcione i imaju podrazumevane vrednosti:

```env
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
PEOPLE_JSON=./people.json
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
```

## Pokretanje

```bash
python app.py
```

Aplikacija će se pokrenuti na `http://127.0.0.1:7860`

## Test pitanja

Probaj ova pitanja tekstualno ili glasovno:

- "Kada je rođen Petar Petrović?"
- "Kada je rođena Jelena Jovanović?"
- "Koji je datum rođenja osobe koja se preziva Nikolić?"
- "Nađi mi datum rođenja za Marka Markovića."
- "Ko je Ana?"

## Demo podaci

Aplikacija dolazi sa 4 demo osobe u `people.json`:

| Ime | Prezime | Datum rođenja |
|-----|---------|---------------|
| Marko | Marković | 12.05.1990 |
| Jelena | Jovanović | 03.11.1988 |
| Petar | Petrović | 21.07.1995 |
| Ana | Nikolić | 15.02.1992 |

## Konfiguracija modela

Možeš promeniti modele u `.env` fajlu:

### Speech-to-Text modeli

```env
OPENAI_STT_MODEL=gpt-4o-transcribe
# ili
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
```

### Embedding modeli

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
# ili
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Chat modeli

```env
OPENAI_CHAT_MODEL=gpt-4o-mini
# ili
OPENAI_CHAT_MODEL=gpt-4o
```

## Struktura projekta

```
.
├── app.py              # Glavna aplikacija
├── people.json         # Demo podaci osoba
├── requirements.txt    # Python zavisnosti
├── README.md           # Dokumentacija
├── .env.example        # Primer konfiguracije
├── .gitignore          # Git ignore pravila
└── chroma_db/          # ChromaDB baza (kreira se automatski)
```

## Bezbednost

- `.env` fajl **nikada** ne treba commitovati u git
- API ključ se čuva **isključivo lokalno**
- Audio fajlovi se ne čuvaju trajno
- ChromaDB baza je lokalna i privatna

## Zavisnosti

- `openai` - OpenAI Python SDK
- `gradio` - Web UI framework
- `chromadb` - Vektorska baza podataka
- `python-dotenv` - Učitavanje .env fajlova
- `numpy` - Numeričke operacije

## Napomene

- Aplikacija automatski konvertuje ćirilicu u latinicu ako STT vrati ćirilični tekst
- ChromaDB baza se kreira automatski pri prvom pokretanju
- Embeddinci se računaju jednom i čuvaju u bazi
- Pretraga koristi kosinusnu sličnost

## Licenca

MIT
# SpeechToTextyzt
