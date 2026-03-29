# Deterministic_RAG

A clean, self-contained legal AI demo for answerability-aware RAG.

## What the website does

The web app lets a user:
- understand the system quickly
- choose a sample contract or upload a PDF
- prepare the document first (with ETA + progress bar)
- ask contract questions
- see whether the question is answerable
- see the short answer, reason, and supporting clauses

The backend preserves the existing core pipeline behavior and only adds a thin API wrapper.

## Project layout

```text
/Users/jonathang/Projects/Deterministic_RAG
├── core_pipeline.py
├── topology_metrics.py
├── server.py
├── scripts/
│   └── evaluate_topology.py
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── sample_docs/
├── uploads/
├── topology_telemetry.jsonl   # created automatically after /api/ask calls
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd /Users/jonathang/Projects/Deterministic_RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY (or Azure settings) in .env
```

## Run the website (frontend + backend)

```bash
cd /Users/jonathang/Projects/Deterministic_RAG
source venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Test with a sample document

1. Put one PDF into:
```text
/Users/jonathang/Projects/Deterministic_RAG/sample_docs
```

2. Refresh the site.
3. Select the sample in the Demo section.
4. Click `Prepare document` and wait until status shows ready.
5. Try questions such as:
- What is the renewal term?
- Can either party terminate for breach?
- What law governs the agreement?
- What happens if one party materially breaches?
- What emails were exchanged between the parties?

## API contract

`POST /api/ask` accepts:
- `question` (required form field)
- `doc_token` (required form field from `/api/prepare-doc`)

Response shape:

```json
{
  "answerable": true,
  "short_answer": "...",
  "reason": "...",
  "supporting_clauses": [
    {"text": "...", "source": "Section ..."}
  ],
  "metadata": {
    "confidence": "high",
    "topology_score": 0.73,
    "topology_predicted": "answerable"
  },
  "document": "example.pdf"
}
```

## Notes

- This is a demo-oriented single-page site, not a production system.
- Upload processing can take time because the full extraction/retrieval pipeline runs.
- Pipelines are cached per document file path + modification time in `server.py` for faster repeat queries.

## Topology telemetry and evaluation

Every `/api/ask` call now writes a telemetry event to:

```text
/Users/jonathang/Projects/Deterministic_RAG/topology_telemetry.jsonl
```

Each event includes:
- topology score/prediction/threshold/confidence
- evidence and risk subscores
- key topology signals used by the scorer
- retrieval stats and latency
- final API answerability result

View recent telemetry events:

```bash
curl "http://127.0.0.1:8000/api/topology-telemetry?limit=50"
```

### Evaluate scorer on a labeled CSV

```bash
cd /Users/jonathang/Projects/Deterministic_RAG
source venv/bin/activate
python scripts/evaluate_topology.py \
  --csv /Users/jonathang/Downloads/rag_topology_1774275304935.csv \
  --report-json /Users/jonathang/Downloads/topology_eval_report.json \
  --errors-csv /Users/jonathang/Downloads/topology_eval_errors.csv
```

The evaluator reports:
- overall metrics (accuracy, precision, recall, F1, AUC, Brier)
- calibration bins
- by-slot-type performance
- signal quality diagnostics (separation and constant/non-numeric fields)
- ablation results (which signals matter most)
- detailed error rows
