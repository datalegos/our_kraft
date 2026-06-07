# Threat Intelligence Data Collector

Modular ingestion: **collect** → **normalize** → **deduplicate** → **JSON** (optional **MongoDB**).

## Layout (as specified)

```text
meru-data-poc/
├── collectors/
│   ├── otx.py
│   ├── urlhaus.py
│   ├── malwarebazaar.py
│   ├── threatfox.py
│   └── njccic.py
├── core/
│   ├── http_client.py   # retries, timeouts, Abuse.ch Auth-Key header helper
│   ├── config.py        # loads .env only
│   └── logger.py
├── models/
│   └── schemas.py       # unified NormalizedIOC shape
├── pipeline/
│   ├── normalize.py
│   └── deduplicate.py
├── main.py
├── .env                 # not committed
└── .env.example
```

## How to run (required way)

From the **`meru-data-poc`** directory (so imports resolve):

```bash
cd meru-data-poc
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env
# edit .env — set OTX_API_KEY and Abuse.ch Auth-Key values as needed

python main.py
python main.py --sources otx,urlhaus,malwarebazaar,threatfox
python main.py --sources njccic
```

Exit codes: `0` success, `1` one or more collectors failed, `2` missing `.env`.

## Authentication

| Source        | Mechanism |
|---------------|-----------|
| OTX           | `X-OTX-API-KEY` |
| URLhaus / MalwareBazaar / ThreatFox | `Auth-Key` **HTTP header** (optional; recommended for quotas) |
| NJCCIC        | none |

When an `Auth-Key` is set, it is sent in the **HTTP header** (per spec) **and** merged into the POST JSON body so Abuse.ch accepts the request on strict endpoints.

## Unified record schema

Each item in `records` is:

```json
{
  "source": "otx",
  "type": "ip",
  "value": "1.2.3.4",
  "malware_family": "",
  "threat_type": "",
  "tags": [],
  "confidence": "medium",
  "first_seen": "",
  "last_seen": "",
  "context": { "campaign": "", "actor": "", "description": "" }
}
```

**Confidence / reliability:** OTX & NJCCIC → `medium`; URLhaus, MalwareBazaar, ThreatFox → `high`.

## Output

- `output/ingestion_<UTC-timestamp>.json` with `stats` + `records`.
- If `MONGODB_URI` is set and `pymongo` is installed, one document is inserted into `ti_ingestion`.

## NJCCIC note

`cyber.nj.gov` often serves an **Incapsula** shell to `requests`. You may get one placeholder document and few IOCs until you use browser automation or an official feed.

## Optional enhancements (not implemented)

Async HTTP, cron wrappers, and enrichment hooks can be added on top of this structure.
