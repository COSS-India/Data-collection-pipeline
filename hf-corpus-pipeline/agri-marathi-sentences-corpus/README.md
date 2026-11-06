# agri-marathi-sentences-corpus (Draft v0.1)

**322,222 natural Marathi sentences about agriculture**, one JSON object per line, gzip-compressed. Built by the pipeline in this folder's parent ([../README.md](../README.md)).

| File | Size | Rows |
|---|---|---|
| `agri_marathi_sentences.jsonl.gz` | 41.8 MB (285 MB unzipped) | 322,222 |

The file is gzipped because plain JSONL exceeds GitHub's 100 MB file limit. sha256: `7c2246b3efd3d82cb915616c10ec0ce62f6b5a1cc6f3e1006fdcc7c24e0e9f55`

## Load
```bash
zcat agri_marathi_sentences.jsonl.gz | head -3        # peek
gunzip -k agri_marathi_sentences.jsonl.gz              # plain .jsonl
```
```python
import pandas as pd
df = pd.read_json("agri_marathi_sentences.jsonl.gz", lines=True)   # or polars: pl.read_ndjson(...)
```

## One row
```json
{"sentence_id": "…", "text": "मोसमी पावसाच्या जोरदार आगमनामुळे राज्यात पेरणीच्या कामाला आता गती मिळणार आहे.",
 "label": "AGRI", "confidence": 1.0, "word_count": 11, "length_bucket": "medium",
 "source_dataset": "ai4bharat/sangraha", "source_license": "CC-BY-4.0",
 "doc_id": "…", "doc_p_agri": 1.0, "prompt_version": "v2", "license_note": null, …}
```
The full field list is in [../docs/DATASET_CARD.md](../docs/DATASET_CARD.md#fields).

## Contents
| Source | Rows | License |
|---|---|---|
| ai4bharat/sangraha (verified, web) | 261,687 | CC-BY-4.0 |
| shivam/marathi_samanantar_processed | 32,713 | Not stated: rows have `license_note` set |
| ai4bharat/Shrutilipi | 15,811 | CC-BY-4.0 |
| ai4bharat/IndicVoices | 9,269 | CC-BY-4.0 |
| ai4bharat/Kathbath | 2,337 | CC-BY-4.0 |
| ai4bharat/Rasa | 405 | CC-BY-4.0 |

- **MILU removed:** its 390 sentences are excluded because MILU is an evaluation benchmark.
- **Model labels:** the labels come from Gemma 4 26B A4B and have not been verified by a human yet. A spot-check found ~92–95% correct on Sangraha rows.
- **More detail:** quality, known issues and licensing are in [../docs/DATASET_CARD.md](../docs/DATASET_CARD.md).
