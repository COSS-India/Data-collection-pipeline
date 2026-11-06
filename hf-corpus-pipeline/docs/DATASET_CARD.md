---
pretty_name: Marathi Agriculture Sentences (Draft v0.1)
language:
- mr
license: other
license_name: mixed-upstream
license_link: LICENSE
task_categories:
- translation
- text-classification
size_categories:
- 100K<n<1M
tags:
- agriculture
- marathi
- indic
configs:
- config_name: default
  data_files:
  - split: train
    path: data/agri-*.parquet
---

# Marathi Agriculture Sentences (Draft v0.1)

**322,222 natural, human-written Marathi sentences about agriculture**, selected from open Marathi datasets and labelled by an LLM classifier. They are meant to be the source side of a Marathi↔English machine-translation corpus in the agriculture domain.

> **Draft status.** The labels come from a model (Gemma 4 26B A4B); no human has verified them yet. One subset (Samanantar, 32,713 rows) has an unclear upstream license (see [Licensing](#licensing)). Do not treat this draft as final.

## Quick start
```python
from datasets import load_dataset
ds = load_dataset("<hf-org>/<dataset-name>", split="train")   # name added at publication
high = ds.filter(lambda r: r["confidence"] >= 0.99)            # stricter subset
```
or with polars: `pl.read_parquet("data/agri-00000.parquet")`.

## Contents

| Source | Sentences | Text type | Label prompt |
|---|---|---|---|
| [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (verified, web) | 261,687 | Web news and articles | v2 |
| [shivam/marathi_samanantar_processed](https://huggingface.co/datasets/shivam/marathi_samanantar_processed) | 32,713 | Parallel-corpus Marathi side (no punctuation) | v1 |
| [ai4bharat/Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi) | 15,811 | All India Radio news text | v1 |
| [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | 9,269 | Speech transcripts (spontaneous + read) | v1 |
| [ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) | 2,337 | Read-speech prompts | v1 |
| [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) | 405 | TTS prompts | v1 |
| **Total** | **322,222** | | |

- **Excluded:** 390 sentences from [ai4bharat/MILU](https://huggingface.co/datasets/ai4bharat/MILU). MILU is an evaluation benchmark, so releasing its text would contaminate MILU scores.
- **Never used:**
  - machine-translated or LLM-generated text, including Sangraha's `synthetic` subset
  - audio-only data
- **Deduplicated:** each sentence appears once across all sources. `all_sources` and `dup_count` record where else it appeared.

## Fields

| Field | Meaning |
|---|---|
| `sentence_id` | Stable id: first 16 hex chars of SHA-1 of the normalized text |
| `text` | The cleaned Marathi sentence |
| `label` | Always `AGRI` in this release |
| `confidence` | Model probability of the label (0–1). 95% of rows are ≥ 0.9 |
| `word_count`, `char_count`, `length_bucket` | Size (`short` ≤8 words, `medium` ≤20, `long` ≤40, `very_long` >40) |
| `has_end_punct` | Ends with । . ? ! |
| `split_method` | `punctuation` (split from a longer row), `none` (row was one sentence), `unsplit_long` (long row without punctuation) |
| `source_dataset`, `source_file`, `source_row` | Exact upstream origin |
| `parent_row_id`, `sentence_index` | Upstream row/document id and the sentence's position in it |
| `all_sources`, `dup_count` | Every dataset the sentence appeared in, and how often |
| `source_license` | License of `source_dataset` as stated on its card |
| `license_note` | Set when the upstream license is unclear (Samanantar copy); null otherwise |
| `doc_id`, `doc_p_agri` | Sangraha only: upstream document id and the document-level P(agriculture) |
| `model_id`, `prompt_version` | `google/gemma-4-26B-A4B-it`; `v1` or `v2` |

## How it was built
1. **Read:** only the text columns of the upstream datasets, pinned to exact commits.
2. **Sangraha document screening:** Gemma first screened 501,062 verified web documents (from the opening and evenly spaced sentences) and kept the 33,821 that are about agriculture.
3. **Prepare:**
   - clean: Unicode NFC; remove tags, URLs and invisible characters
   - split into sentences on । ? ! and sentence-final full stops
   - filter: 3–200 words and ≥ 80% Devanagari
   - deduplicate exactly across sources
4. **Classify:** each sentence is labelled `AGRI` / `NON_AGRI` / `NOT_MARATHI` by **Gemma 4 26B A4B** on vLLM:
   - zero-shot with 27 Marathi few-shot examples
   - temperature 0
   - output constrained to the three labels
5. **Definition of AGRI:** core farming plus the farm economy:
   - crops, soil, irrigation, plant protection, horticulture
   - livestock, fisheries and allied activities
   - farm machinery
   - mandi prices, MSP, crop loans and insurance, schemes
   - weather only when its effect on crops is stated

   Mere mentions of farmers, idioms, recipes and nutrition, gardening, drinking water and purely political news are excluded.

Full technical details are in `docs/TECHNICAL.md` in the code repository: model choice, every vLLM setting, the full prompts, and evaluation.

## Quality
| Check | Result |
|---|---|
| Random spot-check, 100 Sangraha AGRI sentences (by the pipeline author, not independent) | ~92–95% correct |
| Prompt v2 on a hard-case held-out set (200) | AGRI precision 0.50, recall 0.89, accuracy 0.84 |
| Prompt v1 on the same held-out set | AGRI precision 0.36. **The 60,535 v1-labelled rows have not been checked on a random sample** |
| Exact duplicates | 0 |

The held-out set is deliberately full of hard cases, so it understates precision on typical rows. A native-speaker review is pending.

## Known issues
- **Glued sentences:** ~4,100 rows (~1.3%) contain a literal `\n` from the source and hold two or more sentences, sometimes with web boilerplate.
- **Wrong splits:** ~0.3% of rows are cut at Devanagari initials or unit abbreviations ("डॉ. एस. के.", "किग्रॅ.").
- **Near-duplicates:** ~0.2% differ only in numbers; ~1–2% are template sentences (market rates, sugar-factory figures).
- **Samanantar rows:** no punctuation; long rows are tagged `unsplit_long`.
- **Recall:** agriculture sentences inside Sangraha documents judged non-agricultural were never seen.

## Licensing
This dataset is a selection of sentences from upstream datasets, and **each row keeps its upstream license** (`source_license`):
- **CC-BY-4.0:** Sangraha, Shrutilipi, IndicVoices, Kathbath, Rasa (289,509 rows). Attribution to the original creators is required; see the citations below.
- **Unclear: Samanantar copy (32,713 rows).** `shivam/marathi_samanantar_processed` states no license. The original Samanantar is released openly, but this copy's terms are unverified. These rows carry `license_note`. Filter them out (`license_note is null`) if you need a clean CC-BY-4.0 subset.

**The license for the selection, labels and metadata added by this project is not decided yet** and will be set before public release. The pipeline code is Apache-2.0.

## Citation
Please cite the upstream datasets:
- **Sangraha:** Khan et al., *IndicLLMSuite*, ACL 2024
- **Samanantar:** Ramesh et al., TACL 2022
- **Shrutilipi:** Bhogale et al., ICASSP 2023
- **IndicVoices:** Javed et al., ACL Findings 2024
- **Kathbath:** Javed et al., *IndicSUPERB*, AAAI 2023
- **Rasa:** AI4Bharat

Citation for this corpus: see `CITATION.cff` in the code repository.
