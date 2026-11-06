# Marathi agriculture corpus: run report

Generated 2026-11-06T06:50:16+00:00 · model `google/gemma-4-26B-A4B-it` · current prompt `v2` (each row records the prompt that labelled it in `prompt_version`)

## Headline

- **AGRI sentences: 322,612** (target 200,000: **met**)
- Unique sentences classified: 4,196,610 of 4,196,610
- Unclassified (no result yet): 0 · request errors: 0

## Funnel per source

| source | raw rows | sentences after split | after filter | unique (kept copy) | already in a higher-priority source |
|---|---|---|---|---|---|
| shivam/marathi_samanantar_processed | 3,047,226 | 3,049,806 | 2,961,023 | 2,855,635 | 5,261 |
| ai4bharat/Kathbath | 86,448 | 86,448 | 86,448 | 86,334 | 0 |
| ai4bharat/Shrutilipi | 336,475 | 336,475 | 328,157 | 301,198 | 13 |
| ai4bharat/Rasa | 29,955 | 33,665 | 29,684 | 24,819 | 524 |
| ai4bharat/IndicVoices | 302,939 | 302,595 | 256,823 | 251,501 | 476 |
| ai4bharat/indicvoices_r | 19,141 | 19,141 | 18,858 | 23 | 18,806 |
| ai4bharat/MILU | 7,734 | 15,614 | 13,779 | 13,248 | 3 |
| ai4bharat/sangraha | 33,821 | 718,093 | 693,454 | 663,852 | 7,478 |

Duplicates are kept once, credited to the highest-priority source (order in `configs/sources.yaml`); `all_sources` on each row lists every source it appeared in.

## Document filter (general web sources)

Gemma reads a spread excerpt of each document (200 words: the opening 100 + evenly spaced sentences) and the document is kept if labelled AGRI or P(AGRI) ≥ 0.2. Only kept documents are split into sentences; agri sentences inside dropped documents are not recovered.

| source_key | source_file | docs_read | labelled_agri | kept | kept_pct |
|---|---|---|---|---|---|
| sangraha | verified/mar/data-0.parquet | 166,951 | 11,202 | 11,362 | 6.8 |
| sangraha | verified/mar/data-1.parquet | 167,107 | 11,069 | 11,253 | 6.7 |
| sangraha | verified/mar/data-2.parquet | 167,004 | 11,073 | 11,206 | 6.7 |

| P(AGRI) | documents |
|---|---|
| <0.01 | 465,894 |
| 0.01–0.2 | 1,347 |
| 0.2–0.5 | 496 |
| 0.5–0.8 | 514 |
| 0.8–0.99 | 1,122 |
| ≥0.99 | 31,689 |

## Dropped rows by reason

| reason | rows |
|---|---|
| too_short | 157,721 |
| low_devanagari | 15,822 |
| empty_after_clean | 344 |
| too_long | 68 |

## Labels per source (unique sentences)

| source_dataset | AGRI | NON_AGRI | NOT_MARATHI | TOTAL |
|---|---|---|---|---|
| ai4bharat/IndicVoices | 9,269 | 212,748 | 29,484 | 251,501 |
| ai4bharat/Kathbath | 2,337 | 83,057 | 940 | 86,334 |
| ai4bharat/MILU | 390 | 11,892 | 966 | 13,248 |
| ai4bharat/Rasa | 405 | 23,258 | 1,156 | 24,819 |
| ai4bharat/Shrutilipi | 15,811 | 280,836 | 4,551 | 301,198 |
| ai4bharat/indicvoices_r | 0 | 23 | 0 | 23 |
| ai4bharat/sangraha | 261,687 | 397,874 | 4,291 | 663,852 |
| shivam/marathi_samanantar_processed | 32,713 | 2,708,950 | 113,972 | 2,855,635 |

## Labels per prompt version

| prompt_version | AGRI | NON_AGRI | NOT_MARATHI | TOTAL |
|---|---|---|---|---|
| v1 | 60,925 | 3,320,764 | 151,069 | 3,532,758 |
| v2 | 261,687 | 397,874 | 4,291 | 663,852 |

## Labels per length bucket

short ≤8 words · medium ≤20 · long ≤40 · very_long >40

| length_bucket | AGRI | NON_AGRI | NOT_MARATHI | TOTAL |
|---|---|---|---|---|
| short | 68,515 | 1,862,536 | 131,660 | 2,062,711 |
| medium | 191,164 | 1,537,037 | 20,543 | 1,748,744 |
| long | 56,765 | 291,993 | 2,932 | 351,690 |
| very_long | 6,168 | 27,072 | 225 | 33,465 |

## Classifier confidence

| label | n | mean_conf | share_below_0_7 |
|---|---|---|---|
| AGRI | 322,612 | 0.985 | 0.018 |
| NON_AGRI | 3,718,638 | 0.996 | 0.005 |
| NOT_MARATHI | 155,360 | 0.933 | 0.088 |

## Sources

| dataset | url | license | revision | used |
|---|---|---|---|---|
| ai4bharat/Kathbath | https://huggingface.co/datasets/ai4bharat/Kathbath | CC-BY-4.0 | 5b9e92849222 | yes |
| ai4bharat/Shrutilipi | https://huggingface.co/datasets/ai4bharat/Shrutilipi | CC-BY-4.0 | f4beb2f2cfae | yes |
| ai4bharat/Rasa | https://huggingface.co/datasets/ai4bharat/Rasa | CC-BY-4.0 | 632f55c7ac59 | yes |
| ai4bharat/IndicVoices | https://huggingface.co/datasets/ai4bharat/IndicVoices | CC-BY-4.0 | c96f9088f138 | yes |
| ai4bharat/indicvoices_r | https://huggingface.co/datasets/ai4bharat/indicvoices_r | CC-BY-4.0 | 5f4495c91d50 | yes |
| ai4bharat/MILU | https://huggingface.co/datasets/ai4bharat/MILU | CC-BY-4.0 | 846d7016558b | yes |
| shivam/marathi_samanantar_processed | https://huggingface.co/datasets/shivam/marathi_samanantar_processed | unknown (derived from Samanantar, CC0) | 1b42b3e6bcd3 | yes |
| ai4bharat/sangraha | https://huggingface.co/datasets/ai4bharat/sangraha | CC-BY-4.0 | 8b813c3f62d3 | yes |

## Output files

- `output/sentences.parquet`: every unique sentence with all metadata
- `output/agri.jsonl`, `output/non_agri.jsonl`, `output/not_marathi.jsonl`
- `output/rejected.parquet`: rows dropped during cleaning/filtering, with reason
- `output/samples/*.csv`: random samples per label for spot-checking
