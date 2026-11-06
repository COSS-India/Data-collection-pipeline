# Technical notes: classifier selection, hosting, vLLM settings, prompts and the Sangraha run

This document explains how the Marathi agriculture corpus was labelled: which model, how it is served, each setting and why it was chosen, the prompts, and the end-to-end Sangraha run. All numbers come from the logs in `logs/`, `output/report.md` or direct queries on `output/sentences.parquet`. Those folders are not versioned; the repository keeps copies in `reports/` (`report.md`, `manifest.json`, `logs/`, including a startup excerpt of the vLLM server log).

---

## 1. Summary

| Item | Value |
|---|---|
| Goal | 200k+ natural, human-written Marathi agriculture sentences, the source side of an MT parallel corpus, for open release |
| Classifier | `google/gemma-4-26B-A4B-it`, zero-shot plus few-shot prompting, no fine-tuning |
| Serving | vLLM 0.33.0, BF16, 1× NVIDIA RTX PRO 6000 Blackwell (96 GB), OpenAI-compatible API on `127.0.0.1:8000` |
| Labels | Sentence: `AGRI` / `NON_AGRI` / `NOT_MARATHI`. Document: `AGRI` / `NON_AGRI` |
| Result | **322,612 AGRI sentences**: 60,925 from 7 sources (prompt v1) + 261,687 from Sangraha (prompt v2) |
| Unique sentences labelled | 4,196,610, with 0 request errors |

Pipeline: ingest (+ doc-filter for Sangraha) → clean → split → filter → dedup → classify → report → release. Diagrams, per-stage files, commands and counts are in [`PIPELINE.md`](PIPELINE.md).

---

## 2. How the classifier was chosen

### 2.1 Constraints that shaped the choice
| Constraint | Consequence |
|---|---|
| No labelled Marathi agriculture data, and no annotator | A supervised classifier (e.g. fine-tuned MuRIL) cannot be trained without first making training data |
| The topic boundary needed iteration ("is crop-damage compensation agri? is a recipe?") | The definition must be easy to change. A prompt can be edited; a trained model must be retrained |
| The language must be checked too: Hindi, Konkani and Sanskrit are written in Devanagari | Language and topic must be judged in one pass |
| Millions of sentences on one GPU | The model must be fast: small active parameter count, batch-friendly serving |
| Open release | The model license must allow it to be used to produce a released dataset |

### 2.2 Options considered

| Option | What it is | Why not chosen (or chosen) |
|---|---|---|
| Keyword glossary (e.g. the Mahavistaar agri term list) | String matching | Misses implicit agri text, matches idioms and brands ("शेण", "गोड ऊस"). Considered only as a speed prefilter, then dropped when the LLM proved fast enough |
| Fine-tuned MuRIL / IndicBERT | Supervised encoder classifier | Needs thousands of labelled examples that did not exist |
| `laya-multilingual` / Yama (`arunp0/laya-indic`) | Zero-shot embedding-style classifiers | General `laya-multilingual` scores weakly on non-English text (0.45 on an intent benchmark). Yama is built for Indic languages but has no independent evaluation. Kept as a fallback first pass only if Gemma was too slow; it was not needed |
| Qwen3-30B-A3B | Open MoE LLM, ~3B active | A credible alternative of similar speed |
| Sarvam-M (24B) | Dense LLM tuned for Indian languages | Possibly better on Marathi nuance, but dense, so several times slower per token |
| **Gemma 4 26B A4B (chosen)** | Open MoE LLM, ~4B active of ~26B total | See below |

### 2.3 Why Gemma 4 26B A4B
- **Suggested by the project team**, and it fits every constraint:
  - **License:** Apache-2.0 and not gated. No access request, and no restriction on using its outputs in a released dataset.
  - **Speed:** mixture-of-experts with about 4B active parameters per token. It runs at roughly 4B-model cost with the understanding of a much larger model.
  - **Languages:** trained on 140+ languages, including Marathi.
  - **Fits one GPU:** about 52 GB of weights in BF16, which fits a 96 GB GPU unquantized with ~33 GB left for KV cache.
  - **Supported:** vLLM supports the architecture (`Gemma4ForConditionalGeneration`), including constrained output.
- **Honest note: no head-to-head bake-off was run.** The plan proposed comparing Gemma, Qwen and Yama on ~2k sentences. The project owner chose to go straight to Gemma ("don't compare, just do this"). Quality was measured afterwards instead:
  - **Prompt evaluation:** 600 sentences with gold labels, see §6.4.
  - **Random spot-checks:** of the output, see §8.
  - A comparison with another model remains possible with the same code by changing `classify.model`.

---

## 3. Hosting

### 3.1 Hardware and software
| Item | Value |
|---|---|
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB (94.97 GiB usable) |
| CPU / RAM | 8 cores / 62 GB |
| CUDA | Driver only. No CUDA toolkit (no `nvcc`) on the machine |
| Server | vLLM 0.33.0, installed in the project venv (`.venv/bin/vllm`) |
| Model | `google/gemma-4-26B-A4B-it`, downloaded from Hugging Face on first start (~52 GB) |
| Launch | `scripts/serve.sh` (`nohup scripts/serve.sh > logs/vllm_server.log 2>&1 &`) |
| API | OpenAI-compatible `/v1/chat/completions` on `127.0.0.1:8000`. Readiness check: `curl -s http://127.0.0.1:8000/v1/models` |

### 3.2 What the server does at start (from `logs/vllm_server.log`)
| Step | Observed |
|---|---|
| Architecture | `Gemma4ForConditionalGeneration`, 1 GPU (no tensor or pipeline parallelism) |
| Attention backend | **Triton**. vLLM forces it because Gemma 4 mixes head dimensions (256 for sliding-window layers, 512 for full-attention layers) and FlashAttention-4 is not available for that |
| MoE backend | Triton unquantized MoE (`TritonExperts`). FlashInfer MoE kernels are unavailable without `nvcc` |
| Chunked prefill | On, with `max_num_batched_tokens=16384` |
| Weights + non-torch memory | 50.84 GiB |
| Peak activations | 1.31 GiB |
| CUDA graphs | 0.21 GiB (piecewise + full graphs captured) |
| **KV cache** | **33.33 GiB = 158,288 tokens** |
| Engine init | ~42 s after weights load. The total cold start with cached weights was ~7 minutes |

---

## 4. vLLM server settings (`scripts/serve.sh`)

Each flag: what it does, the value used, and why.

| Setting | Value | What it does | Why this value |
|---|---|---|---|
| `--dtype` | `bfloat16` | Weight and compute precision | The model's native precision. No quantization, so labels are not affected by quantization error. It fits the GPU, so there was no reason to quantize |
| `--max-model-len` | `4096` | Maximum prompt + output tokens per request | The sentence prompt is 1,935 tokens. Sentences are at most 200 words (the filter stage drops longer ones), and the output is ≤5 tokens. The document prompt is 754 tokens + a ≤200-word excerpt. 4096 covers both with margin. A smaller limit also lowers the per-sequence KV reservation |
| `--gpu-memory-utilization` | `0.90` | Share of GPU memory vLLM may use (85.5 GiB here) | Leaves ~9 GiB headroom for the CUDA context and fragmentation. Everything above the weights becomes KV cache (33 GiB) |
| `--enable-prefix-caching` | on | Reuses KV blocks of identical prompt prefixes across requests | **The biggest speed lever.** Every request starts with the same ~1,935-token system prompt + 27 few-shot turns. With caching, that prefix is computed once and only the sentence (~10–40 tokens) is new work per request |
| `--max-num-seqs` | `512` | Maximum sequences scheduled at once | Requests are tiny (a few new prompt tokens, ≤5 output tokens), so many can run together. 512 is above the client's 256 in-flight requests, so the server never limits concurrency |
| `--max-num-batched-tokens` | `16384` | Token budget per scheduler step (also the chunked-prefill chunk size) | Lets one step take many new short prompts. Large enough to fill the GPU; small enough to keep step latency low |
| `--limit-mm-per-prompt.image` | `0` | Maximum images per prompt | Gemma 4 is multimodal. Text-only use means vLLM does not reserve memory or profiling for the vision encoder |
| `--generation-config` | `vllm` | Ignore the model repo's `generation_config.json` sampling defaults | The model repo ships sampling defaults (`do_sample: true`, temperature 1.0, top_k 64, top_p 0.95). With `vllm`, only the request's own parameters apply (temperature 0), so runs are deterministic |
| `--async-scheduling` | on | Overlaps CPU scheduling of the next step with GPU execution of the current one | With 1-token-ish requests, CPU scheduling overhead is a large share of each step. Overlapping it raises throughput |
| `--api-server-count` | `4` | Number of API front-end processes (HTTP parsing, tokenization, detokenization) | At ~300+ requests/s one Python front-end process becomes CPU-bound. Four spread that work over the 8 cores |
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind address | Reachable from this machine only. No authentication is needed because nothing outside can connect |
| env `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Use vLLM's PyTorch sampler instead of FlashInfer's | FlashInfer's sampler JIT-compiles CUDA and needs `nvcc`, which is not installed. At temperature 0 (greedy) both samplers pick the same token, so labels are unchanged |

**Deliberately not used:**

| Not used | Why |
|---|---|
| Quantization (FP8 / AWQ / GPTQ) | BF16 fits; no reason to risk label drift |
| `--reasoning-parser` / thinking mode | Thinking would generate hundreds of tokens per sentence. It stays off (see `enable_thinking: false` below) |
| Tensor or pipeline parallelism | One GPU holds the whole model |
| Speculative decoding | Outputs are 2–5 tokens, so there is nothing to speed up |

---

## 5. Request settings (each API call)

Built in `src/marathi_agri/classify.py` (`build_request`, `build_doc_request`).

| Field | Sentence request | Document request | Why |
|---|---|---|---|
| `messages` | system prompt + 27 few-shot user/assistant turns + `<text>…</text>` | system prompt + 4 few-shot turns + `<doc>…</doc>` | Fixed prefix, so prefix caching works. The input sits in tags so the model never confuses the text with instructions |
| `temperature` | `0` | `0` | Greedy and deterministic: the same input always gets the same label |
| `max_tokens` | `5` | `5` | The longest label, `NOT_MARATHI`, is exactly 5 tokens (`NOT` `_` `MAR` `ATH` `I`). `AGRI` is 2 and `NON_AGRI` is 4 |
| `structured_outputs` | `{"choice": ["AGRI","NON_AGRI","NOT_MARATHI"]}` | `{"choice": ["AGRI","NON_AGRI"]}` | Constrained decoding: the model can *only* emit one of the labels. No parsing failures, no explanations, no stray text |
| `chat_template_kwargs` | `{"enable_thinking": false}` | same | Gemma 4's chat template inserts an empty thought block. The answer starts immediately |
| `logprobs` | `true` | `true` | Gives a probability for each emitted token |
| `top_logprobs` | not set | `5` | The document filter needs the probability of the label that was *not* chosen too (see P(AGRI) below) |

**How the scores are computed:**
- **Sentence `confidence`** = `exp(sum of logprobs of the emitted label's tokens)`: the model's probability of the full label it gave. Rows keep this, so users can apply their own cutoff. 238,055 of 261,687 Sangraha AGRI sentences have confidence ≥ 0.99.
- **Document `P(AGRI)`** is read from the *first* generated token's top-5 alternatives:
  - The first tokens differ: `AGRI` starts with `AG`, `NON_AGRI` with `NON`.
  - Each candidate token is credited to the label it is a prefix of. A token that could start both labels, or neither, is ignored.
  - The two masses are normalized: `P(AGRI) = m_AGRI / (m_AGRI + m_NON_AGRI)`.
  - If no candidate is usable, it falls back to 1.0 for an `AGRI` answer and 0.0 otherwise.

---

## 6. Client settings (how requests are sent)

Settings in `configs/pipeline.yaml` → `classify:`.

| Setting | Value | Why |
|---|---|---|
| HTTP library | `aiohttp` (async) | Thousands of concurrent short requests from few threads |
| `client_processes` | `4` | One Python process tops out at ~300 requests/s (JSON encoding + decoding). Four processes remove the client bottleneck |
| `concurrency` | `256` in flight in total (64 per process) | Keeps the server's batch full without exceeding `max-num-seqs 512` |
| Request body | Built once per sentence from a pre-encoded JSON template (`RequestEncoder`) | Avoids re-serializing the ~2k-token prompt for every request |
| `timeout_s` | `120` | Generous; requests normally finish in well under a second |
| `max_retries` | `5`, exponential backoff `min(30, 2^attempt) s` + jitter | Retries HTTP 408, 429, 500, 502, 503 and 504, plus timeouts and connection errors. Malformed responses are not retried and are recorded as errors |
| `shard_size` | `50,000` results per parquet shard | An interrupted run resumes from the last finished shard |
| Result key | `sentence_id` + `prompt_version` | A sentence is never re-sent under the same prompt. A new prompt version can co-exist with the old one, and the report takes the newest per sentence |

**Measured effect of the client design:** the first, simple client managed **64 sentences/s**. Adding API-server processes alone did not help: the bottleneck was the Python client. With async aiohttp, pre-encoded bodies and 4 processes it reached **~340 sentences/s** (trial: 20,000 sentences in 59.2 s = 338/s).

---

## 7. Prompts

Both prompts live in `src/marathi_agri/prompt.py`; the full text is in the [Appendix](#appendix-full-prompt-texts).

### 7.1 Shared agriculture definition ("core farming + agri economy")
Ten topic groups:
1. Crops
2. Soil and nutrients
3. Water and irrigation
4. Plant protection
5. Horticulture and plantation crops
6. Livestock and allied activities (dairy, poultry, fisheries, beekeeping…)
7. Farm machinery and technology
8. Agri economy (APMC prices, MSP, procurement, credit, insurance, schemes, FPOs, agro-processing)
9. Weather **only** when its effect on crops is stated
10. Agri institutions

Both prompts use this same text, so the document filter and the sentence classifier agree on what "agri" means.

### 7.2 Sentence prompt structure
| Part | Content | Why |
|---|---|---|
| Role | "Expert annotator building a Marathi agriculture text corpus" | Sets the task framing |
| Instructions language | English | The model follows English instructions most reliably. The data stays Marathi |
| Step 1: language | Decide `NOT_MARATHI` first, with Marathi, Hindi and Konkani signal words | Devanagari is shared by Hindi, Konkani and Sanskrit. Without explicit cues these get labelled Marathi |
| Step 2: topic | AGRI definition + explicit NON_AGRI list + rules | Most errors are false AGRI on texts that merely *mention* farming |
| Few-shot | 27 Marathi examples as real user/assistant turns | Examples teach the boundary better than rules alone. As chat turns they also show the exact answer format |
| Tie-break | "When unsure → NON_AGRI" | Precision matters more than recall: the corpus is meant to be agri-only |
| Output | "Exactly one label. No explanation." | Backed by constrained decoding |

Size: **1,935 tokens** for the fixed part, shared by every request through prefix caching.

### 7.3 v1 → v2: what changed and which v1 error each change fixes
| v1 problem (seen in v1 output) | v2 change |
|---|---|
| 151k sentences labelled `NOT_MARATHI`, many of them short or spoken Marathi ("हो हो चालेल") | Step 1 judges **language only, never quality, length or completeness**. Fragments, headlines, fillers and spelling errors are Marathi |
| Fluent Marathi about other states or countries labelled `NOT_MARATHI` | "The TOPIC never decides the language" |
| Konkani had no explicit cues (only Hindi did) | Konkani signal words (हांव, आसा, म्हाका…) and a Konkani few-shot example |
| Nutrition facts as AGRI ("टोमॅटोमध्ये क जीवनसत्त्व…") | Nutrition or health benefits of produce → NON_AGRI |
| Idioms, a juice brand, cow dung for floors | Idioms, brands and household uses → NON_AGRI |
| Drinking-water supply, canal corruption | Drinking water; dams or canals without crop irrigation → NON_AGRI |
| House plants, botany lessons | Gardening and general botany → NON_AGRI |
| Bare word lists ("भोपळा आणि गाजर") | A bare list with no statement → NON_AGRI |
| "Industry, agriculture, education…" budget lines | Agriculture as one item in a list of sectors → NON_AGRI |
| Protests and loan waivers: inconsistent labels | **Borderline rule:** AGRI only with farm-economy substance (amounts, scheme terms, prices, crop damage); the political event alone → NON_AGRI |
| "Farmers in distress again" as AGRI | Farmer distress or "help farmers" without a crop, activity, price, amount or scheme detail → NON_AGRI |
| Informal farm talk missed | "Everyday farm talk" (weeding, fodder, selling produce) → AGRI |

14 few-shot examples were added: 13 in v1, 27 in v2.

### 7.4 Evaluation: v1 vs v2
Gold labels on 600 sentences drawn from the corpus and deliberately enriched with hard cases (`data/eval/`). Reproduce with `marathi-agri eval-prompt --csv data/eval/holdout.csv`.

| Set | Prompt | AGRI precision | AGRI recall | Accuracy | Marathi wrongly marked NOT_MARATHI |
|---|---|---|---|---|---|
| Dev (400; v2 was tuned once on it) | v1 | 0.51 | 0.97 | 0.57 | 99 |
| | v2 | 0.67 | 0.83 | 0.83 | 22 |
| **Held-out (200; never used for tuning)** | v1 | 0.36 | 1.00 | 0.51 | 50 |
| | **v2** | **0.50** | **0.89** | **0.84** | **5** |

**Caveats:**
- The gold labels were written by Claude, who also wrote the prompt. This is not an independent measurement.
- The sets are hard-case-heavy, so precision on random corpus sentences is much higher (§8).
- Raising the confidence cutoff barely helps: at ≥ 0.99 on dev, precision is 0.72 and recall 0.68. Errors are confident, not borderline.

### 7.5 Document prompt `d1` (Sangraha first pass)
| Part | Content | Why |
|---|---|---|
| Input | A **spread-read excerpt** of one web document, in `<doc>…</doc>` | See §8.2 |
| Rule | AGRI if the main topic **or a substantial part** is farming or the farm economy | Digest pages mix stories. A document with one solid agri story should be kept |
| Bias | "This is a first-pass filter: if a substantial part… answer AGRI" | A missed document loses all its agri sentences; a wrongly kept document only costs GPU time, because every sentence is re-judged |
| Few-shot | 4 excerpts: soybean sowing (AGRI), municipal election that mentions farmers (NON_AGRI), milk procurement price (AGRI), carrot halwa recipe (NON_AGRI) | Each pair contrasts a real agri topic with a near-miss |
| Labels | AGRI / NON_AGRI only | Language is checked later, per sentence |

Size: **754 tokens** fixed.

---

## 8. The Sangraha run, end to end

### 8.1 Source choice
| Decision | Value | Why |
|---|---|---|
| Dataset | `ai4bharat/sangraha`, revision `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8` (pinned) | Large, human-written Marathi web text, CC-BY-4.0 |
| Subset | **`verified` only** | `unverified` has fewer quality checks; `synthetic` is machine-translated, which breaks the "natural text only" rule |
| Files | `verified/mar/data-0/1/2.parquet` (3 of 34) | Enough to pass the 200k target; more can be added later as config entries |
| Rows | `type == "web"` only | `pdf` rows are OCR (the sample seen was Sanskrit grammar); `speech` rows are transcripts |
| Unit | Each row is **one whole document** (`doc_id`, `text`, `type`). Median 229 words | Splitting into sentences is done by this pipeline |
| Old sources | Not re-labelled | Instruction from the project owner: classify only Sangraha with v2 and keep the 7 earlier sources' v1 labels |

Why a document filter at all: only ~7% of Sangraha documents are about agriculture. Classifying every sentence of a file (~3.5M sentences) would take ~4 h of GPU per file. Screening documents first takes ~45 minutes per file in total.

### 8.2 Stage 1: document filter (`marathi-agri doc-filter`, `src/marathi_agri/docfilter.py`)
**Spread read** (deterministic, `spread_excerpt`):
1. Clean the document, then split it into sentences.
2. If it is ≤200 words, use it whole.
3. Otherwise take whole opening sentences up to **100 words** (`doc_head_words`), because news puts the topic first.
4. Fill the remaining budget up to **200 words** (`doc_excerpt_words`) with sentences at **evenly spaced positions** through the rest, in document order, so an agri story in the middle of a digest page is still seen.
5. Mark gaps between non-adjacent parts with ` … `.

Rejected alternatives:
- **First 200 words only:** misses the middle of digest pages.
- **Random sentences:** can skip the headline, and is not reproducible.
- **Chunk-by-chunk or whole document:** 2–3× the GPU cost.

**Keep rule:** keep if the label is `AGRI` **or** `P(AGRI) ≥ 0.2` (`doc_keep_threshold`). The threshold leans toward recall because every sentence is judged again in stage 3.

**Results:**

| File | Web documents | Labelled AGRI | Kept (incl. P ≥ 0.2) | Kept % | Time | Speed |
|---|---|---|---|---|---|---|
| data-0 | 166,951 | 11,202 | 11,362 | 6.8% | 1,848 s | 90 docs/s |
| data-1 | 167,107 | 11,069 | 11,253 | 6.7% | 1,876 s | 89 docs/s |
| data-2 | 167,004 | 11,073 | 11,206 | 6.7% | 1,877 s | 89 docs/s |
| **Total** | **501,062** | 33,344 | **33,821** | 6.7% | ~93 min | 0 errors |

P(AGRI) distribution over all 501,062 documents:

| P(AGRI) | Documents |
|---|---|
| < 0.01 | 465,894 |
| 0.01–0.2 | 1,347 |
| 0.2–0.5 | 496 |
| 0.5–0.8 | 514 |
| 0.8–0.99 | 1,122 |
| ≥ 0.99 | 31,689 |

The model is decisive: only ~3.5k documents fall between 0.01 and 0.99. The 0.2 threshold adds 477 documents on top of the AGRI label.

**Audit trail:**
- `data/docfilter/sangraha/<file>.parquet` records every document read, with `doc_label`, `doc_p_agri`, `kept`, model and prompt version.
- Kept documents go to `data/raw/sangraha/` in the standard raw schema, so the existing stages run unchanged.
- The stage can be resumed per file. If any request fails, nothing is written for that file.

### 8.3 Stage 2: prepare (`marathi-agri prepare`, CPU only, ~2 minutes for all sources)
| Step | Sangraha numbers | Notes |
|---|---|---|
| Kept documents | 33,821 | |
| Split into sentences | 718,093 | Split on `।` `?` `!` and sentence-final `.`; abbreviations, decimals and list numbers are protected; newlines are hard breaks |
| After filter | 693,454 | Dropped: fewer than 3 words, more than 200 words, or less than 80% Devanagari letters |
| Already in the old corpus | 7,478 | Exact match on the normalized key. The old copy is kept because Sangraha is last in `sources.yaml` priority |
| **Unique new sentences** | **663,852** | Each keeps `doc_id`, `parent_row_id` (file + row), `sentence_index` and `doc_p_agri` |

### 8.4 Stage 3: sentence classification (`marathi-agri classify --only sangraha`, prompt v2)
| Metric | Value |
|---|---|
| Sentences | 663,852 (Sangraha only; the 3,532,758 older sentences kept their v1 labels) |
| Time | 46 minutes (06:03 → 06:49) |
| Speed | ~240 sentences/s, vs ~335/s in phase 1. Sangraha sentences are longer: mean 13.8 words vs 10.2 |
| Errors | 0 |
| **AGRI** | **261,687 (39.4%)** |
| NON_AGRI | 397,874 |
| NOT_MARATHI | 4,291 (0.6%) |

Why only ~40% of sentences from agri documents are AGRI: even an agri article contains bylines, dates, political names and event details, and each sentence is judged on its own. The AGRI share rises with sentence length: 27% for short sentences, 42% for medium, 50% for long.

### 8.5 Final corpus
| Source group | Prompt | AGRI |
|---|---|---|
| 7 earlier sources (Samanantar, Shrutilipi, IndicVoices, Kathbath, Rasa, MILU, IndicVoices-R) | v1 | 60,925 |
| Sangraha verified (3 files) | v2 | 261,687 |
| **Total** | | **322,612** |

Every row records `prompt_version`, `confidence`, `source_dataset`, `source_file`, `source_row`, `source_license`, `all_sources`, `dup_count`, and for Sangraha also `doc_id` and `doc_p_agri`.

---

## 9. Quality and known limitations

**Spot-check** (Claude, not independent): 100 random Sangraha AGRI sentences → 92 clearly correct, 3 borderline, 5 wrong, so **~92–95% precision**.
- **Errors:** passing mentions or admin and political news (a list of meeting topics, an APMC election, appointments to an agri-university board).
- **Independent check:** a second sample of 100 for review by a native speaker is in `data/eval/sangraha_agri_check.csv`.

| Limitation | Size | Note |
|---|---|---|
| No independent (human) accuracy measurement yet | — | Needed before public release |
| v1-labelled portion never checked on a random sample | 60,925 AGRI | v1 precision on the hard held-out set was 0.36 |
| Glued sentences: the source contains a literal `\n` (backslash + n) that the splitter does not treat as a break | ~4,100 AGRI (1.3%) | Some carry web boilerplate |
| Splitter cuts at Devanagari initials and units ("डॉ. एस. के.", "किग्रॅ.") | ~0.3% of AGRI | |
| Near-duplicates (differ only in numbers) and template sentences | ~0.2% and ~1–2% | Exact duplicates: 0 |
| Document-filter recall: agri sentences inside documents judged NON_AGRI are never seen | Not measured | Accepted trade-off for ~5× less GPU time |
| Short, fluent Marathi sometimes labelled NOT_MARATHI | Small | Model behaviour, also seen without constrained decoding |

---

## 10. Reproduce

```bash
nohup scripts/serve.sh > logs/vllm_server.log 2>&1 &     # start the classifier server
curl -s http://127.0.0.1:8000/v1/models                  # wait until it lists the model
.venv/bin/marathi-agri doc-trial --n 200                 # optional: check the document filter on 200 docs
.venv/bin/marathi-agri doc-filter                        # screen Sangraha documents
.venv/bin/marathi-agri prepare                           # clean → split → filter → dedup (all sources)
.venv/bin/marathi-agri classify --only sangraha          # label new Sangraha sentences with the current prompt
.venv/bin/marathi-agri report                            # write output/ and output/report.md
.venv/bin/marathi-agri eval-prompt --csv data/eval/holdout.csv   # re-score the prompt
```

All stages can be resumed. Settings are in `configs/pipeline.yaml` and sources in `configs/sources.yaml`.

---

## Appendix: full prompt texts

Generated from `src/marathi_agri/prompt.py` (sentence prompt `v2`, document prompt `d1`). Each request is: system message, then the few-shot examples as alternating user/assistant turns, then the input as the final user turn.

### A.1 Sentence prompt v2: system message

```text
You are an expert annotator building a Marathi agriculture text corpus.
You receive ONE text segment. Classify it with exactly one label.

STEP 1 – LANGUAGE CHECK. Judge ONLY the language, never the quality, length or completeness.
Output NOT_MARATHI only if the text is:
- Hindi, Konkani, Sanskrit, Nepali or any other language, even if written in Devanagari
- English, or Marathi written in Latin letters (romanized)
- meaningless gibberish or garbled OCR text
Marathi signals: आहे/आहेत, होते, नाही, आणि, च्या/चे/ची, ला/ना, मध्ये, साठी.
Hindi signals: है/हैं, था, नहीं, और, का/की/के, में, लिए.
Konkani signals: हांव, आसा, म्हाका, तुका, खूब, जाता.
These are STILL MARATHI, go to STEP 2: short phrases, headlines, incomplete fragments,
numbers or dates written in words, informal or spoken Marathi with fillers (हो हो, बरं, म्हणजे),
spelling mistakes, Marathi with a few English loanwords (ट्रॅक्टर, युरिया, ड्रोन).
The TOPIC never decides the language: Marathi text about other states, countries, foreign
people, places or brands is still Marathi.

STEP 2 – If Marathi, decide AGRI or NON_AGRI.
AGRI = the main subject is farming or the farm economy:
- Crops: sowing, seeds, varieties, kharif/rabi, growth, yield, harvest, storage
- Soil and nutrients: soil testing, fertilizer, manure, compost
- Water: irrigation, drip/sprinkler, farm ponds; dams/canals only when about irrigation
- Plant protection: pests, diseases, weeds, pesticides, spraying
- Horticulture and plantation crops: fruits, vegetables, flowers, sugarcane, cotton, nurseries
- Livestock and allied: dairy, cattle, goat, poultry, fisheries, beekeeping, sericulture, fodder, animal health
- Farm machinery and technology: tractors, implements, agri drones, polyhouse
- Agri economy: APMC/mandi arrivals and prices, MSP, procurement, agri exports, crop loans,
  crop insurance, farmer schemes and subsidies, FPOs, agro-processing, cold storage
- Weather ONLY when its effect on crops or farmers is stated (rain/drought/hail damaging crops)
- Agri institutions: agriculture department, KVK, agri universities, advice to farmers
- Everyday farm talk, even informal or incomplete: weeding, hoeing, watering or spraying crops,
  fodder and care of farm animals, selling one's produce, demand for fruits or flowers in markets
Borderline – farmer protests, loan waivers, compensation:
- AGRI if the text states farm-economy substance: compensation or loan amounts, scheme terms,
  crop prices, crop damage.
- NON_AGRI if it only reports the political event: who protested, roads blocked, statements
  with no farm content.
NON_AGRI:
- Politics, elections, protests with no farm-economy substance (see borderline rule)
- General weather forecasts with no mention of crops
- Food, cooking, recipes, diet, and nutrition or health benefits of fruits, vegetables or milk
- Farm words used as idioms, proverbs, metaphors, brand names, or household uses
  (e.g. cow dung for floors or rituals)
- Drinking-water supply; dams, canals or water projects described only as construction,
  contracts or corruption, without crop irrigation
- Gardening, house plants, trees in general, plant biology lessons not about growing crops
- A bare list of words with no statement (e.g. "भोपळा आणि गाजर")
- Agriculture mentioned only as one item in a list of sectors or topics
- Village life, crime, accidents, festivals; forests, wildlife, environment (unless about farming)
- Everything else: sports, films, business, health, education, etc.
Rules:
- Merely mentioning a farmer, a village or a crop name is NOT enough. The text must be about
  farm work, farm produce or farm income.
- Farmer distress, farmer suicides, or promises and demands to "help farmers" are NON_AGRI
  unless the text names a crop, a farm activity, a price, an amount or a scheme detail.
- When unsure between AGRI and NON_AGRI, choose NON_AGRI.

Answer with exactly one label: AGRI, NON_AGRI or NOT_MARATHI. No explanation.
```

### A.2 Sentence prompt v2: few-shot turns (27)

Each user turn is `<text>…</text>`; the assistant turn is the label.

| # | User text | Assistant label |
|---|---|---|
| 1 | सोयाबीनवरील खोडमाशीच्या नियंत्रणासाठी शेतकऱ्यांनी वेळेवर फवारणी करावी. | `AGRI` |
| 2 | टोमॅटोमध्ये क जीवनसत्त्व भरपूर प्रमाणात असते. | `NON_AGRI` |
| 3 | यंदा लासलगाव बाजार समितीत कांद्याला प्रति क्विंटल २२०० रुपये दर मिळाला. | `AGRI` |
| 4 | हो हो चालेल, मी उद्या सकाळी येतो | `NON_AGRI` |
| 5 | ठिबक सिंचनामुळे उसाच्या उत्पादनात तीस टक्क्यांनी वाढ झाली | `AGRI` |
| 6 | शहराचा पिण्याच्या पाण्याचा पुरवठा दोन दिवस बंद राहणार आहे. | `NON_AGRI` |
| 7 | गाईंना संतुलित पशुखाद्य दिल्यास दुधाचे उत्पादन वाढते. | `AGRI` |
| 8 | त्याची अवस्था म्हणजे जसे पेरले तसे उगवले अशीच झाली. | `NON_AGRI` |
| 9 | अवकाळी पावसामुळे द्राक्षबागांचे मोठे नुकसान झाले आहे. | `AGRI` |
| 10 | आमच्या बाल्कनीत तुळस आणि कोरफडीच्या कुंड्या आहेत. | `NON_AGRI` |
| 11 | अतिवृष्टीने बाधित शेतकऱ्यांना हेक्टरी दहा हजार रुपये मदत जाहीर झाली. | `AGRI` |
| 12 | शेतकरी संघटनांनी महामार्गावर रास्ता रोको आंदोलन केले. | `NON_AGRI` |
| 13 | पीक विमा योजनेसाठी शेतकऱ्यांना ३१ जुलैपर्यंत अर्ज करता येईल. | `AGRI` |
| 14 | भोपळा आणि गाजर | `NON_AGRI` |
| 15 | रोज सकाळी गुरांना चारा टाकायचा आणि मग शेतात खुरपणी करायची | `AGRI` |
| 16 | या निर्णयामुळे शेतकरी पुन्हा अडचणीत सापडला आहे. | `NON_AGRI` |
| 17 | दिवाळीमुळे बाजारात झेंडूच्या फुलांना मागणी वाढली आहे. | `AGRI` |
| 18 | बिहारमध्ये पुरामुळे अनेक गावांचा संपर्क तुटला आहे. | `NON_AGRI` |
| 19 | आमच्या शेतात यंदा कापूस लय चांगला आलाय | `AGRI` |
| 20 | उद्योग, शेती, शिक्षण आणि आरोग्य या क्षेत्रांसाठी अर्थसंकल्पात तरतूद करण्यात आली. | `NON_AGRI` |
| 21 | शेतकऱ्यांच्या प्रश्नांवर विरोधकांनी विधानसभेत सरकारला धारेवर धरले. | `NON_AGRI` |
| 22 | पुढील दोन दिवस मुंबईत मुसळधार पावसाचा इशारा देण्यात आला आहे. | `NON_AGRI` |
| 23 | पुरणपोळी बनवण्यासाठी हरभऱ्याची डाळ रात्रभर भिजत ठेवावी. | `NON_AGRI` |
| 24 | भारताने ऑस्ट्रेलियाविरुद्धचा सामना सहा गडी राखून जिंकला. | `NON_AGRI` |
| 25 | किसानों को इस साल गेहूं का अच्छा दाम मिला है। | `NOT_MARATHI` |
| 26 | हांव आयज बाजाराक वता. | `NOT_MARATHI` |
| 27 | shetkaryanna pik vimyacha labh milnar | `NOT_MARATHI` |

### A.3 Document prompt d1: system message

```text
You are an expert annotator building a Marathi agriculture text corpus.
You receive an EXCERPT of ONE Marathi web document: its opening, then sentences sampled
evenly from the rest of the document, separated by "…".
Decide whether the document is about agriculture.

AGRI = the document's main topic, or a substantial part of it, is farming or the farm economy:
- Crops: sowing, seeds, varieties, kharif/rabi, growth, yield, harvest, storage
- Soil and nutrients: soil testing, fertilizer, manure, compost
- Water: irrigation, drip/sprinkler, farm ponds; dams/canals only when about irrigation
- Plant protection: pests, diseases, weeds, pesticides, spraying
- Horticulture and plantation crops: fruits, vegetables, flowers, sugarcane, cotton, nurseries
- Livestock and allied: dairy, cattle, goat, poultry, fisheries, beekeeping, sericulture, fodder, animal health
- Farm machinery and technology: tractors, implements, agri drones, polyhouse
- Agri economy: APMC/mandi arrivals and prices, MSP, procurement, agri exports, crop loans,
  crop insurance, farmer schemes and subsidies, FPOs, agro-processing, cold storage
- Weather ONLY when its effect on crops or farmers is stated (rain/drought/hail damaging crops)
- Agri institutions: agriculture department, KVK, agri universities, advice to farmers
NON_AGRI = farming is absent or only mentioned in passing: politics or protests with no
farm-economy content, food and recipes, general weather, drinking water, gardening, and
everything else (sports, films, crime, business, health, education, etc.).

This is a first-pass filter: if a substantial part of the excerpt is about farming or the
farm economy, answer AGRI.

Answer with exactly one label: AGRI or NON_AGRI. No explanation.
```

### A.4 Document prompt d1: few-shot turns (4)

Each user turn is `<doc>…</doc>`; the assistant turn is the label. Gaps are marked with `…`.

| # | Excerpt | Assistant label |
|---|---|---|
| 1 | यंदा मराठवाड्यात खरीप हंगामात सोयाबीनची पेरणी मोठ्या प्रमाणात वाढली आहे. कृषी विभागाच्या अंदाजानुसार सोयाबीनखालील क्षेत्र दहा टक्क्यांनी वाढेल. … बियाण्यांचा तुटवडा जाणवत असल्याने शेतकऱ्यांनी प्रमाणित बियाणे वापरावे, असे आवाहन करण्यात आले. … पुढील आठवड्यात जिल्ह्यात चांगल्या पावसाचा अंदाज आहे. | `AGRI` |
| 2 | महापालिका निवडणुकीसाठी सर्व पक्षांनी उमेदवारांची यादी जाहीर केली आहे. सत्ताधारी पक्षाने अनेक नव्या चेहऱ्यांना संधी दिली. … प्रचार सभेत त्यांनी शेतकरी आणि कामगारांचाही उल्लेख केला. … मतदान पुढील महिन्यात होणार असून निकाल दुसऱ्या दिवशी जाहीर होतील. | `NON_AGRI` |
| 3 | जिल्हा दूध संघाने गाईच्या दुधाच्या खरेदी दरात प्रतिलिटर दोन रुपयांची वाढ केली आहे. … पशुखाद्याचे दर वाढल्याने दूध उत्पादक शेतकरी अडचणीत आले होते. … नवे दर एक तारखेपासून लागू होतील, असे संघाच्या अध्यक्षांनी सांगितले. | `AGRI` |
| 4 | हिवाळ्यात गाजराचा हलवा बनवण्यासाठी ताजे लाल गाजर किसून घ्यावे. … त्यात दूध आणि साखर घालून मंद आचेवर शिजवावे. … वरून काजू, बदाम घालून गरम गरम वाढावा. | `NON_AGRI` |
