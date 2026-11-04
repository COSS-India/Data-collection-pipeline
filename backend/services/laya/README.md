# laya-multilingual service

Zero-shot `agri` / `agri_subdomain` classification with [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), batched for corpus-scale runs.

## Deploy (GPU box)

```bash
mkdir -p ~/data-collection-pipeline/laya && cd ~/data-collection-pipeline/laya
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python "laya==0.3.20" "transformers==5.17.0" fastapi "uvicorn>=0.30"
# copy server.py here, then:
tmux new -d -s dcp-laya "bash -c 'cd ~/data-collection-pipeline/laya && .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8160 2>&1 | tee -a server.log'"
curl localhost:8160/health
```

Env: `LAYA_BATCH_SIZE` (default 64), `LAYA_AGRI_THRESHOLD` (default 0.7).

## Use from the pipeline

```bash
ssh -N -L 8160:127.0.0.1:8160 <user>@<gpu-box> &      # tunnel
DCP_LAYA_URL=http://127.0.0.1:8160 uv run dcp serve   # or: dcp run ...
```

With `DCP_LAYA_URL` set, `laya_agri` and `laya_agri_subdomain` replace the keyword baseline for those tasks.

## Notes

- Throughput on the DGX Spark (GB10): about 94 sentences/s for both tasks together, so 500k sentences take roughly 1.5 h per task.
- Wording matters a lot. Keep label descriptions short (see the comment in `server.py`) and re-check accuracy in the evaluation UI after any change.
