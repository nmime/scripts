# Vast.ai Huihui GLM-5.1 Q3_K on-start script

This directory contains a Vast.ai on-start script that builds CUDA-enabled `ggml-org/llama.cpp`, downloads the pinned `huihui-ai/Huihui-GLM-5.1-abliterated-GGUF` Q3_K GGUF shards, and starts `llama-server` automatically. By default it runs directly from the split GGUF shard set, so it avoids the slow 359GB merge step and reduces temporary disk requirements.

## Recommended Vast template settings, no Splox webhook required

Set these environment variables in the Vast.ai template:

```text
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/nmime/scripts/main/vast-glm51/onstart.sh
LLAMA_API_KEY=<your-known-secret-key>
PORT=8080
CTX=4096
USE_SPLIT_MODEL=1
ARIA_CONCURRENT=8
ARIA_SPLIT=8
CUDA_ARCH=80
```

If you set `LLAMA_API_KEY` in the template, you already know the bearer token and do not need to SSH into the instance to read a generated key.

Recommended container settings:

- **Image path:** `nvidia/cuda`
- **Tag:** `12.4.1-devel-ubuntu22.04`
- **Docker options:** `--shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 -p 8080:8080`
- **Expose:** expose `8080` TCP in Vast if you do not use an explicit `-p 8080:8080` mapping.
- **Hardware:** 8x A100 80GB and at least 1TB disk; 1.5TB+ disk is preferred if you enable merged-model mode.

## On-start command

Use the Vast `PROVISIONING_SCRIPT` environment variable above when available. If your template needs an explicit on-start command instead, use:

```bash
bash -lc 'curl -fsSL https://raw.githubusercontent.com/nmime/scripts/main/vast-glm51/onstart.sh -o /root/onstart.sh && chmod +x /root/onstart.sh && exec /root/onstart.sh'
```

`READY_WEBHOOK_URL` is optional. The script no longer needs a Splox webhook to expose connection details because it writes root-only ready files locally. If you do provide `READY_WEBHOOK_URL`, the script posts `/root/huihui/ready.json` before starting `llama-server`; callback failures are logged but do not stop startup.

## Environment variables

Primary template variables:

- `WORKDIR`: working directory; defaults to `/root/huihui`.
- `MODEL_ID`: Hugging Face model repo; defaults to `huihui-ai/Huihui-GLM-5.1-abliterated-GGUF`.
- `REV`: pinned Hugging Face revision; defaults to `69821bc88f2f36680374601dfbdaf441d659920b`.
- `PORT`: llama-server listen port; defaults to `8080`.
- `CTX`: context size; defaults to `4096`.
- `LLAMA_API_KEY`: optional known bearer token. If set, the script stores it in `/root/huihui/api_key`; if unset, it generates a key. The key is never printed in logs.
- `HF_TOKEN`: optional Hugging Face token for authenticated downloads. It is used as an aria2/curl Authorization header and is never printed in logs.
- `ARIA_CONCURRENT`: aria2 concurrent downloads; defaults to `8`.
- `ARIA_SPLIT`: aria2 split count; defaults to `8`.
- `ARIA_CONN_PER_SERVER`: aria2 per-server connections; defaults to `ARIA_SPLIT`.
- `ARIA_SUMMARY_INTERVAL`: aria2 summary interval; defaults to `60`.
- `CUDA_ARCH`: CMake CUDA architecture; defaults to `80` for A100/sm80.
- `USE_SPLIT_MODEL`: defaults to `1`. Starts `llama-server` from the first split shard and skips merge.
- `MERGE_AFTER_DOWNLOAD`: defaults to `0`. Set to `1` to merge shards and serve the merged file.
- `CLEAN_RAW_AFTER_MERGE`: defaults to `0`. Set to `1` only if you want raw shards deleted after the merged model validates.
- `READY_WEBHOOK_URL`: optional callback URL; not required for normal Vast usage.

## Public URL on Vast.ai

The public URL is usually **not** `http://host:8080`. Vast maps container port `8080` to a public host port and exposes that mapping with environment variables:

- `PUBLIC_IPADDR`: public host/IP.
- `VAST_TCP_PORT_8080`: public TCP port mapped to container port `8080`.

For the default port, the chat endpoint is:

```text
http://PUBLIC_IPADDR:VAST_TCP_PORT_8080/v1/chat/completions
```

For example, if `PUBLIC_IPADDR=203.0.113.10` and `VAST_TCP_PORT_8080=41234`, use:

```text
http://203.0.113.10:41234/v1/chat/completions
```

If you set `PORT` to another value, the script dynamically reads `VAST_TCP_PORT_$PORT` to compute the public endpoint.

## Ready files and manual retrieval

The script writes:

- `/root/huihui/onstart.log`: startup logs.
- `/root/huihui/api_key`: llama-server API key, mode `600`.
- `/root/huihui/ready.json`: status, model alias, public host/port, `base_url`, `chat_url`, API key file path, API key source, and the API key for root-only convenience, mode `600`.
- `/root/huihui/ready.env`: shell exports for `LLM_BASE_URL`, `LLM_CHAT_URL`, `LLM_MODEL`, `LLM_API_KEY_FILE`, and `LLM_API_KEY`, mode `600`.

Manual SSH retrieval examples:

```bash
ssh root@<vast-public-hostname-or-ip> 'source /root/huihui/ready.env && printf "%s\n" "$LLM_CHAT_URL"'
ssh root@<vast-public-hostname-or-ip> 'cat /root/huihui/ready.json'
ssh root@<vast-public-hostname-or-ip> 'cat /root/huihui/api_key'
```

## API usage

If you set `LLAMA_API_KEY` in the Vast template, use that value as the bearer token. Otherwise retrieve it from `/root/huihui/api_key` or source `/root/huihui/ready.env` over SSH.

```bash
LLM_CHAT_URL="http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8080}/v1/chat/completions"
LLAMA_API_KEY='<your-template-LLAMA_API_KEY-or-/root/huihui/api_key>'

curl -sS "${LLM_CHAT_URL}" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "huihui-glm-5.1-q3_k",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7
  }'
```

## Speed notes

- `USE_SPLIT_MODEL=1` avoids the 359GB merge step and starts directly from the split model shard set.
- `CUDA_ARCH=80` limits CUDA compilation to A100/sm80 instead of compiling many architectures.
- The llama.cpp checkout uses a shallow partial clone, and the build targets only `llama-server`, `llama-cli`, and `llama-gguf-split`.
- aria2 download concurrency is configurable with `ARIA_CONCURRENT`, `ARIA_SPLIT`, and `ARIA_CONN_PER_SERVER`; downloads are resumable.
- For repeated rentals, a custom Docker image with prebuilt llama.cpp is faster.
- A persistent volume or Vast snapshot with the downloaded model is the fastest option because it avoids re-downloading hundreds of GB.
