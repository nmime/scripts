# Vast.ai Huihui GLM-5.1 Q3_K on-start script

This directory contains a Vast.ai on-start script that builds the latest CUDA-enabled `ggml-org/llama.cpp`, downloads the pinned `huihui-ai/Huihui-GLM-5.1-abliterated-GGUF` Q3_K GGUF shards, merges them, and starts `llama-server`.

## Vast template settings

- **Image path:** `nvidia/cuda`
- **Tag:** `12.4.1-devel-ubuntu22.04`
- **Docker options:** `--shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864`
- **Expose:** `8080` TCP
- **Hardware:** 8x A100 80GB and at least 1TB disk; 1.5TB+ disk is preferred for safer shard download, merge, and build headroom.

## On-start command

```bash
bash -lc 'curl -fsSL https://raw.githubusercontent.com/nmime/scripts/main/vast-glm51/onstart.sh -o /root/onstart.sh && chmod +x /root/onstart.sh && exec /root/onstart.sh'
```

For fully automatic Splox/chat callback mode, set `READY_WEBHOOK_URL` before running the script in the Vast on-start command:

```bash
bash -lc 'export READY_WEBHOOK_URL="https://splox.io/api/v1/events/019ea682-330c-73f0-aae7-dee23445e1f4"; curl -fsSL https://raw.githubusercontent.com/nmime/scripts/main/vast-glm51/onstart.sh -o /root/onstart.sh && chmod +x /root/onstart.sh && exec /root/onstart.sh'
```

The script writes logs to `/root/huihui/onstart.log`, stores or generates the API key at `/root/huihui/api_key` with mode `600`, writes startup endpoint metadata to `/root/huihui/ready.json` with mode `600`, and starts `llama-server` on container port `8080` unless `PORT` is set.

Optional environment variables:

- `HF_TOKEN`: Hugging Face token for authenticated downloads.
- `LLAMA_API_KEY`: pre-set server bearer token; if unset, the script generates one.
- `WORKDIR`: working directory; defaults to `/root/huihui`.
- `PORT`: llama-server listen port; defaults to `8080`.
- `CTX`: context size; defaults to `4096`.
- `READY_WEBHOOK_URL`: optional callback URL. When set, the script POSTs `/root/huihui/ready.json` after the merged model validates and before `llama-server` is exec'd. Callback failures are logged but do not stop startup.

## Automatic endpoint metadata and callback

Vast exposes the public endpoint mapping for container port `8080` with these environment variables:

- `PUBLIC_IPADDR`: public host/IP.
- `VAST_TCP_PORT_8080`: public TCP port mapped to container port `8080`.

When both are present, the script writes `base_url` as `http://$PUBLIC_IPADDR:$VAST_TCP_PORT_8080` and `chat_url` as `http://$PUBLIC_IPADDR:$VAST_TCP_PORT_8080/v1/chat/completions` in `/root/huihui/ready.json`. The status is `starting` because the callback is sent immediately before `exec llama-server`.

Retrieve the generated metadata and API key manually over SSH:

```bash
ssh root@<vast-public-hostname-or-ip> 'cat /root/huihui/ready.json'
ssh root@<vast-public-hostname-or-ip> 'cat /root/huihui/api_key'
```

Or copy them locally:

```bash
scp root@<vast-public-hostname-or-ip>:/root/huihui/ready.json ./ready.json
scp root@<vast-public-hostname-or-ip>:/root/huihui/api_key ./api_key
```

## API usage

After startup, read the bearer key from the instance:

```bash
cat /root/huihui/api_key
```

Call the OpenAI-compatible endpoint through the Vast.ai public port that maps to container TCP `8080`. In fully automatic mode this is `http://$PUBLIC_IPADDR:$VAST_TCP_PORT_8080/v1/chat/completions`:

```bash
VAST_HOST="${PUBLIC_IPADDR}"
VAST_PORT="${VAST_TCP_PORT_8080}"
LLAMA_API_KEY='<contents-of-/root/huihui/api_key>'

curl -sS "http://${VAST_HOST}:${VAST_PORT}/v1/chat/completions" \
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

In the Vast.ai UI/API, make sure the public port mapping points to container port `8080`.
