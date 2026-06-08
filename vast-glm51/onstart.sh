#!/usr/bin/env bash
set -Eeuo pipefail

WORKDIR="${WORKDIR:-/root/huihui}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
MODEL_REPO="${MODEL_REPO:-huihui-ai/Huihui-GLM-5.1-abliterated-GGUF}"
REVISION="${REVISION:-69821bc88f2f36680374601dfbdaf441d659920b}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Q3_K-GGUF}"
MODEL_ALIAS="${MODEL_ALIAS:-huihui-glm-5.1-q3_k}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
MIN_GPU_COUNT="${MIN_GPU_COUNT:-8}"
MIN_GPU_MEM_MIB="${MIN_GPU_MEM_MIB:-80000}"
MIN_ROOT_FREE_GB="${MIN_ROOT_FREE_GB:-900}"
MIN_MERGED_BYTES="${MIN_MERGED_BYTES:-322122547200}" # 300 GiB

LOG_FILE="${WORKDIR}/onstart.log"
LLAMA_DIR="${WORKDIR}/llama.cpp"
SHARDS_DIR="${WORKDIR}/shards"
MERGED_DIR="${WORKDIR}/merged"
MERGED_MODEL="${MERGED_DIR}/huihui-glm-5.1-abliterated.Q3_K.gguf"
API_KEY_FILE="${WORKDIR}/api_key"
READY_JSON="${WORKDIR}/ready.json"
ARIA2_INPUT="${WORKDIR}/hf-aria2-input.txt"
HF_FILE_LIST="${WORKDIR}/hf-files.txt"

mkdir -p "${WORKDIR}"
touch "${LOG_FILE}"
chmod 600 "${LOG_FILE}" || true
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

sleep_forever() {
  log "$*"
  log "Sleeping forever so the Vast.ai instance remains inspectable. See ${LOG_FILE}."
  sleep infinity
}

on_error() {
  local rc=$?
  local line=${BASH_LINENO[0]:-${LINENO}}
  log "ERROR: command failed near line ${line} with exit code ${rc}."
  sleep_forever "Startup did not complete."
}
trap on_error ERR

require_root() {
  if [[ ${EUID} -ne 0 ]]; then
    sleep_forever "ERROR: this on-start script must run as root for apt and /root paths."
  fi
}

verify_gpus() {
  log "Checking GPU count and memory before setup."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    sleep_forever "ERROR: nvidia-smi is not available; cannot verify GPUs."
  fi

  mapfile -t gpu_memories < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print int($1)}')
  local gpu_count=${#gpu_memories[@]}
  if (( gpu_count < MIN_GPU_COUNT )); then
    sleep_forever "ERROR: found ${gpu_count} GPU(s); need at least ${MIN_GPU_COUNT}."
  fi

  local min_mem=999999999
  local mem
  for mem in "${gpu_memories[@]}"; do
    if (( mem < min_mem )); then
      min_mem=${mem}
    fi
  done

  if (( min_mem < MIN_GPU_MEM_MIB )); then
    sleep_forever "ERROR: minimum GPU memory is ${min_mem} MiB; need at least ${MIN_GPU_MEM_MIB} MiB per GPU."
  fi
  log "GPU check passed: ${gpu_count} GPU(s), minimum memory ${min_mem} MiB."
}

verify_root_disk() {
  log "Checking free disk under /root before model download."
  local free_mib required_mib
  free_mib=$(df -Pm /root | awk 'NR==2 {print $4}')
  required_mib=$(( MIN_ROOT_FREE_GB * 1024 ))
  if (( free_mib < required_mib )); then
    sleep_forever "ERROR: /root has ${free_mib} MiB free; need at least ${required_mib} MiB (${MIN_ROOT_FREE_GB} GB)."
  fi
  log "Disk check passed: /root has ${free_mib} MiB free."
}

install_dependencies() {
  log "Installing build and download dependencies."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    python3 \
    python3-pip \
    curl \
    wget \
    jq \
    aria2 \
    ca-certificates \
    openssl \
    libcurl4-openssl-dev \
    git-lfs
  git lfs install --skip-smudge --system || git lfs install --skip-smudge
}

clone_or_update_llama_cpp() {
  log "Cloning/updating latest ggml-org/llama.cpp."
  if [[ -d "${LLAMA_DIR}/.git" ]]; then
    git -C "${LLAMA_DIR}" fetch --depth 1 origin
    git -C "${LLAMA_DIR}" remote set-head origin -a || true
    local default_ref
    default_ref=$(git -C "${LLAMA_DIR}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)
    default_ref="${default_ref:-master}"
    git -C "${LLAMA_DIR}" reset --hard "origin/${default_ref}"
    git -C "${LLAMA_DIR}" clean -fdx
  else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
  fi
}

build_llama_cpp() {
  log "Building llama.cpp with CUDA and curl support."
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  cmake --build "${LLAMA_DIR}/build" --config Release --target llama-server llama-cli llama-gguf-split --parallel "$(nproc)"
  test -x "${LLAMA_DIR}/build/bin/llama-server"
  test -x "${LLAMA_DIR}/build/bin/llama-cli"
  test -x "${LLAMA_DIR}/build/bin/llama-gguf-split"
}

hf_curl_args() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' -H "Authorization: Bearer ${HF_TOKEN}"
  fi
}

create_hf_file_list() {
  log "Fetching Hugging Face file list for ${MODEL_REPO} at pinned revision ${REVISION}."
  local api_url="https://huggingface.co/api/models/${MODEL_REPO}/revision/${REVISION}"
  local tmp_json="${WORKDIR}/hf-model.json"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "${api_url}" -o "${tmp_json}"
  else
    curl -fsSL "${api_url}" -o "${tmp_json}"
  fi

  jq -r --arg dir "${MODEL_SUBDIR}" '.siblings[].rfilename | select(startswith($dir + "/")) | select(endswith(".gguf"))' "${tmp_json}" | sort > "${HF_FILE_LIST}"
  rm -f "${tmp_json}"

  local count
  count=$(wc -l < "${HF_FILE_LIST}" | tr -d ' ')
  if [[ "${count}" != "${EXPECTED_SHARDS}" ]]; then
    log "Discovered files:"
    sed 's/^/  /' "${HF_FILE_LIST}"
    sleep_forever "ERROR: discovered ${count} GGUF shard(s), expected exactly ${EXPECTED_SHARDS}."
  fi
  log "Discovered exactly ${count} GGUF shard(s)."
}

write_aria2_input() {
  mkdir -p "${SHARDS_DIR}"
  : > "${ARIA2_INPUT}"
  chmod 600 "${ARIA2_INPUT}"

  local rel filename url
  while IFS= read -r rel; do
    filename=$(basename "${rel}")
    url="https://huggingface.co/${MODEL_REPO}/resolve/${REVISION}/${rel}?download=true"
    {
      printf '%s\n' "${url}"
      printf '  dir=%s\n' "${SHARDS_DIR}"
      printf '  out=%s\n' "${filename}"
      if [[ -n "${HF_TOKEN:-}" ]]; then
        printf '  header=Authorization: Bearer %s\n' "${HF_TOKEN}"
      fi
    } >> "${ARIA2_INPUT}"
  done < "${HF_FILE_LIST}"
}

download_shards() {
  if [[ -s "${MERGED_MODEL}" ]]; then
    local existing_size
    existing_size=$(stat -c '%s' "${MERGED_MODEL}")
    if (( existing_size > MIN_MERGED_BYTES )); then
      log "Merged model already exists and validates (${existing_size} bytes); skipping shard download."
      return
    fi
  fi

  verify_root_disk
  create_hf_file_list
  write_aria2_input

  log "Downloading ${EXPECTED_SHARDS} GGUF shards with aria2 resume enabled."
  aria2c \
    --input-file="${ARIA2_INPUT}" \
    --continue=true \
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=64M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --retry-wait=10 \
    --max-tries=0 \
    --summary-interval=60 \
    --console-log-level=warn

  local shard_count
  shard_count=$(find "${SHARDS_DIR}" -maxdepth 1 -type f -name '*.gguf' | wc -l | tr -d ' ')
  if [[ "${shard_count}" != "${EXPECTED_SHARDS}" ]]; then
    sleep_forever "ERROR: downloaded ${shard_count} GGUF shard(s), expected exactly ${EXPECTED_SHARDS}."
  fi
  log "Downloaded exactly ${shard_count} GGUF shard(s)."
}

merge_and_validate_model() {
  mkdir -p "${MERGED_DIR}"
  if [[ -s "${MERGED_MODEL}" ]]; then
    local existing_size
    existing_size=$(stat -c '%s' "${MERGED_MODEL}")
    if (( existing_size > MIN_MERGED_BYTES )); then
      log "Merged model already validates at ${MERGED_MODEL} (${existing_size} bytes)."
      return
    fi
    log "Existing merged model is too small (${existing_size} bytes); replacing it."
    rm -f "${MERGED_MODEL}"
  fi

  mapfile -t shard_files < <(find "${SHARDS_DIR}" -maxdepth 1 -type f -name '*.gguf' | sort)
  if [[ "${#shard_files[@]}" != "${EXPECTED_SHARDS}" ]]; then
    sleep_forever "ERROR: found ${#shard_files[@]} local shard(s), expected exactly ${EXPECTED_SHARDS} before merge."
  fi

  log "Merging shards into ${MERGED_MODEL}."
  "${LLAMA_DIR}/build/bin/llama-gguf-split" --merge "${shard_files[0]}" "${MERGED_MODEL}"

  local merged_size
  merged_size=$(stat -c '%s' "${MERGED_MODEL}")
  if (( merged_size <= MIN_MERGED_BYTES )); then
    sleep_forever "ERROR: merged model is ${merged_size} bytes; expected more than ${MIN_MERGED_BYTES} bytes. Raw shards were kept."
  fi

  log "Merged model validates (${merged_size} bytes). Removing raw shards to save disk."
  rm -f "${shard_files[@]}"
  rm -f "${ARIA2_INPUT}" "${HF_FILE_LIST}"
  rmdir "${SHARDS_DIR}" 2>/dev/null || true
}

prepare_api_key() {
  if [[ -n "${LLAMA_API_KEY:-}" ]]; then
    umask 077
    printf '%s' "${LLAMA_API_KEY}" > "${API_KEY_FILE}"
  elif [[ ! -s "${API_KEY_FILE}" ]]; then
    umask 077
    openssl rand -hex 32 > "${API_KEY_FILE}"
  fi
  chmod 600 "${API_KEY_FILE}"
}

write_ready_json() {
  local public_host="${PUBLIC_IPADDR:-}"
  local public_port="${VAST_TCP_PORT_8080:-}"
  local base_url=""
  local chat_url=""
  local timestamp
  local api_key
  local tmp_file

  if [[ -n "${public_host}" && -n "${public_port}" ]]; then
    base_url="http://${public_host}:${public_port}"
    chat_url="${base_url}/v1/chat/completions"
  fi

  timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  api_key=$(tr -d '\r\n' < "${API_KEY_FILE}")

  umask 077
  tmp_file=$(mktemp "${WORKDIR}/ready.json.tmp.XXXXXX")
  jq -n \
    --arg status "starting" \
    --arg timestamp "${timestamp}" \
    --arg model_alias "${MODEL_ALIAS}" \
    --arg public_host "${public_host}" \
    --arg public_port "${public_port}" \
    --arg base_url "${base_url}" \
    --arg chat_url "${chat_url}" \
    --arg api_key_file "${API_KEY_FILE}" \
    --arg api_key "${api_key}" \
    --arg workdir "${WORKDIR}" \
    --arg listen_host "0.0.0.0" \
    --arg listen_port "${PORT}" \
    '{
      status: $status,
      timestamp: $timestamp,
      model_alias: $model_alias,
      public_host: $public_host,
      public_port: $public_port,
      base_url: $base_url,
      chat_url: $chat_url,
      api_key_file: $api_key_file,
      api_key: $api_key,
      workdir: $workdir,
      listen_host: $listen_host,
      listen_port: $listen_port,
      public_endpoint_env_mapping: {
        public_host: "PUBLIC_IPADDR",
        public_port_for_container_8080: "VAST_TCP_PORT_8080",
        base_url: "http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8080}",
        chat_url: "http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_8080}/v1/chat/completions"
      }
    }' > "${tmp_file}"
  chmod 600 "${tmp_file}"
  mv "${tmp_file}" "${READY_JSON}"
  chmod 600 "${READY_JSON}"
  log "Wrote startup endpoint metadata to ${READY_JSON} with status starting. Public endpoint maps PUBLIC_IPADDR and VAST_TCP_PORT_8080 to container 8080."
}

post_ready_webhook() {
  if [[ -z "${READY_WEBHOOK_URL:-}" ]]; then
    return
  fi

  local http_code=""
  local curl_rc=0
  set +e
  http_code=$(curl -sS -o /dev/null -w '%{http_code}' \
    -X POST \
    -H 'Content-Type: application/json' \
    --data-binary "@${READY_JSON}" \
    "${READY_WEBHOOK_URL}" 2>/dev/null)
  curl_rc=$?
  set -e

  if [[ ${curl_rc} -eq 0 && "${http_code}" == 2* ]]; then
    log "READY_WEBHOOK_URL callback succeeded with HTTP ${http_code}."
  else
    log "READY_WEBHOOK_URL callback failed with curl exit ${curl_rc}, HTTP ${http_code:-000}; continuing startup."
  fi
}

start_server() {
  prepare_api_key
  write_ready_json
  post_ready_webhook
  log "Starting llama-server on 0.0.0.0:${PORT} with alias ${MODEL_ALIAS}. API key is stored at ${API_KEY_FILE}."
  exec "${LLAMA_DIR}/build/bin/llama-server" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MERGED_MODEL}" \
    --alias "${MODEL_ALIAS}" \
    --api-key-file "${API_KEY_FILE}" \
    --ctx-size "${CTX}" \
    --n-gpu-layers -1 \
    --split-mode layer \
    --tensor-split 1,1,1,1,1,1,1,1 \
    --batch-size 1024 \
    --ubatch-size 256 \
    --parallel 1 \
    --cont-batching
}

main() {
  log "Starting Vast.ai Huihui GLM-5.1 Q3_K deployment."
  require_root
  verify_gpus
  install_dependencies
  clone_or_update_llama_cpp
  build_llama_cpp
  download_shards
  merge_and_validate_model
  start_server
}

main "$@"
