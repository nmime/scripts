#!/usr/bin/env bash
set -Eeuo pipefail

WORKDIR="${WORKDIR:-/root/huihui}"
MODEL_ID="${MODEL_ID:-huihui-ai/Huihui-GLM-5.1-abliterated-GGUF}"
MODEL_REPO="${MODEL_REPO:-${MODEL_ID}}"
REV="${REV:-69821bc88f2f36680374601dfbdaf441d659920b}"
REVISION="${REVISION:-${REV}}"
PORT="${PORT:-8080}"
CTX="${CTX:-98304}"
if [[ -z "${ARIA_CONCURRENT+x}" && -n "${MAX_DOWNLOADS:-}" ]]; then
  ARIA_CONCURRENT="${MAX_DOWNLOADS}"
fi
ARIA_CONCURRENT="${ARIA_CONCURRENT:-8}"
if [[ -z "${ARIA_SPLIT+x}" && -n "${ARIA2_SPLIT:-}" ]]; then
  ARIA_SPLIT="${ARIA2_SPLIT}"
fi
ARIA_SPLIT="${ARIA_SPLIT:-8}"
if [[ -z "${ARIA_CONN_PER_SERVER+x}" && -n "${ARIA2_CONN_PER_SERVER:-}" ]]; then
  ARIA_CONN_PER_SERVER="${ARIA2_CONN_PER_SERVER}"
fi
ARIA_CONN_PER_SERVER="${ARIA_CONN_PER_SERVER:-${ARIA_SPLIT}}"
ARIA_SUMMARY_INTERVAL="${ARIA_SUMMARY_INTERVAL:-60}"
USE_SPLIT_MODEL="${USE_SPLIT_MODEL:-1}"
MERGE_AFTER_DOWNLOAD="${MERGE_AFTER_DOWNLOAD:-0}"
CLEAN_RAW_AFTER_MERGE="${CLEAN_RAW_AFTER_MERGE:-0}"
CUDA_ARCH="${CUDA_ARCH:-80}"
BUILD_JOBS="${BUILD_JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || printf '1')}"
SKIP_DOWNLOAD_IF_COMPLETE="${SKIP_DOWNLOAD_IF_COMPLETE:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1,1,1,1,1,1,1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
PARALLEL="${PARALLEL:-1}"
FLASH_ATTN="${FLASH_ATTN:-}"

MODEL_PREFIX="Q3_K-GGUF"
MODEL_PATTERN="${MODEL_PREFIX}-*.gguf"
MODEL_ALIAS="huihui-glm-5.1-q3_k"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
MIN_TOTAL_BYTES=$((300 * 1024 * 1024 * 1024))
MIN_GPU_COUNT="${MIN_GPU_COUNT:-8}"

LLAMA_DIR="${WORKDIR}/llama.cpp"
MODEL_DIR="${WORKDIR}/models/${MODEL_REPO//\//__}/${REVISION}"
MERGED_MODEL="${WORKDIR}/models/${MODEL_REPO//\//__}/${MODEL_PREFIX}.gguf"
LOG_FILE="${WORKDIR}/onstart.log"
API_KEY_FILE="${WORKDIR}/api_key"
READY_JSON="${WORKDIR}/ready.json"
READY_ENV="${WORKDIR}/ready.env"
ARIA2_INPUT="${WORKDIR}/hf-aria2-input.txt"

mkdir -p "${WORKDIR}"
touch "${LOG_FILE}"
chmod 600 "${LOG_FILE}" || true
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

sleep_forever() {
  log "$*"
  log "Sleeping forever to keep the Vast instance available for debugging."
  while true; do sleep 3600; done
}

require_root() {
  if [[ ${EUID} -ne 0 ]]; then
    sleep_forever "ERROR: this on-start script must run as root for apt and /root paths."
  fi
}

install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  local packages=(
    git ca-certificates curl jq python3 python3-pip python3-venv
    build-essential cmake ninja-build ccache aria2 openssl pciutils
  )
  local missing=()
  local pkg
  for pkg in "${packages[@]}"; do
    dpkg -s "${pkg}" >/dev/null 2>&1 || missing+=("${pkg}")
  done
  if (( ${#missing[@]} == 0 )); then
    log "Required apt packages are already installed; skipping apt install."
    return
  fi
  apt-get update
  apt-get install -y --no-install-recommends "${missing[@]}"
}

check_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    sleep_forever "ERROR: nvidia-smi not found. Use a CUDA devel image with NVIDIA runtime enabled."
  fi

  local gpu_count
  gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
  log "Detected ${gpu_count} GPU(s)."
  if (( gpu_count < MIN_GPU_COUNT )); then
    sleep_forever "ERROR: expected at least ${MIN_GPU_COUNT} GPUs for this template; detected ${gpu_count}."
  fi
}

log_config() {
  local dynamic_vast_var="VAST_TCP_PORT_${PORT}"
  local dynamic_vast_port="${!dynamic_vast_var:-}"
  local fallback_vast_port=""
  if [[ "${PORT}" == "8080" ]]; then
    fallback_vast_port="${VAST_TCP_PORT_8080:-}"
  fi

  log "Configuration: WORKDIR=${WORKDIR}"
  log "Configuration: MODEL_ID=${MODEL_REPO}"
  log "Configuration: REV=${REVISION}"
  log "Configuration: PORT=${PORT} CTX=${CTX} CUDA_ARCH=${CUDA_ARCH} BUILD_JOBS=${BUILD_JOBS}"
  log "Configuration: ARIA_CONCURRENT=${ARIA_CONCURRENT} ARIA_SPLIT=${ARIA_SPLIT} ARIA_CONN_PER_SERVER=${ARIA_CONN_PER_SERVER} ARIA_SUMMARY_INTERVAL=${ARIA_SUMMARY_INTERVAL} SKIP_DOWNLOAD_IF_COMPLETE=${SKIP_DOWNLOAD_IF_COMPLETE}"
  log "Configuration: USE_SPLIT_MODEL=${USE_SPLIT_MODEL} MERGE_AFTER_DOWNLOAD=${MERGE_AFTER_DOWNLOAD} CLEAN_RAW_AFTER_MERGE=${CLEAN_RAW_AFTER_MERGE}"
  log "Configuration: server flags N_GPU_LAYERS=${N_GPU_LAYERS} SPLIT_MODE=${SPLIT_MODE} TENSOR_SPLIT=${TENSOR_SPLIT} BATCH_SIZE=${BATCH_SIZE} UBATCH_SIZE=${UBATCH_SIZE} PARALLEL=${PARALLEL} FLASH_ATTN=${FLASH_ATTN:-<unset>}"
  log "Configuration: PUBLIC_HOST=${PUBLIC_HOST:-<unset>} PUBLIC_PORT=${PUBLIC_PORT:-<unset>} PUBLIC_IPADDR=${PUBLIC_IPADDR:-<unset>} ${dynamic_vast_var}=${dynamic_vast_port:-<unset>} VAST_TCP_PORT_8080=${fallback_vast_port:-${VAST_TCP_PORT_8080:-<unset>}}"
  log "Configuration: LLAMA_API_KEY=$([[ -n "${LLAMA_API_KEY:-}" ]] && printf '<set>' || printf '<unset>') HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && printf '<set>' || printf '<unset>') READY_WEBHOOK_URL=$([[ -n "${READY_WEBHOOK_URL:-}" ]] && printf '<set>' || printf '<unset>')"
}

llama_build_is_usable() {
  local build_dir="${LLAMA_DIR}/build"
  local cache_file="${build_dir}/CMakeCache.txt"
  local cached_arch=""

  if [[ ! -x "${build_dir}/bin/llama-server" || ! -x "${build_dir}/bin/llama-cli" || ! -x "${build_dir}/bin/llama-gguf-split" ]]; then
    return 1
  fi
  if [[ ! -f "${cache_file}" ]]; then
    log "Existing llama.cpp build is missing CMakeCache.txt; rebuild required."
    return 1
  fi
  if ! grep -Eq '^GGML_CUDA(:[^=]*)?=ON$' "${cache_file}"; then
    log "Existing llama.cpp build was not configured with GGML_CUDA=ON; rebuild required."
    return 1
  fi
  cached_arch=$(awk -F= '$1 ~ /^CMAKE_CUDA_ARCHITECTURES(:|$)/ {print $2; exit}' "${cache_file}")
  if [[ -z "${cached_arch}" ]]; then
    log "Existing llama.cpp build has no cached CMAKE_CUDA_ARCHITECTURES; rebuild required."
    return 1
  fi
  if [[ "${cached_arch}" != "${CUDA_ARCH}" ]]; then
    log "Existing llama.cpp build CUDA architecture (${cached_arch}) does not match requested CUDA_ARCH=${CUDA_ARCH}; rebuild required."
    return 1
  fi

  return 0
}

clone_or_update_llama_cpp() {
  if llama_build_is_usable; then
    log "llama.cpp build is usable for CUDA architecture ${CUDA_ARCH}; skipping clone/configure/build."
    return
  fi

  if [[ -d "${LLAMA_DIR}/.git" ]]; then
    log "Updating existing llama.cpp checkout."
    git -C "${LLAMA_DIR}" fetch --depth 1 --filter=blob:none origin
    local default_ref
    default_ref=$(git -C "${LLAMA_DIR}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)
    default_ref="${default_ref:-master}"
    git -C "${LLAMA_DIR}" checkout "${default_ref}"
    git -C "${LLAMA_DIR}" reset --hard "origin/${default_ref}"
  else
    log "Cloning llama.cpp with shallow partial clone."
    git clone --depth 1 --filter=blob:none https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
  fi

  log "Configuring llama.cpp CUDA build for CUDA architecture ${CUDA_ARCH}."
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  log "Building llama-server, llama-cli, and llama-gguf-split only with BUILD_JOBS=${BUILD_JOBS}."
  cmake --build "${LLAMA_DIR}/build" --config Release --target llama-server llama-cli llama-gguf-split --parallel "${BUILD_JOBS}"
}

hf_curl_args() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' -H "Authorization: Bearer ${HF_TOKEN}"
  fi
}

discover_shards() {
  mkdir -p "${MODEL_DIR}"
  local tmp_json="${WORKDIR}/hf-model-files.json"
  local api_url="https://huggingface.co/api/models/${MODEL_REPO}/tree/${REVISION}?recursive=1"

  log "Discovering GGUF shards from Hugging Face API."
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "${api_url}" -o "${tmp_json}"
  else
    curl -fsSL "${api_url}" -o "${tmp_json}"
  fi

  mapfile -t SHARD_RELS < <(
    jq -r '
      .[]
      | select(.type == "file")
      | .path
      | select(test("(^|/)[^/]+-[0-9]{5}-of-[0-9]{5}\\.gguf$"))
    ' "${tmp_json}" | sort -V
  )

  if [[ ${#SHARD_RELS[@]} -ne ${EXPECTED_SHARDS} ]]; then
    log "Hugging Face API discovered ${#SHARD_RELS[@]} shards; falling back to deterministic ${EXPECTED_SHARDS}-shard list."
    SHARD_RELS=()
    local i
    for i in $(seq -f "%05g" 1 "${EXPECTED_SHARDS}"); do
      SHARD_RELS+=("${MODEL_PREFIX}/${MODEL_PREFIX}-${i}-of-$(printf "%05d" "${EXPECTED_SHARDS}").gguf")
    done
  fi

  if [[ ${#SHARD_RELS[@]} -ne ${EXPECTED_SHARDS} ]]; then
    sleep_forever "ERROR: expected ${EXPECTED_SHARDS} shard names but have ${#SHARD_RELS[@]}."
  fi

  log "Using ${#SHARD_RELS[@]} shard URLs."
}

write_aria2_input() {
  : > "${ARIA2_INPUT}"
  chmod 600 "${ARIA2_INPUT}" || true
  local rel url out target_dir
  for rel in "${SHARD_RELS[@]}"; do
    url="https://huggingface.co/${MODEL_REPO}/resolve/${REVISION}/${rel}?download=true"
    out="${rel##*/}"
    if [[ "${rel}" == */* ]]; then
      target_dir="${MODEL_DIR}/${rel%/*}"
      mkdir -p "${target_dir}"
    else
      target_dir="${MODEL_DIR}"
    fi
    {
      printf '%s\n' "${url}"
      printf '  dir=%s\n' "${target_dir}"
      printf '  out=%s\n' "${out}"
      if [[ -n "${HF_TOKEN:-}" ]]; then
        printf '  header=Authorization: Bearer %s\n' "${HF_TOKEN}"
      fi
    } >> "${ARIA2_INPUT}"
  done
}

cached_shards_complete() {
  mkdir -p "${MODEL_DIR}"
  mapfile -t SHARD_FILES < <(find "${MODEL_DIR}" -type f -name "${MODEL_PATTERN}" | sort -V)
  if [[ ${#SHARD_FILES[@]} -ne ${EXPECTED_SHARDS} ]]; then
    log "Cached shard check: expected ${EXPECTED_SHARDS} shard files in ${MODEL_DIR}, found ${#SHARD_FILES[@]}."
    return 1
  fi

  local stats total_bytes zero_count
  stats=$(python3 - "${SHARD_FILES[@]}" <<'PYCACHE'
import os, sys
files = sys.argv[1:]
zeros = [p for p in files if os.path.getsize(p) <= 0]
total = sum(os.path.getsize(p) for p in files)
print(f"total={total}")
print(f"zeros={len(zeros)}")
PYCACHE
)
  total_bytes=$(awk -F= '/^total=/{print $2}' <<< "${stats}")
  zero_count=$(awk -F= '/^zeros=/{print $2}' <<< "${stats}")

  if (( zero_count > 0 )); then
    log "Cached shard check: ${zero_count} shard files are empty."
    return 1
  fi
  if (( total_bytes < MIN_TOTAL_BYTES )); then
    log "Cached shard check: shard total size ${total_bytes} bytes is below MIN_TOTAL_BYTES=${MIN_TOTAL_BYTES}."
    return 1
  fi

  log "Cached shard check passed for ${#SHARD_FILES[@]} shards; total size ${total_bytes} bytes."
  return 0
}

download_shards() {
  if [[ "${SKIP_DOWNLOAD_IF_COMPLETE}" == "1" ]] && cached_shards_complete; then
    log "SKIP_DOWNLOAD_IF_COMPLETE=1 and cached shards are complete; skipping Hugging Face API discovery and aria2 download."
    return
  fi

  discover_shards
  write_aria2_input
  log "Downloading/resuming shards with aria2: concurrent=${ARIA_CONCURRENT}, split=${ARIA_SPLIT}, connections/server=${ARIA_CONN_PER_SERVER}."
  aria2c \
    --input-file="${ARIA2_INPUT}" \
    --max-concurrent-downloads="${ARIA_CONCURRENT}" \
    --split="${ARIA_SPLIT}" \
    --max-connection-per-server="${ARIA_CONN_PER_SERVER}" \
    --continue=true \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval="${ARIA_SUMMARY_INTERVAL}" \
    --console-log-level=warn
}

validate_shards() {
  if ! cached_shards_complete; then
    sleep_forever "ERROR: shard cache is incomplete after download; see cached shard check above."
  fi
  log "Validated ${#SHARD_FILES[@]} shards."
}

merge_model() {
  if [[ -s "${MERGED_MODEL}" ]]; then
    log "Merged model already exists at ${MERGED_MODEL}; validating."
    "${LLAMA_DIR}/build/bin/llama-cli" --model "${MERGED_MODEL}" --no-warmup --n-predict 1 --prompt "ping" >/tmp/huihui-validate.log 2>&1 || {
      cat /tmp/huihui-validate.log >&2 || true
      sleep_forever "ERROR: existing merged model validation failed."
    }
    return
  fi

  mkdir -p "$(dirname "${MERGED_MODEL}")"
  log "Merging ${EXPECTED_SHARDS} shards into ${MERGED_MODEL}. This is skipped when USE_SPLIT_MODEL=1 and MERGE_AFTER_DOWNLOAD=0."
  "${LLAMA_DIR}/build/bin/llama-gguf-split" --merge "${SHARD_FILES[0]}" "${MERGED_MODEL}"
  log "Validating merged model metadata."
  "${LLAMA_DIR}/build/bin/llama-cli" --model "${MERGED_MODEL}" --no-warmup --n-predict 1 --prompt "ping" >/tmp/huihui-validate.log 2>&1 || {
    cat /tmp/huihui-validate.log >&2 || true
    sleep_forever "ERROR: merged model validation failed."
  }
  if [[ "${CLEAN_RAW_AFTER_MERGE}" == "1" ]]; then
    log "CLEAN_RAW_AFTER_MERGE=1; removing raw shards after successful merged model validation."
    rm -f -- "${SHARD_FILES[@]}"
  fi
}

prepare_api_key() {
  if [[ -n "${LLAMA_API_KEY:-}" ]]; then
    umask 077
    printf '%s' "${LLAMA_API_KEY}" > "${API_KEY_FILE}"
    API_KEY_SOURCE="env"
  elif [[ ! -s "${API_KEY_FILE}" ]]; then
    umask 077
    openssl rand -hex 32 > "${API_KEY_FILE}"
    API_KEY_SOURCE="generated"
  else
    chmod 600 "${API_KEY_FILE}" || true
    API_KEY_SOURCE="existing_file"
  fi
  chmod 600 "${API_KEY_FILE}" || true
}

compute_public_endpoint() {
  PUBLIC_HOST="${PUBLIC_HOST:-${PUBLIC_IPADDR:-}}"
  local dynamic_vast_var="VAST_TCP_PORT_${PORT}"
  PUBLIC_PORT="${PUBLIC_PORT:-}"
  if [[ -z "${PUBLIC_PORT}" ]]; then
    PUBLIC_PORT="${!dynamic_vast_var:-}"
  fi
  if [[ -z "${PUBLIC_PORT}" && "${PORT}" == "8080" ]]; then
    PUBLIC_PORT="${VAST_TCP_PORT_8080:-}"
  fi

  BASE_URL=""
  CHAT_URL=""
  if [[ -n "${PUBLIC_HOST}" && -n "${PUBLIC_PORT}" ]]; then
    BASE_URL="http://${PUBLIC_HOST}:${PUBLIC_PORT}"
    CHAT_URL="${BASE_URL}/v1/chat/completions"
  fi
}

write_ready_files() {
  prepare_api_key
  compute_public_endpoint
  local now api_key
  now=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  api_key=$(<"${API_KEY_FILE}")

  umask 077
  jq -n \
    --arg status "starting" \
    --arg timestamp "${now}" \
    --arg model "${MODEL_ALIAS}" \
    --arg model_id "${MODEL_REPO}" \
    --arg revision "${REVISION}" \
    --arg public_host "${PUBLIC_HOST}" \
    --arg public_port "${PUBLIC_PORT}" \
    --arg base_url "${BASE_URL}" \
    --arg chat_url "${CHAT_URL}" \
    --arg api_key_file "${API_KEY_FILE}" \
    --arg api_key_source "${API_KEY_SOURCE}" \
    --arg api_key "${api_key}" \
    --arg port "${PORT}" \
    --arg ctx "${CTX}" \
    --arg use_split_model "${USE_SPLIT_MODEL}" \
    --arg merge_after_download "${MERGE_AFTER_DOWNLOAD}" \
    '{
      status: $status,
      timestamp: $timestamp,
      model: $model,
      model_id: $model_id,
      revision: $revision,
      public_host: $public_host,
      public_port: $public_port,
      base_url: $base_url,
      chat_url: $chat_url,
      api_key_file: $api_key_file,
      api_key_source: $api_key_source,
      api_key: $api_key,
      port: ($port | tonumber),
      ctx: ($ctx | tonumber),
      use_split_model: ($use_split_model == "1"),
      merge_after_download: ($merge_after_download == "1")
    }' > "${READY_JSON}"
  chmod 600 "${READY_JSON}"

  cat > "${READY_ENV}" <<EOF_READY_ENV
export LLM_BASE_URL=$(printf '%q' "${BASE_URL}")
export LLM_CHAT_URL=$(printf '%q' "${CHAT_URL}")
export LLM_MODEL=$(printf '%q' "${MODEL_ALIAS}")
export LLM_API_KEY_FILE=$(printf '%q' "${API_KEY_FILE}")
export LLM_API_KEY=$(printf '%q' "${api_key}")
EOF_READY_ENV
  chmod 600 "${READY_ENV}"

  log "Wrote ready metadata to ${READY_JSON} and ${READY_ENV}; API key stored at ${API_KEY_FILE}."
  if [[ -n "${BASE_URL}" ]]; then
    log "Public base URL: ${BASE_URL}"
    log "Public chat URL: ${CHAT_URL}"
  else
    log "Public base URL unavailable; PUBLIC_IPADDR and/or VAST_TCP_PORT_${PORT} are not set."
  fi
}

post_ready_webhook() {
  if [[ -z "${READY_WEBHOOK_URL:-}" ]]; then
    log "READY_WEBHOOK_URL is unset; skipping optional webhook."
    return
  fi

  log "Posting ready metadata to optional webhook."
  if ! curl -fsSL -X POST \
    -H 'Content-Type: application/json' \
    --data-binary "@${READY_JSON}" \
    "${READY_WEBHOOK_URL}" >/tmp/huihui-ready-webhook.out 2>/tmp/huihui-ready-webhook.err; then
    log "WARNING: READY_WEBHOOK_URL callback failed; continuing startup."
    cat /tmp/huihui-ready-webhook.err >&2 || true
  fi
}

choose_model_path() {
  SERVER_MODEL="${SHARD_FILES[0]}"
  if [[ "${USE_SPLIT_MODEL}" == "1" && "${MERGE_AFTER_DOWNLOAD}" != "1" ]]; then
    log "USE_SPLIT_MODEL=1; using first split shard directly and skipping merged-model creation."
    return
  fi

  merge_model
  SERVER_MODEL="${MERGED_MODEL}"
}

add_flash_attn_arg_if_requested() {
  if [[ -z "${FLASH_ATTN}" ]]; then
    return
  fi

  local server_help flash_attn_lower
  server_help=$("${LLAMA_DIR}/build/bin/llama-server" --help 2>&1 || true)
  if ! grep -q -- '--flash-attn' <<< "${server_help}"; then
    log "WARNING: FLASH_ATTN=${FLASH_ATTN} requested, but this llama-server does not advertise --flash-attn; ignoring."
    return
  fi

  flash_attn_lower="${FLASH_ATTN,,}"
  case "${flash_attn_lower}" in
    1|true|yes|on|enabled)
      SERVER_ARGS+=(--flash-attn)
      ;;
    0|false|no|off|disabled)
      if grep -q -- '--no-flash-attn' <<< "${server_help}"; then
        SERVER_ARGS+=(--no-flash-attn)
      else
        log "WARNING: FLASH_ATTN=${FLASH_ATTN} requested, but this llama-server does not advertise --no-flash-attn; ignoring."
      fi
      ;;
    *)
      log "WARNING: FLASH_ATTN=${FLASH_ATTN} is not a recognized boolean value; ignoring to avoid passing an unsafe llama-server flag."
      ;;
  esac
}

start_server() {
  choose_model_path
  write_ready_files
  post_ready_webhook
  log "Starting llama-server on 0.0.0.0:${PORT} with alias ${MODEL_ALIAS}. API key is stored at ${API_KEY_FILE}."
  SERVER_ARGS=(
    "${LLAMA_DIR}/build/bin/llama-server"
    --host 0.0.0.0
    --port "${PORT}"
    --model "${SERVER_MODEL}"
    --alias "${MODEL_ALIAS}"
    --api-key-file "${API_KEY_FILE}"
    --ctx-size "${CTX}"
    --n-gpu-layers "${N_GPU_LAYERS}"
    --split-mode "${SPLIT_MODE}"
    --tensor-split "${TENSOR_SPLIT}"
    --batch-size "${BATCH_SIZE}"
    --ubatch-size "${UBATCH_SIZE}"
    --parallel "${PARALLEL}"
    --cont-batching
  )
  add_flash_attn_arg_if_requested
  exec "${SERVER_ARGS[@]}"
}

main() {
  require_root
  log_config
  install_packages
  check_gpus
  clone_or_update_llama_cpp
  download_shards
  validate_shards
  start_server
}

main "$@"
