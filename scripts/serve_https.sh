#!/usr/bin/env bash
# serve_https.sh — serve the AgroVision FastAPI server (and /mobile PWA) over
# HTTPS so that a phone's Service Worker will actually register.
#
# Why this exists: browsers (Chrome/Android in particular) refuse to
# register a Service Worker on a plain http:// origin unless it is
# "localhost". Testing the offline PWA on a real phone over the LAN
# (http://192.168.x.x:8000/mobile/) means the SW silently never registers,
# so IndexedDB caching / offline mode look "broken" even though the app
# code is fine. This script gives you two ways to fix that:
#
#   A) mkcert  — issue a locally-trusted TLS cert for "localhost" + your
#                LAN IP, and run uvicorn with --ssl-keyfile/--ssl-certfile.
#                Best for repeated on-site testing; requires installing
#                mkcert's root CA on the phone once (instructions below).
#   B) tunnel  — punch a public HTTPS URL to your local server via
#                cloudflared (or ngrok, if installed). Zero setup on the
#                phone, but the URL is different every run (unless you
#                have a paid/named tunnel) and traffic leaves your LAN.
#
# Usage:
#   scripts/serve_https.sh mkcert   [--port 8000]
#   scripts/serve_https.sh tunnel   [--port 8000]
#   scripts/serve_https.sh          # interactive: asks which mode to use
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT=8000
MODE="${1:-}"
if [[ "$MODE" == "--port" ]]; then MODE=""; fi

# Allow --port N anywhere in the args.
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--port" ]]; then
    PORT="${args[$((i+1))]:-8000}"
  fi
done

VENV_PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="$(command -v python3)"
fi

detect_lan_ip() {
  # macOS
  if command -v ipconfig >/dev/null 2>&1 && ipconfig getifaddr en0 >/dev/null 2>&1; then
    ipconfig getifaddr en0
    return
  fi
  if command -v ipconfig >/dev/null 2>&1 && ipconfig getifaddr en1 >/dev/null 2>&1; then
    ipconfig getifaddr en1
    return
  fi
  # Linux fallback
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}' && return
  fi
  echo "127.0.0.1"
}

LAN_IP="$(detect_lan_ip)"

print_android_cert_instructions() {
  cat <<EOF

--------------------------------------------------------------------------
Установка корневого сертификата mkcert на Android (один раз на устройство)
--------------------------------------------------------------------------
1. На компьютере узнайте путь к корневому CA mkcert:
     mkcert -CAROOT
   В этой папке лежит файл rootCA.pem.

2. Перекиньте rootCA.pem на телефон любым способом (AirDrop-аналог, USB,
   Google Drive, отправить себе в Telegram и т.п.) и переименуйте
   его в rootCA.crt (Android определяет тип файла по расширению).

3. На Android: Настройки → Безопасность (Security) → Шифрование и
   учётные данные (Encryption & credentials) → Установить сертификат
   (Install a certificate) → сертификат ЦС (CA certificate) → выберите
   rootCA.crt → подтвердите предупреждение "Небезопасно" (это ожидаемо
   для локального dev-сертификата).

4. На Android 11+ система может потребовать сначала задать PIN/пароль
   экрана блокировки — это нормально, сертификат ставится в
   пользовательское хранилище (Settings → Security → Trusted
   credentials → USER tab, чтобы проверить, что он появился).

5. Откройте в браузере телефона:
     https://${LAN_IP}:${PORT}/mobile/
   Замок должен быть зелёным/закрытым, без предупреждений — теперь
   Service Worker сможет зарегистрироваться и офлайн-кэш заработает.

Убедитесь, что телефон и компьютер в одной Wi-Fi сети, и что файрвол
компьютера не блокирует входящие подключения на порт ${PORT}.
--------------------------------------------------------------------------
EOF
}

run_mkcert() {
  if ! command -v mkcert >/dev/null 2>&1; then
    echo "mkcert не установлен." >&2
    echo "Установка на macOS:  brew install mkcert nss" >&2
    echo "Установка на Linux:  см. https://github.com/FiloSottile/mkcert#installation" >&2
    exit 1
  fi

  mkcert -install >/dev/null 2>&1 || true

  CERT_DIR="$ROOT_DIR/.certs"
  mkdir -p "$CERT_DIR"
  CERT_FILE="$CERT_DIR/agrovision-dev.pem"
  KEY_FILE="$CERT_DIR/agrovision-dev-key.pem"

  echo "Генерация локального сертификата для: localhost, 127.0.0.1, ${LAN_IP}"
  mkcert -cert-file "$CERT_FILE" -key-file "$KEY_FILE" localhost 127.0.0.1 "$LAN_IP"

  print_android_cert_instructions

  echo "Запуск uvicorn с TLS на https://0.0.0.0:${PORT} ..."
  echo "  Локально:     https://localhost:${PORT}/mobile/"
  echo "  С телефона:   https://${LAN_IP}:${PORT}/mobile/"
  echo ""
  exec "$VENV_PY" -m uvicorn case1.server.app:app \
    --host 0.0.0.0 --port "$PORT" \
    --ssl-keyfile "$KEY_FILE" --ssl-certfile "$CERT_FILE"
}

run_tunnel() {
  # Start the plain-HTTP server in the background, then expose it via a
  # public HTTPS tunnel (cloudflared preferred, ngrok as fallback).
  echo "Запуск локального сервера на http://127.0.0.1:${PORT} (в фоне) ..."
  "$VENV_PY" -m uvicorn case1.server.app:app --host 127.0.0.1 --port "$PORT" &
  SERVER_PID=$!
  trap 'echo "Остановка сервера (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true' EXIT
  sleep 2

  if command -v cloudflared >/dev/null 2>&1; then
    echo "Туннель через cloudflared (Cloudflare Quick Tunnel) ..."
    echo "Дождитесь строки вида 'https://<random>.trycloudflare.com' и откройте <url>/mobile/ на телефоне."
    cloudflared tunnel --url "http://127.0.0.1:${PORT}"
  elif command -v ngrok >/dev/null 2>&1; then
    echo "Туннель через ngrok ..."
    echo "Откройте выданный https://*.ngrok-free.app URL + /mobile/ на телефоне."
    ngrok http "$PORT"
  else
    echo "Не найден ни cloudflared, ни ngrok." >&2
    echo "Установка (macOS):  brew install cloudflared   (или: brew install ngrok/ngrok/ngrok)" >&2
    exit 1
  fi
}

case "$MODE" in
  mkcert)
    run_mkcert
    ;;
  tunnel)
    run_tunnel
    ;;
  "")
    echo "Выберите режим HTTPS для тестирования PWA на телефоне:"
    echo "  1) mkcert  — локальный доверенный сертификат (рекомендуется для повторных тестов в поле)"
    echo "  2) tunnel  — публичный HTTPS через cloudflared/ngrok (без установки сертификата на телефон)"
    read -r -p "Ваш выбор [1/2]: " choice
    case "$choice" in
      1) run_mkcert ;;
      2) run_tunnel ;;
      *) echo "Отмена."; exit 1 ;;
    esac
    ;;
  *)
    echo "Неизвестный режим: $MODE (используйте 'mkcert' или 'tunnel')" >&2
    exit 1
    ;;
esac
