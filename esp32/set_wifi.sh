#!/usr/bin/env bash
# Prompts for the WiFi password (hidden) and uploads wifi_config.py to the ESP32.
# Password never touches the git repo, the terminal history, or the chat.
set -e
read -rp "WiFi SSID (2.4 GHz network): " SSID
read -rsp "Password for $SSID: " PASS; echo
TMP="$(mktemp)"
printf 'SSID = "%s"\nPASSWORD = "%s"\n' "$SSID" "$PASS" > "$TMP"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate isaaclab
python -m mpremote connect /dev/ttyUSB0 cp "$TMP" :wifi_config.py
shred -u "$TMP" 2>/dev/null || rm -f "$TMP"
unset PASS
echo ""
echo "wifi_config.py uploaded to the ESP32 (password NOT stored on the PC)."
