#!/usr/bin/env bash
# Turn off unattended updates on a lynxduck board.
#
# The board runs builds from our `lynxduck` branch. With the stock `auto_apply = "mandatory"`,
# the six-hourly check installs any mandatory official release on its own and replaces our build.
# This sets `auto_apply = "off"` in the board's /etc/robot/updater.toml and restarts updaterd.
#
# install.sh never overwrites that file, so running this once per board is enough. Updates
# still work when asked for by hand (`robotctl update ...` or the app).
#
# Usage: scripts/lynxduck-no-autoupdate.sh microduck@192.168.1.42
#        (or set DUCK_BOARD and pass nothing)

set -euo pipefail

BOARD="${1:-${DUCK_BOARD:-}}"
if [ -z "$BOARD" ]; then
    echo "usage: $0 user@board-address   (or set DUCK_BOARD)" >&2
    exit 2
fi

ssh -t "$BOARD" '
set -eu
CONFIG=/etc/robot/updater.toml
if [ ! -f "$CONFIG" ]; then
    echo "$CONFIG not found: install the robot software first" >&2
    exit 1
fi
if ! grep -q "^auto_apply" "$CONFIG"; then
    echo "$CONFIG has no auto_apply line; edit it by hand" >&2
    exit 1
fi
sudo sed -i "s/^auto_apply.*/auto_apply = \"off\"/" "$CONFIG"
sudo systemctl restart updaterd
echo "now: $(grep "^auto_apply" "$CONFIG")"
'
