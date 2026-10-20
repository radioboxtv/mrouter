#!/bin/bash
# mrouter launcher for launchd. Starts LM Studio's server if down, then serves forever.
export PATH="$HOME/.lmstudio/bin:$PATH"

# Best-effort: bring up LM Studio's local server (routing 'sensitive' depends on it).
if ! curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null 2>&1; then
    "$HOME/.lmstudio/bin/lms" server start > /dev/null 2>&1 || true
fi

cd "$HOME/workspace/model-router" || exit 1
exec /usr/bin/python3 -c "from mrouter import discovery, proxy; proxy.serve_forever(discovery.load_config())"
