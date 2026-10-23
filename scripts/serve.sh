#!/bin/bash
# mrouter launcher for launchd. `mrouter init` writes a launchd plist pointing here.
# Daemons get a bare PATH: add the user's local/mise tool dirs so the setec CLI
# (used to fetch decision-model keys) resolves.
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$HOME/.lmstudio/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

# LM Studio startup, local-model benchmarking, and learned-adjustment recompute
# all happen automatically inside `mrouter serve` now — nothing else to do here.
cd "$HOME/workspace/model-router" || exit 1
exec /usr/bin/python3 -m mrouter serve
