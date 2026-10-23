#!/bin/bash
# mrouter launcher for launchd. `mrouter init` writes a launchd plist pointing here.
# Daemons get a bare PATH: add the user's local/mise tool dirs so the setec CLI
# (used to fetch decision-model keys) resolves, AND so a `python3` on PATH can
# be found at all if the hardcoded `/usr/bin/python3` below isn't available
# (e.g. a from-scratch macOS without Xcode Command Line Tools).
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$HOME/.lmstudio/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

# LM Studio startup, local-model benchmarking, and learned-adjustment recompute
# all happen automatically inside `mrouter serve` now — nothing else to do here.
cd "$HOME/workspace/model-router" || exit 1

# Prefer the interpreter `mrouter init` recorded at install time (it ran `pip
# install -e .`, so this is guaranteed to have mrouter's own package AND any
# optional deps like rumps importable). Fall back to system python3 only if
# that record is missing (pre-packaging installs) -- never silently run a
# DIFFERENT interpreter than the one a human actually pip-installed into,
# which is exactly the "pip said Requirement already satisfied but `mrouter
# tray` still can't find rumps" trap this replaces.
if [ -f ".mrouter_python" ]; then
    PY="$(cat .mrouter_python)"
else
    PY="/usr/bin/python3"
fi
exec "$PY" -m mrouter serve
