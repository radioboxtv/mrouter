"""mrouter init: the one manual step. Detect, configure, install, wire — then never again.

Spirit: the CLI audit found that every mrouter command except `serve` was
either a debug tool or a side effect that should be automatic (bench, learn).
`init` is the deliberate exception — it's the *one* place a human makes real
decisions (where's my gateway, do I want this running at login, do I want my
local Hermes Agent pointed at it), and it should take exactly one run.

What it does:
  0. Install mrouter as a real package (`pip install -e .`) using the
     interpreter that's actually running `init` right now, and record that
     interpreter's path so launchd (which gets a bare PATH, no venv/mise
     context) execs the SAME one instead of a hardcoded guess.
  1. Detect LM Studio (`lms` binary, local server) — no user input needed.
  2. Configure (or confirm) the cloud gateway endpoint + key env var.
  3. Write config.json from config.example.json if missing.
  4. Install a launchd agent (macOS) so `serve` runs at login and restarts on
     crash/logout — the same role scripts/serve.sh + a hand-written plist
     played before, now one command.
  5. Offer to point a local Hermes Agent profile at mrouter as `model.default
     = auto` with `model.provider = custom`, `model.base_url` = the proxy —
     and set up a cloud fallback so Hermes survives mrouter being down.

Idempotent: safe to re-run. Each step detects its current state and either
leaves it alone (already correct), offers to fix it, or explains why it's
skipping (e.g. --no-launchd, non-macOS, Hermes not installed).
"""

import json
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE, "config.json")
EXAMPLE_PATH = os.path.join(BASE, "config.example.json")
LAUNCHD_LABEL = "com.mrouter.serve"
LAUNCHD_PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
SERVE_SCRIPT = os.path.join(BASE, "scripts", "serve.sh")
# Records which python3 interpreter `init` installed mrouter into (see
# _step_install below), so scripts/serve.sh (launchd, which gets a bare PATH
# with no venv/mise context) uses the SAME interpreter a human actually
# `pip install`ed optional deps (rumps) into, instead of a hardcoded guess
# like the old `/usr/bin/python3` that silently diverged from whatever
# `python3` resolved to in the user's actual shell.
PYTHON_RECORD_PATH = os.path.join(BASE, ".mrouter_python")
# Older/ad-hoc installs used other labels pointing at the same serve.sh; if one
# of these is already loaded, installing our own on top would start a second
# `serve` process fighting over the same proxy port. Detected by scanning
# ProgramArguments for serve.sh rather than guessing every possible label.
_LAUNCHD_AGENTS_DIR = os.path.expanduser("~/Library/LaunchAgents")


def _step_install(non_interactive):
    """Install mrouter as a real package (`pip install -e .`) using
    `sys.executable` -- the interpreter actually running `mrouter init` right
    now, i.e. whatever `python3` resolved to in the user's shell/mise/venv --
    and record that interpreter's path so launchd (scripts/serve.sh), which
    gets a bare PATH, execs the SAME interpreter instead of a hardcoded guess.

    This also gives `mrouter` a real console-script entry point (via
    pyproject.toml's [project.scripts]) so there's no more hand-written
    ~/.local/bin/mrouter wrapper for anyone to maintain or get out of sync.
    """
    print("\n[1/6] Package install")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", BASE, "--quiet"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        with open(PYTHON_RECORD_PATH, "w") as f:
            f.write(sys.executable)
        print(f"  installed editable (pip install -e .) with {sys.executable}")
        print(f"  recorded interpreter -> {PYTHON_RECORD_PATH} (used by scripts/serve.sh under launchd)")
    except subprocess.CalledProcessError as e:
        print(f"  pip install -e . failed ({e}); `mrouter` commands still work via "
              f"`python3 -m mrouter` from {BASE}")
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e}); `mrouter` commands still work via "
              f"`python3 -m mrouter` from {BASE}")


def _ask(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        val = ""
    return val or default


def _find_lms_bin():
    candidates = [
        os.path.expanduser("~/.lmstudio/bin/lms"),
        shutil.which("lms"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _step_lm_studio(non_interactive):
    print("\n[2/6] LM Studio")
    lms_bin = _find_lms_bin()
    if lms_bin:
        print(f"  found: {lms_bin}")
    else:
        print("  not found at ~/.lmstudio/bin/lms or on PATH — install LM Studio "
              "(https://lmstudio.ai) for local/sensitive routing; cloud-only routing "
              "still works without it.")
    return lms_bin


def _step_config(gateway_url, gateway_key_env, non_interactive):
    print("\n[3/6] config.json")
    if os.path.exists(CONFIG_PATH):
        print(f"  already exists: {CONFIG_PATH} (leaving as-is — edit it directly, or "
              "delete it and re-run init to start fresh)")
        with open(CONFIG_PATH) as f:
            return json.load(f)

    if not os.path.exists(EXAMPLE_PATH):
        print(f"  ERROR: {EXAMPLE_PATH} missing — can't bootstrap config.json")
        return None
    with open(EXAMPLE_PATH) as f:
        cfg = json.load(f)

    if gateway_url is None and not non_interactive:
        gateway_url = _ask("Cloud gateway base URL (blank to skip cloud, local-only)", "")
    if gateway_url:
        cfg.setdefault("providers", {}).setdefault("aperture", {})["api_base"] = gateway_url
        cfg["providers"]["aperture"]["enabled"] = True
        if gateway_key_env is None and not non_interactive:
            gateway_key_env = _ask("Env var holding the gateway API key", "APERTURE_API_KEY")
        cfg["providers"]["aperture"]["api_key_env"] = gateway_key_env or "APERTURE_API_KEY"
    else:
        cfg.setdefault("providers", {}).setdefault("aperture", {})["enabled"] = False
        print("  cloud gateway skipped — set providers.aperture.enabled/api_base in "
              "config.json later, or re-run init.")

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote: {CONFIG_PATH}")
    return cfg


def _find_existing_serve_agent():
    """Scan ~/Library/LaunchAgents for any plist whose ProgramArguments already
    point at our serve.sh (by label `com.mrouter.serve` or any older/ad-hoc
    label from a pre-init install) — installing a second one would collide on
    the same proxy port. Returns the plist path, or None."""
    if not os.path.isdir(_LAUNCHD_AGENTS_DIR):
        return None
    try:
        import plistlib
    except ImportError:
        plistlib = None
    for name in os.listdir(_LAUNCHD_AGENTS_DIR):
        if not name.endswith(".plist"):
            continue
        path = os.path.join(_LAUNCHD_AGENTS_DIR, name)
        try:
            if plistlib is not None:
                with open(path, "rb") as f:
                    data = plistlib.load(f)
                args = data.get("ProgramArguments") or []
                if any(SERVE_SCRIPT == a or a.endswith("/scripts/serve.sh") for a in args):
                    return path
            else:
                with open(path) as f:
                    if "serve.sh" in f.read():
                        return path
        except Exception:
            continue
    return None


def _step_launchd(skip, non_interactive):
    print("\n[4/6] Run at login (launchd, macOS)")
    if skip:
        print("  skipped (--no-launchd)")
        return
    if sys.platform != "darwin":
        print("  skipped (not macOS) — run `mrouter serve` under your init system of choice "
              "(systemd unit, pm2, etc.)")
        return
    if os.path.exists(LAUNCHD_PLIST):
        print(f"  already installed: {LAUNCHD_PLIST}")
        return
    existing = _find_existing_serve_agent()
    if existing:
        print(f"  found an existing launchd agent already running mrouter: {existing}")
        print("  skipping — installing a second one would start two `serve` processes "
              "competing for the same port. Remove the old one first if you want init's "
              "agent instead:")
        print(f"    launchctl bootout gui/$(id -u) {existing} && rm {existing}")
        return
    if not non_interactive:
        if _ask("Install a launchd agent to run `mrouter serve` at login? (y/n)", "y").lower() not in ("y", "yes"):
            print("  skipped by user")
            return

    log_path = os.path.join(BASE, "mrouter.launchd.log")
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{SERVE_SCRIPT}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""
    os.makedirs(os.path.dirname(LAUNCHD_PLIST), exist_ok=True)
    with open(LAUNCHD_PLIST, "w") as f:
        f.write(plist)
    try:
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", LAUNCHD_PLIST],
                        capture_output=True, text=True, timeout=10)
        print(f"  installed + loaded: {LAUNCHD_PLIST}")
        print(f"  logs: {log_path}")
    except Exception as e:
        print(f"  wrote {LAUNCHD_PLIST} but could not load it ({e}) — run:")
        print(f"    launchctl bootstrap gui/$(id -u) {LAUNCHD_PLIST}")


def _hermes_bin():
    return shutil.which("hermes")


def _step_hermes(cfg, skip, non_interactive):
    print("\n[5/6] Hermes Agent integration")
    if skip:
        print("  skipped (--no-hermes)")
        return
    hb = _hermes_bin()
    if not hb:
        print("  `hermes` not found on PATH — skipping. Wire it up later with:")
        _print_hermes_commands(cfg)
        return
    if not non_interactive:
        if _ask("Point this machine's Hermes Agent at mrouter as the 'auto' model? (y/n)", "y").lower() not in ("y", "yes"):
            print("  skipped by user. Commands to do it later:")
            _print_hermes_commands(cfg)
            return

    host = (cfg or {}).get("proxy", {}).get("host", "127.0.0.1")
    port = (cfg or {}).get("proxy", {}).get("port", 8730)
    base_url = f"http://{host}:{port}/v1"
    cmds = [
        [hb, "config", "set", "model.provider", "custom"],
        [hb, "config", "set", "model.default", "auto"],
        [hb, "config", "set", "model.base_url", base_url],
    ]
    for c in cmds:
        try:
            subprocess.run(c, capture_output=True, text=True, timeout=15, check=True)
        except Exception as e:
            print(f"  FAILED: {' '.join(c)} ({e})")
            return
    print(f"  hermes model.provider=custom, model.default=auto, model.base_url={base_url}")
    print("  reminder: set a cloud fallback so Hermes survives the proxy being down:")
    print("    hermes fallback add")


def _print_hermes_commands(cfg):
    host = (cfg or {}).get("proxy", {}).get("host", "127.0.0.1")
    port = (cfg or {}).get("proxy", {}).get("port", 8730)
    base_url = f"http://{host}:{port}/v1"
    print(f"    hermes config set model.provider custom")
    print(f"    hermes config set model.default auto")
    print(f"    hermes config set model.base_url {base_url}")
    print(f"    hermes fallback add   # keep a cloud model up in case mrouter is down")


def _step_doctor(cfg):
    print("\n[6/6] doctor")
    from . import doctor as doctor_mod
    from . import discovery
    try:
        config = cfg or discovery.load_config()
        report = doctor_mod.run(config)
        doctor_mod.print_report(report)
    except Exception as e:
        print(f"  doctor run failed ({type(e).__name__}: {e}) — run `mrouter doctor` manually once "
              "LM Studio / the gateway are reachable.")


def run(gateway_url=None, gateway_key_env=None, skip_launchd=False, skip_hermes=False,
        non_interactive=False):
    print("mrouter init")
    print("=" * 40)
    _step_install(non_interactive)
    _step_lm_studio(non_interactive)
    cfg = _step_config(gateway_url, gateway_key_env, non_interactive)
    _step_launchd(skip_launchd, non_interactive)
    _step_hermes(cfg, skip_hermes, non_interactive)
    _step_doctor(cfg)
    print("\nDone. Day-to-day, you should only ever need:")
    print("  mrouter doctor     # health check, run whenever something feels off")
    print("(serve is already running via launchd if you installed it above; otherwise: mrouter serve)")
    return 0
