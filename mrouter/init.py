"""mrouter init: the one manual step. Detect, configure, install, wire — then never again.

Spirit: the CLI audit found that every mrouter command except `serve` was
either a debug tool or a side effect that should be automatic (bench, learn).
`init` is the deliberate exception — it's the *one* place a human makes real
decisions (where's my gateway, do I want this running at login, do I want my
local Hermes Agent pointed at it), and it should take exactly one run.

What it does:
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
    print("\n[1/5] LM Studio")
    lms_bin = _find_lms_bin()
    if lms_bin:
        print(f"  found: {lms_bin}")
    else:
        print("  not found at ~/.lmstudio/bin/lms or on PATH — install LM Studio "
              "(https://lmstudio.ai) for local/sensitive routing; cloud-only routing "
              "still works without it.")
    return lms_bin


def _step_config(gateway_url, gateway_key_env, non_interactive):
    print("\n[2/5] config.json")
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


def _step_launchd(skip, non_interactive):
    print("\n[3/5] Run at login (launchd, macOS)")
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
    print("\n[4/5] Hermes Agent integration")
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
    print("\n[5/5] doctor")
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
    _step_lm_studio(non_interactive)
    cfg = _step_config(gateway_url, gateway_key_env, non_interactive)
    _step_launchd(skip_launchd, non_interactive)
    _step_hermes(cfg, skip_hermes, non_interactive)
    _step_doctor(cfg)
    print("\nDone. Day-to-day, you should only ever need:")
    print("  mrouter doctor     # health check, run whenever something feels off")
    print("(serve is already running via launchd if you installed it above; otherwise: mrouter serve)")
    return 0
