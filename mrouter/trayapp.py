"""mrouter tray: a macOS menu-bar app showing live perf, spend, and model
activity from the proxy's /v1/metrics endpoints.

This is the ONE place mrouter takes on a dependency beyond the stdlib
(`rumps`) -- deliberately isolated here so the core proxy/CLI stay
zero-dependency. `mrouter tray` prints an install hint and exits cleanly if
rumps isn't present, instead of crashing the whole package import.

Design: poll /v1/metrics/live every `metrics.tray_poll_seconds` (default 10s)
for the menu-bar title (spend + p90 TTFT) and the dropdown's KPI rows +
model list. The dropdown shows each active/recent model's use case(s),
truncated to a couple of names plus "+N more" when a model serves several
use cases (see `format_use_cases`), so the menu stays a fixed, scannable
width regardless of how many use cases route to one model.
"""

import json
import urllib.error
import urllib.request

from . import discovery, metrics_store

WINDOW_CHOICES = ["1h", "24h"]


def format_use_cases(use_cases, max_shown=2):
    """Join a model's use-case list for display, truncating long lists.

    use_cases: list of {"use_case": str, "count": int} (already sorted by
    count desc by metrics_store.window_summary). Mirrors the dashboard's
    ui.html `formatUseCases` so tray and dashboard never disagree on the
    "+N more" cutoff.
    """
    if not use_cases:
        return "\u2014"
    names = [u["use_case"] for u in use_cases]
    if len(names) <= max_shown:
        return ", ".join(names)
    shown = ", ".join(names[:max_shown])
    extra = len(names) - max_shown
    return f"{shown} +{extra} more"


def fmt_usd(v):
    if v is None:
        return "\u2014"
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"


def fmt_ms(s):
    if s is None:
        return "\u2014"
    return f"{round(s * 1000)}ms" if s < 1 else f"{s:.2f}s"


def icon_title(summary):
    """Short menu-bar title: spend + p90 TTFT, the two numbers people glance
    at most (cost creeping up, latency degrading)."""
    return f"{fmt_usd(summary.get('spend_total_usd'))} \u00b7 {fmt_ms(summary.get('ttft_p90_s'))}"


def model_rows(summary, max_rows=8):
    """Sorted (by request count desc) model rows for the dropdown, each with
    a live/recent status dot, truncated use-case label, and error flag."""
    models = summary.get("models") or {}
    rows = sorted(models.items(), key=lambda kv: -kv[1]["requests"])[:max_rows]
    out = []
    for key, m in rows:
        reliability_pct = None
        if m["requests"]:
            reliability_pct = round((m["requests"] - m["errors"]) / m["requests"] * 100)
        out.append({
            "key": key,
            "display": key.split("/")[-1],
            "status": m.get("status", "recent"),
            "requests": m["requests"],
            "reliability_pct": reliability_pct,
            "use_cases_label": format_use_cases(m.get("use_cases")),
        })
    return out


class MetricsClient:
    """Thin HTTP client for the proxy's metrics endpoints. No deps beyond stdlib."""

    def __init__(self, config):
        self.config = config
        host = config["proxy"]["host"]
        port = config["proxy"]["port"]
        self.base = f"http://{host}:{port}"

    def _get(self, path):
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())

    def live(self):
        """Cheap recent-window snapshot for the icon tick."""
        return self._get("/v1/metrics/live")

    def window(self, window):
        return self._get(f"/v1/metrics?window={window}")


def _try_import_rumps():
    try:
        import rumps  # type: ignore
        return rumps
    except ImportError:
        return None


def build_app(config, rumps_mod):
    """Construct the rumps.App with its menu wired to live metrics. Split out
    from main() so tests can exercise menu construction without an actual
    macOS app event loop."""
    client = MetricsClient(config)
    state = {"window": (config.get("metrics", {}) or {}).get("default_window", "1h")}

    app = rumps_mod.App("mrouter", title="mrouter \u00b7 \u2026")

    def refresh(_=None):
        try:
            live = client.live()
            app.title = icon_title(live)
            summary = client.window(state["window"])
            _rebuild_menu(app, rumps_mod, summary, state, refresh)
        except (urllib.error.URLError, OSError):
            app.title = "mrouter \u00b7 unreachable"

    poll_s = int((config.get("metrics", {}) or {}).get("tray_poll_seconds", 10))
    timer = rumps_mod.Timer(refresh, poll_s)
    timer.start()
    refresh()
    return app


def _rebuild_menu(app, rumps_mod, summary, state, refresh_cb):
    items = []

    def make_window_button(w):
        def _click(_sender):
            state["window"] = w
            refresh_cb()
        checkmark = "\u2713 " if state["window"] == w else "  "
        btn = rumps_mod.MenuItem(f"{checkmark}{w}", callback=_click)
        return btn

    for w in WINDOW_CHOICES:
        items.append(make_window_button(w))
    items.append(None)  # separator
    items.append(rumps_mod.MenuItem(f"Spend: {fmt_usd(summary.get('spend_total_usd'))}"))
    items.append(rumps_mod.MenuItem(f"Requests: {summary.get('requests', 0)}"))
    items.append(rumps_mod.MenuItem(f"Error rate: {summary.get('error_rate', 0) * 100:.1f}%"))
    items.append(rumps_mod.MenuItem(f"p90 TTFT: {fmt_ms(summary.get('ttft_p90_s'))}"))
    tpm = summary.get("tokens_per_min_p90")
    items.append(rumps_mod.MenuItem(f"p90 tok/min: {round(tpm):,}" if tpm is not None else "p90 tok/min: \u2014"))
    items.append(None)
    items.append(rumps_mod.MenuItem("Active & recent models"))
    for row in model_rows(summary):
        dot = "\u25cf" if row["status"] == "live" else "\u25cb"
        rel = f"{row['reliability_pct']}%" if row["reliability_pct"] is not None else "?"
        label = f"{dot} {row['display']} ({row['use_cases_label']}) \u00b7 {row['requests']} \u00b7 {rel}"
        items.append(rumps_mod.MenuItem(label))
    items.append(None)
    items.append(rumps_mod.MenuItem("Open full dashboard\u2026", callback=lambda _s: _open_dashboard(app)))
    items.append(rumps_mod.MenuItem("Quit", callback=lambda _s: rumps_mod.quit_application()))

    app.menu.clear()
    app.menu.update(items)


def _open_dashboard(app):
    import webbrowser
    cfg = discovery.load_config()
    host = cfg["proxy"]["host"]
    port = cfg["proxy"]["port"]
    webbrowser.open(f"http://{host}:{port}/ui")


def main():
    rumps_mod = _try_import_rumps()
    if rumps_mod is None:
        print("mrouter tray requires the 'rumps' package (macOS menu-bar apps).")
        print("Install it with:  python3 -m pip install rumps")
        return 1
    config = discovery.load_config()
    app = build_app(config, rumps_mod)
    app.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
