"""Serve the cross-sell playback dashboard as a local web page.

    python -m rl_nba.serve                     # generate + serve on http://127.0.0.1:8000
    rl-nba-dashboard --port 9000               # console script, custom port
    rl-nba-dashboard --from-file path.html     # serve an existing HTML, skip regeneration

From a notebook, use :func:`serve_background`, which starts the server on a
daemon thread and returns immediately (call ``server.shutdown()`` to stop).

The dashboard is regenerated from the current config on each launch (so config
edits show up), written to a temp directory, and served over localhost only.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import tempfile
import threading
import webbrowser
from functools import partial
from pathlib import Path

from rl_nba.config import AppConfig, load_config
from rl_nba.playback import write_dashboard


def _build_dashboard_dir(
    *,
    config: AppConfig | None = None,
    from_file: str | Path | None = None,
    model: str = "champion",
    rounds: int | None = None,
    snapshots: int = 380,
) -> Path:
    """Write the dashboard (freshly rendered, or copied from a file) into a temp dir."""
    directory = Path(tempfile.mkdtemp(prefix="rl_nba_dashboard_"))
    if from_file is not None:
        source = Path(from_file)
        if not source.is_file():
            raise SystemExit(f"--from-file not found: {source}")
        (directory / "index.html").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        if config is None:
            raise ValueError("Provide either config or from_file")
        write_dashboard(
            config, directory / "index.html", model=model, rounds=rounds, snapshots=snapshots
        )
    return directory


def _bind_server(
    directory: Path, host: str, port: int, max_tries: int = 20
) -> tuple[socketserver.TCPServer, int]:
    """Bind a one-directory static server, scanning forward for a free port."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    last_error: OSError | None = None
    for candidate in range(port, port + max_tries):
        try:
            return socketserver.TCPServer((host, candidate), handler), candidate
        except OSError as error:
            last_error = error
    raise SystemExit(
        f"No free port in {port}–{port + max_tries - 1} on {host} ({last_error}). "
        "Pass a different --port."
    )


def serve_background(
    config: AppConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str = "champion",
    rounds: int | None = None,
    snapshots: int = 380,
) -> tuple[socketserver.TCPServer, str]:
    """Serve the dashboard on a daemon thread (non-blocking). Returns ``(server, url)``.

    Intended for notebooks: the call returns immediately while the server keeps
    running. Stop it with ``server.shutdown()``. ``model`` picks which model to
    play back (``"champion"`` by default, or a type name like ``"linucb"``).
    """
    directory = _build_dashboard_dir(
        config=config, model=model, rounds=rounds, snapshots=snapshots
    )
    server, bound_port = _bind_server(directory, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{bound_port}/"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve the cross-sell playback dashboard on localhost."
    )
    parser.add_argument(
        "--config", default="config/rl_nba_config.yml", help="config to render from"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: localhost only)"
    )
    parser.add_argument("--port", type=int, default=8000, help="preferred port (default: 8000)")
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="episode length to replay (default: the config's full experiment.n_rounds)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=380,
        help="decision cards sampled across the run (default: 380)",
    )
    parser.add_argument(
        "--model",
        default="champion",
        help="model to play back: champion (best of the config's models) or a type "
        "name like linucb / rff_ucb / random (default: champion)",
    )
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--from-file", default=None, help="serve this existing HTML instead of regenerating"
    )
    args = parser.parse_args(argv)

    if args.from_file:
        directory = _build_dashboard_dir(from_file=args.from_file)
        print(f"Serving existing dashboard: {args.from_file}")
    else:
        directory = _build_dashboard_dir(
            config=load_config(args.config),
            model=args.model,
            rounds=args.rounds,
            snapshots=args.snapshots,
        )
        print(f"Generated dashboard from {args.config} (full learning run).")

    server, port = _bind_server(directory, args.host, args.port)
    url = f"http://{args.host}:{port}/"
    print(f"\n  Playback dashboard  →  {url}\n  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
