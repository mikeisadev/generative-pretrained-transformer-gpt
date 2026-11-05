"""Aggiorna i dati della web app e la avvia in locale.

    python scripts/serve_webapp.py            # http://localhost:8000
    python scripts/serve_webapp.py --port 9000 --no-export

Se esiste ``checkpoints/gpt.pt`` la scena mostra i pesi addestrati,
altrimenti quelli di un modello appena inizializzato.
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

from gpt.export import PROJECT_ROOT, export

WEBAPP_DIR = PROJECT_ROOT / "webapp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "gpt.pt")
    parser.add_argument("--no-export", action="store_true", help="usa il model.json già presente")
    args = parser.parse_args()

    if not args.no_export:
        spec = export(args.checkpoint)
        print(f"dati esportati da: {spec['source']}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEBAPP_DIR))
    # Solo localhost: la web app non va esposta sulla rete dell'aula.
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"web app su http://localhost:{args.port}  (Ctrl+C per fermare)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nserver fermato")


if __name__ == "__main__":
    main()
