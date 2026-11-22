"""Load local settings and start the FastAPI workspace with python -m backend."""
import argparse
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the claims backend")
    parser.add_argument("--reload", action="store_true", help="Restart on backend Python changes during local development")
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8010, access_log=False,
                reload=args.reload, reload_dirs=[str(Path(__file__).resolve().parent)] if args.reload else None)


if __name__ == "__main__":
    main()
