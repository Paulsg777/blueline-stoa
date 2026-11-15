"""Print every Gemini model this key can call (reads GEMINI_API_KEY from the environment or .env next to this file)."""
import os, json, ssl, urllib.request
try:                                   # python.org builds on macOS ship without root certificates; certifi has them
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()
f = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(f):
    for line in open(f):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not key: raise SystemExit("no GEMINI_API_KEY in the environment or .env")
url, tok = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", ""
while True:
    req = urllib.request.Request(url + (f"&pageToken={tok}" if tok else ""), headers={"x-goog-api-key": key})
    d = json.load(urllib.request.urlopen(req, timeout=30, context=CTX))
    for m in d.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(m["name"].removeprefix("models/"), "|", m.get("displayName", ""))
    tok = d.get("nextPageToken")
    if not tok: break
