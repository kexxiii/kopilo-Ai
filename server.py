"""
╔══════════════════════════════════════════════════════════════════╗
║  KOPILO AI — MERIDIAN  · server.py                              ║
║  Flask backend  ·  llama-cpp-python inference                   ║
╠══════════════════════════════════════════════════════════════════╣
║  HOST CONFIGURATION  (edit the section below)                   ║
║    1. Add your model files to the MODELS list                   ║
║    2. Set "default": True on whichever model loads at startup   ║
║    3. Run:  python server.py                                     ║
║    4. Point the frontend to  http://your-host:5000              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
#  ▼▼▼  EDIT THIS BLOCK — HOST CONFIGURATION  ▼▼▼
# ══════════════════════════════════════════════════════════════════

SERVER_HOST = "0.0.0.0"     # "0.0.0.0" = accessible on network, "127.0.0.1" = local only
SERVER_PORT = 5000
DEBUG       = False          # Set True for development logging

# List of available models.
# • "id"          — unique identifier used internally and by the frontend
# • "name"        — display name shown in the UI
# • "path"        — path to the .gguf file (absolute or relative to this script)
# • "description" — short hint shown in the model switcher dropdown
# • "default"     — exactly ONE model should have  "default": True
#                   this model loads automatically when the server starts
# • "n_ctx"       — context window size (tokens).  512–4096 typical
# • "n_threads"   — CPU threads to use.  Set to your core count for best speed
# • "n_gpu_layers"— layers to offload to GPU (0 = CPU only, -1 = all layers)

MODELS = [
    {
        "id":          "mistral-7b",
        "name":        "Mistral 7B Instruct",
        "path":        "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Fast · General purpose · Recommended",
        "default":     True,
        "n_ctx":       4096,
        "n_threads":   8,
        "n_gpu_layers": 0,
    },
    {
        "id":          "llama3-8b",
        "name":        "Llama 3 8B Instruct",
        "path":        "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "description": "Strong reasoning · Longer context",
        "default":     False,
        "n_ctx":       8192,
        "n_threads":   8,
        "n_gpu_layers": 0,
    },
    {
        "id":          "phi3-mini",
        "name":        "Phi-3 Mini",
        "path":        "models/Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Very fast · Low RAM · Lightweight",
        "default":     False,
        "n_ctx":       4096,
        "n_threads":   4,
        "n_gpu_layers": 0,
    },
]

# System prompts per-model (optional — keyed by model id)
SYSTEM_PROMPTS = {
    "mistral-7b": (
        "You are Kopilo AI, a helpful and knowledgeable assistant. "
        "Be concise, accurate, and friendly. Format code in markdown code blocks."
    ),
    "llama3-8b": (
        "You are Kopilo AI, an expert assistant with strong reasoning skills. "
        "Think step by step and be thorough in your answers."
    ),
    "phi3-mini": (
        "You are Kopilo AI. Be helpful, clear, and brief."
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are Kopilo AI, a helpful assistant. Be concise and accurate."
)

# ══════════════════════════════════════════════════════════════════
#  END OF HOST CONFIGURATION — no need to edit below this line
# ══════════════════════════════════════════════════════════════════

import os
import logging
import threading
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kopilo")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── State ──────────────────────────────────────────────────────────
_llm        = None          # active llama_cpp.Llama instance
_active_id  = None          # id string of the loaded model
_model_lock = threading.Lock()

# ── Helpers ────────────────────────────────────────────────────────

def _model_cfg(model_id: str) -> dict | None:
    return next((m for m in MODELS if m["id"] == model_id), None)

def _default_cfg() -> dict | None:
    default = next((m for m in MODELS if m.get("default")), None)
    return default or (MODELS[0] if MODELS else None)

def _load_model(model_id: str) -> tuple[bool, str]:
    """Load a model by id.  Returns (success, message)."""
    global _llm, _active_id

    cfg = _model_cfg(model_id)
    if not cfg:
        return False, f"Unknown model id: {model_id}"

    path = Path(cfg["path"])
    if not path.is_absolute():
        path = Path(__file__).parent / path

    if not path.exists():
        return False, f"Model file not found: {path}"

    # Unload current model to free RAM
    with _model_lock:
        if _llm is not None:
            log.info("Unloading %s", _active_id)
            del _llm
            _llm = None
            _active_id = None

    log.info("Loading %s  [%s]", cfg["name"], path)
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(path),
            n_ctx=cfg.get("n_ctx", 2048),
            n_threads=cfg.get("n_threads", 4),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            verbose=False,
        )
        with _model_lock:
            _llm = llm
            _active_id = model_id
        log.info("✓ Model ready: %s", cfg["name"])
        return True, cfg["name"]
    except Exception as exc:
        log.error("Failed to load %s: %s", cfg["name"], exc)
        return False, str(exc)


def require_model(f):
    """Decorator: return 503 if no model is loaded."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if _llm is None:
            return jsonify({"error": "No model loaded"}), 503
        return f(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/api/status")
def status():
    try:
        import duckduckgo_search  # noqa: F401
        has_ddg = True
    except ImportError:
        has_ddg = False

    return jsonify({
        "ok":          True,
        "model_loaded": _llm is not None,
        "model_name":  _model_cfg(_active_id)["name"] if _active_id else None,
        "active_model": _active_id,
        "has_ddg":     has_ddg,
    })


@app.route("/api/models")
def get_models():
    """Return the configured model list with active flag."""
    return jsonify({
        "active_model": _active_id,
        "models": [
            {
                "id":          m["id"],
                "name":        m["name"],
                "description": m.get("description", ""),
                "is_default":  bool(m.get("default")),
                "active":      m["id"] == _active_id,
            }
            for m in MODELS
        ],
    })


@app.route("/api/model/switch", methods=["POST"])
def switch_model():
    """Switch to a different model by id."""
    data = request.get_json(force=True, silent=True) or {}
    model_id = data.get("model_id", "").strip()

    if not model_id:
        return jsonify({"success": False, "error": "model_id is required"}), 400

    if model_id == _active_id:
        cfg = _model_cfg(model_id)
        return jsonify({"success": True, "model_name": cfg["name"] if cfg else model_id})

    success, msg = _load_model(model_id)
    if success:
        cfg = _model_cfg(model_id)
        return jsonify({"success": True, "model_name": cfg["name"] if cfg else model_id})
    return jsonify({"success": False, "error": msg}), 500


@app.route("/api/chat", methods=["POST"])
@require_model
def chat():
    """
    Chat endpoint.
    Body: {
        "messages": [{"role": "user"|"assistant", "content": "..."}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": false,
        "system": "optional override system prompt"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    messages  = data.get("messages", [])
    temp      = float(data.get("temperature", 0.7))
    max_tok   = int(data.get("max_tokens", 512))
    streaming = bool(data.get("stream", False))
    sys_over  = data.get("system", "").strip()

    # Build system prompt
    sys_prompt = sys_over or SYSTEM_PROMPTS.get(_active_id, DEFAULT_SYSTEM_PROMPT)

    # Build the full message list
    full_messages = [{"role": "system", "content": sys_prompt}] + messages

    if streaming:
        def generate():
            with _model_lock:
                for chunk in _llm.create_chat_completion(
                    messages=full_messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=True,
                ):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text  = delta.get("content", "")
                    if text:
                        yield text
        return Response(stream_with_context(generate()), mimetype="text/plain")

    # Non-streaming
    with _model_lock:
        result = _llm.create_chat_completion(
            messages=full_messages,
            temperature=temp,
            max_tokens=max_tok,
        )
    reply = result["choices"][0]["message"]["content"]
    return jsonify({"reply": reply, "model": _active_id})


@app.route("/api/search")
def search():
    """DuckDuckGo search proxy."""
    query    = request.args.get("q", "").strip()
    max_res  = int(request.args.get("max", 5))
    do_fetch = request.args.get("fetch", "true").lower() == "true"

    if not query:
        return jsonify({"error": "No query"}), 400

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return jsonify({"error": "duckduckgo_search not installed"}), 501

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_res))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if do_fetch:
        import urllib.request
        for r in results:
            try:
                req = urllib.request.Request(
                    r["href"],
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=4) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                # Very simple text extraction
                import re
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s{2,}", " ", text).strip()
                r["page_text"] = text[:2000]
            except Exception:
                r["page_text"] = ""

    return jsonify({"results": results})


@app.route("/api/search/ai", methods=["POST"])
@require_model
def search_ai():
    """Run a web search then summarise results with the active model."""
    data    = request.get_json(force=True, silent=True) or {}
    query   = data.get("query", "").strip()
    context = data.get("context", "")

    if not query:
        return jsonify({"error": "No query"}), 400

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    snippets = "\n\n".join(
        f"[{i+1}] {h['title']}\n{h['body']}" for i, h in enumerate(hits)
    )
    prompt = (
        f"Using the search results below, answer the query concisely.\n\n"
        f"Query: {query}\n\n"
        f"Results:\n{snippets}\n\n"
        f"{'Additional context: '+context if context else ''}\n\nAnswer:"
    )

    with _model_lock:
        result = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS.get(_active_id, DEFAULT_SYSTEM_PROMPT)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    answer = result["choices"][0]["message"]["content"]
    return jsonify({"answer": answer, "results": hits})


@app.route("/api/files/upload", methods=["POST"])
def upload_file():
    """Accept file uploads and return extracted text."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f    = request.files["file"]
    name = f.filename or "file"
    data = f.read()

    text = ""
    ext  = Path(name).suffix.lower()

    try:
        if ext == ".pdf":
            import io
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                text = "[PDF support requires: pip install pypdf]"

        elif ext in (".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html", ".css"):
            text = data.decode("utf-8", errors="replace")

        elif ext in (".docx",):
            try:
                import io, zipfile, xml.etree.ElementTree as ET
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                ns   = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = " ".join(t.text for t in tree.findall(".//w:t", ns) if t.text)
            except Exception as exc:
                text = f"[DOCX parse error: {exc}]"
        else:
            text = f"[File type {ext} not supported for text extraction]"

    except Exception as exc:
        text = f"[Error extracting text: {exc}]"

    return jsonify({
        "name":  name,
        "size":  len(data),
        "text":  text[:8000],   # cap at 8k chars for context safety
        "chars": len(text),
    })


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════

def startup():
    """Auto-load the default model before the server accepts requests."""
    cfg = _default_cfg()
    if not cfg:
        log.warning("No models configured in MODELS list — skipping auto-load")
        return

    log.info("━" * 58)
    log.info("  KOPILO AI — MERIDIAN  ·  server.py")
    log.info("━" * 58)
    log.info("  Configured models:")
    for m in MODELS:
        tag = " ← DEFAULT" if m.get("default") else ""
        log.info("    • %-30s  %s%s", m["name"], m["id"], tag)
    log.info("━" * 58)
    log.info("  Auto-loading default: %s", cfg["name"])

    success, msg = _load_model(cfg["id"])
    if success:
        log.info("  ✓ Default model loaded: %s", msg)
    else:
        log.error("  ✗ Failed to load default model: %s", msg)
        log.error("  Check the 'path' in the MODELS config above.")

    log.info("━" * 58)
    log.info("  API running at http://%s:%s", SERVER_HOST, SERVER_PORT)
    log.info("━" * 58)


if __name__ == "__main__":
    startup()
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG, threaded=True)
