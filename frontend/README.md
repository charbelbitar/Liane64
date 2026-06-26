
## 1 — Start the backend (same as before)

From your `RAG/` root:
```bash
pip install fastapi uvicorn
uvicorn app.api:app --reload --port 8000
```

## 2 — Serve the frontend

Browsers block `fetch()` calls from `file://` pages in some cases, so serve
this folder over a tiny local HTTP server rather than double-clicking `index.html`.

**Option A — Python (already installed almost everywhere):**
```bash
cd frontend
python3 -m http.server 5500
```
Open: http://localhost:5500

**Option B — VS Code "Live Server" extension** — right-click `index.html` → "Open with Live Server".


## 3 — CORS note

`app/api.py` currently allows requests from `http://localhost:5173` (the old Vite port).
Since this frontend runs on a different port (e.g. `5500`), update the CORS origins in `app/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Customizing

- **API URL** — edit `API_BASE` at the top of `app.js` if your backend runs elsewhere.
- **Markdown rendering** — `app.js` includes a tiny hand-rolled markdown renderer (bold, italics, code, lists). It's intentionally minimal; if you need full Markdown support, you can drop in a CDN script like:
  ```html
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  ```
  and replace the `renderMarkdown()` function with `marked.parse(text)`.
