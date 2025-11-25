from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .ingestion import (
    UPLOAD_DIR,
    CHECKOUT_PATH,
    ingest_files,
    save_checkout_html,
    load_checkout_excerpt,
    load_checkout_html,
)
from .models import (
    GenerateScriptRequest,
    GenerateScriptResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
    IngestResponse,
)
from .rag import generate_selenium_script, generate_test_cases

app = FastAPI(
    title="Autonomous QA Agent Backend",
    version="0.1.0",
    description="FastAPI backend powering ingestion, test case generation, and Selenium scripts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve checkout HTML statically at /checkout/...
app.mount(
    "/checkout",
    StaticFiles(directory=str(CHECKOUT_PATH.parent), html=True),
    name="checkout",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    reset: bool = Form(False),
    docs: List[UploadFile] = File(default_factory=list),
    checkout: Optional[UploadFile] = File(None),
):
    """
    Upload support docs + checkout HTML and build the knowledge base.
    """
    if not docs and checkout is None:
        raise HTTPException(status_code=400, detail="Upload at least one document or checkout.html.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for file in docs:
        dest = UPLOAD_DIR / file.filename
        dest.write_bytes(await file.read())
        saved_paths.append(dest)

    html_saved = False
    if checkout:
        content = await checkout.read()
        save_checkout_html(content, filename=checkout.filename)
        html_saved = True

    result = ingest_files(saved_paths, reset=reset)
    return IngestResponse(
        docs_ingested=result.docs_ingested,
        chunks_added=result.chunks_added,
        sources=result.sources,
        html_saved=html_saved,
    )


@app.post("/generate-testcases", response_model=GenerateTestCasesResponse)
def generate_testcases_endpoint(request: GenerateTestCasesRequest):
    return generate_test_cases(request.query, top_k=request.top_k)


@app.post("/generate-script", response_model=GenerateScriptResponse)
def generate_script_endpoint(request: GenerateScriptRequest):
    return generate_selenium_script(
        request.test_case,
        top_k=request.top_k,
        base_url=request.base_url,
    )


@app.get("/checkout-html")
def checkout_html():
    """
    Convenience endpoint to fetch the current checkout HTML.
    """
    return {"html": load_checkout_html(), "excerpt": load_checkout_excerpt()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
