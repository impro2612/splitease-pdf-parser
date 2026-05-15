import asyncio
import base64
import io
import logging
import os
import re
from typing import Optional

import pdfplumber
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MAX_FILE_BYTES = 15 * 1024 * 1024   # 15 MB decoded
MAX_PAGES     = 30
PARSE_TIMEOUT = 60                   # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "https://splitease.vercel.app")

app = FastAPI(title="SplitEase PDF Parser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_ALLOWED_ORIGIN],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ParseRequest(BaseModel):
    # ~10 MB base64 ≈ 7.5 MB decoded PDF — prevents memory exhaustion on Render's free tier
    pdf_base64: str = Field(..., max_length=10_000_000)
    password: Optional[str] = Field(None, max_length=512)


# Matches Indian bank amounts: digits with optional commas and one decimal part
_AMOUNT_RE = re.compile(r"^\d[\d,]*\.?\d*$")
_FOOTER_RE = re.compile(
    r"(statementsummary|openingbalance|generatedon|hdfcbanklimited|closingbalanceincludes|"
    r"registeredofficeaddress|contentsofthisstatement|gstn|drcount|crcount)",
    re.IGNORECASE,
)


def parse_amount(text: str) -> float:
    if not text:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    if not cleaned:
        return 0.0
    try:
        val = float(cleaned)
        return val if val > 0 else 0.0
    except ValueError:
        return 0.0


def looks_like_amount(text: str) -> bool:
    """True only for words that look like monetary amounts (e.g. '2,554.79')."""
    t = text.strip()
    return bool(_AMOUNT_RE.match(t)) and ("," in t or "." in t)


def parse_date(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", text.strip())
    if not m:
        return ""
    d, mo, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        # POSIX pivot: 00-68 → 2000s, 69-99 → 1900s (same rule as strptime %y)
        pivot = int(y)
        year = str(2000 + pivot) if pivot <= 68 else str(1900 + pivot)
    else:
        year = y
    try:
        from datetime import datetime
        datetime(int(year), int(mo), int(d))
    except ValueError:
        return ""
    return f"{year}-{mo.zfill(2)}-{d.zfill(2)}"


def group_words_into_rows(words: list, y_tol: float = 4) -> list:
    """Group pdfplumber word dicts by their vertical (top) position."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows = []
    cur = [sorted_w[0]]
    cur_top: float = sorted_w[0]["top"]
    for w in sorted_w[1:]:
        if abs(w["top"] - cur_top) <= y_tol:
            cur.append(w)
        else:
            rows.append(sorted(cur, key=lambda x: x["x0"]))
            cur = [w]
            cur_top = w["top"]
    rows.append(sorted(cur, key=lambda x: x["x0"]))
    return rows


def _classify_header_word(text: str) -> str:
    """Map a header cell text to a column role."""
    t = text.lower().strip(".")
    if "date" in t and "value" not in t:
        return "date"
    if any(k in t for k in ["narration", "description", "particulars"]):
        return "narr"
    if any(k in t for k in ["chq/ref", "chq", "refno", "ref.no", "ref no", "reference", "utr"]):
        return "ref"
    if any(k in t for k in ["withdrawal", "debit", "dr"]):
        return "debit"
    if any(k in t for k in ["deposit", "credit", "cr"]):
        return "credit"
    return "other"


def detect_word_columns(rows: list, page_w: float) -> tuple[Optional[int], dict]:
    """Detect header columns from grouped word rows and build reusable bounds."""
    header_idx = None
    all_cols: list = []
    role_cx: dict = {}

    for i, row in enumerate(rows):
        joined = " ".join(w["text"].lower() for w in row)
        if not (("withdrawal" in joined or "debit" in joined) and
                ("deposit" in joined or "credit" in joined) and
                "date" in joined and
                "narration" in joined):
            continue
        header_idx = i
        for w in row:
            role = _classify_header_word(w["text"])
            cx = (w["x0"] + w["x1"]) / 2
            all_cols.append((cx, role))
            if role != "other":
                role_cx.setdefault(role, cx)
        all_cols.sort(key=lambda x: x[0])
        break

    if header_idx is None:
        return None, {}

    # Fill in HDFC defaults for any role that wasn't detected
    defaults = {"date": 60.0, "narr": 230.0, "ref": 355.0, "debit": 415.0, "credit": 495.0}
    for role, default_cx in defaults.items():
        if role not in role_cx:
            role_cx[role] = default_cx
            all_cols.append((default_cx, role))
    all_cols.sort(key=lambda x: x[0])

    def bounds_for_cx(target_cx: float):
        for i, (v, _) in enumerate(all_cols):
            if abs(v - target_cx) < 1.0:
                lo = (all_cols[i - 1][0] + target_cx) / 2 if i > 0 else 0.0
                hi = (target_cx + all_cols[i + 1][0]) / 2 if i < len(all_cols) - 1 else page_w
                return lo, hi
        return 0.0, page_w

    d_lo,  d_hi  = bounds_for_cx(role_cx["date"])
    n_lo,  n_hi  = bounds_for_cx(role_cx["narr"])
    r_lo,  r_hi  = bounds_for_cx(role_cx["ref"])
    db_lo, db_hi = bounds_for_cx(role_cx["debit"])
    cr_lo, cr_hi = bounds_for_cx(role_cx["credit"])

    return header_idx, {
        "date_cx": role_cx["date"],
        "narr_cx": role_cx["narr"],
        "ref_cx": role_cx["ref"],
        "debit_cx": role_cx["debit"],
        "credit_cx": role_cx["credit"],
        "date_bounds": (d_lo, d_hi),
        "narr_bounds": (n_lo, n_hi),
        "ref_bounds": (r_lo, r_hi),
        "debit_bounds": (db_lo, db_hi),
        "credit_bounds": (cr_lo, cr_hi),
    }


def extract_row_narration(row: list, spec: dict) -> str:
    """
    Extract narration from a dated transaction row.

    Primary strategy uses the detected narration bounds. For statements where
    the first narration token is very short (e.g. "UPI-CRED", "UPI-FLIPKART",
    "ACHD-"), the token can sit too close to the date column and fall outside
    the narration bounds. In that case, fall back to collecting non-date,
    non-amount, non-reference words that live between the date and amount
    columns.
    """
    def mid(w) -> float:
        return (w["x0"] + w["x1"]) / 2

    _, d_hi = spec["date_bounds"]
    n_lo, n_hi = spec["narr_bounds"]
    db_lo, _ = spec["debit_bounds"]

    narr_words = [w["text"] for w in row if n_lo <= mid(w) <= n_hi]
    narr_text = " ".join(narr_words).strip()
    if narr_text:
        return narr_text

    fallback_words: list[str] = []
    for w in row:
        text = w["text"].strip()
        cx = mid(w)
        if not text:
            continue
        if cx >= db_lo:
            continue
        if parse_date(text):
            continue
        if looks_like_amount(text):
            continue
        if re.fullmatch(r"\d{8,}", re.sub(r"\D", "", text)):
            continue
        fallback_words.append(text)

    return " ".join(fallback_words).strip()


def extract_by_words(page, prev_spec: Optional[dict] = None) -> tuple[list, dict]:
    """
    Extract transactions using word x/y coordinates.

    Handles bank PDFs (e.g. HDFC) where the outer table border has no internal
    row lines, causing pdfplumber's table extractor to merge all transactions on
    a page into a single multi-value cell.  We instead group words by their
    y-position and assign them to columns by x-position.
    """
    words = page.extract_words(x_tolerance=5, y_tolerance=3)
    if not words:
        return [], prev_spec or {}

    rows = group_words_into_rows(words)
    page_w = float(page.width)
    header_idx, detected_spec = detect_word_columns(rows, page_w)
    spec = detected_spec or prev_spec or {}
    if not spec:
        return [], {}

    def mid(w) -> float:
        return (w["x0"] + w["x1"]) / 2

    def best_amount(row_words, lo, hi, col_cx_val) -> float:
        """Return the amount from the word in [lo, hi] that looks most like a number."""
        candidates = [w for w in row_words if lo <= mid(w) <= hi and looks_like_amount(w["text"])]
        if not candidates:
            return 0.0
        # Prefer the word whose centre is closest to the column header centre
        best = min(candidates, key=lambda w: abs(mid(w) - col_cx_val))
        return parse_amount(best["text"])

    d_lo, d_hi = spec["date_bounds"]
    n_lo, n_hi = spec["narr_bounds"]
    r_lo, r_hi = spec["ref_bounds"]
    db_lo, db_hi = spec["debit_bounds"]
    cr_lo, cr_hi = spec["credit_bounds"]

    # Start after header row if present; otherwise skip the repeated account
    # summary area and wait for the first actual dated row.
    start_idx = header_idx + 1 if header_idx is not None else 0

    transactions = []
    current: Optional[dict] = None

    def flush_current():
        nonlocal current
        if current and current.get("date") and current.get("description") and current.get("amount", 0) > 0:
            current["description"] = re.sub(r"\s+", " ", current["description"]).strip()
            transactions.append(current)
        current = None

    prev_top: Optional[float] = None
    for row in rows[start_idx:]:
        row_top = row[0]["top"] if row else None
        date_text = " ".join(w["text"] for w in row if d_lo <= mid(w) <= d_hi)
        date_str = parse_date(date_text)
        narr_text = extract_row_narration(row, spec)

        if not date_str:
            gap = (row_top - prev_top) if (row_top is not None and prev_top is not None) else 0
            outside_narr = [w for w in row if not (n_lo <= mid(w) <= n_hi)]
            # Continuation lines are tightly spaced and live almost entirely in
            # the narration column. Summary/footer blocks should not be glued to
            # the previous transaction.
            joined_row = " ".join(w["text"] for w in row)
            if _FOOTER_RE.search(joined_row):
                flush_current()
                prev_top = row_top
                continue
            if gap > 25:
                flush_current()
                prev_top = row_top
                continue
            if current and narr_text and len(outside_narr) <= 1:
                current["description"] = f"{current['description']} {narr_text}".strip()
            prev_top = row_top
            continue

        flush_current()

        if not narr_text:
            continue

        debit = best_amount(row, db_lo, db_hi, spec["debit_cx"])
        credit = best_amount(row, cr_lo, cr_hi, spec["credit_cx"])
        if debit <= 0 and credit <= 0:
            continue
        if debit > 0 and credit > 0:
            continue

        current = {
            "date":        date_str,
            "description": narr_text,
            "reference":   " ".join(w["text"] for w in row if r_lo <= mid(w) <= r_hi).strip(),
            "amount":      round(debit if debit > 0 else credit, 2),
            "type":        "debit" if debit > 0 else "credit",
        }
        prev_top = row_top

    flush_current()
    return transactions, spec


# ── Existing table-based extraction (kept for banks that do have row lines) ─

def is_header_row(row: list) -> bool:
    text = " ".join(str(c or "") for c in row).lower()
    return (
        ("withdrawal" in text or "debit" in text)
        and ("deposit" in text or "credit" in text)
        and ("date" in text or "narration" in text)
    )


def find_column_indices(header_row: list) -> dict:
    cols: dict = {}
    for i, cell in enumerate(header_row or []):
        text = str(cell or "").lower().strip()
        if "date" not in cols and "date" in text and "value" not in text:
            cols["date"] = i
        if "narration" not in cols and any(k in text for k in ["narration", "description", "particulars"]):
            cols["narration"] = i
        if "ref" not in cols and any(k in text for k in ["chq/ref", "chq", "refno", "ref.no", "ref no", "reference", "utr"]):
            cols["ref"] = i
        if "withdrawal" not in cols and any(k in text for k in ["withdrawal", "debit", "dr"]):
            cols["withdrawal"] = i
        if "deposit" not in cols and any(k in text for k in ["deposit", "credit", "cr"]):
            cols["deposit"] = i
    return cols


def extract_from_table(table: list) -> list:
    if not table:
        return []

    col_idx: dict = {}
    start_row = 0
    for i, row in enumerate(table):
        if row and is_header_row(row):
            col_idx = find_column_indices(row)
            start_row = i + 1
            break

    if not col_idx:
        col_idx = {"date": 0, "narration": 1, "withdrawal": 4, "deposit": 5}

    date_col = col_idx.get("date", 0)
    narr_col = col_idx.get("narration", 1)
    wdl_col  = col_idx.get("withdrawal", 4)
    ref_col  = col_idx.get("ref", 2)
    dep_col  = col_idx.get("deposit", 5)

    transactions = []
    for row in table[start_row:]:
        if not row:
            continue

        date_str = parse_date(str(row[date_col] if date_col < len(row) else ""))
        if not date_str:
            continue

        narration  = str(row[narr_col] if narr_col < len(row) else "").replace("\n", " ").strip()
        withdrawal = parse_amount(str(row[wdl_col] if wdl_col < len(row) else ""))
        deposit    = parse_amount(str(row[dep_col] if dep_col < len(row) else ""))

        if withdrawal <= 0 and deposit <= 0:
            continue
        if withdrawal > 0 and deposit > 0:
            continue
        if not narration:
            continue

        transactions.append({
            "date":        date_str,
            "description": narration,
            "reference":   str(row[ref_col] if ref_col < len(row) else "").replace("\n", " ").strip(),
            "amount":      round(withdrawal if withdrawal > 0 else deposit, 2),
            "type":        "debit" if withdrawal > 0 else "credit",
        })

    return transactions


def extract_transactions(content: bytes, password: Optional[str]) -> list:
    open_kwargs = {"password": password} if password else {}
    all_transactions: list = []
    word_spec: Optional[dict] = None

    with pdfplumber.open(io.BytesIO(content), **open_kwargs) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            # Strategy 1: line-based table extraction
            line_tables = page.extract_tables({
                "vertical_strategy":   "lines",
                "horizontal_strategy": "lines",
            })
            table_txns: list = []
            for table in (line_tables or []):
                table_txns.extend(extract_from_table(table))

            # Strategy 2: word-coordinate extraction — handles HDFC-style merged rows
            # Reuse previously detected column positions on later pages that do not
            # repeat the transaction table header.
            word_txns, word_spec = extract_by_words(page, word_spec)
            page_txns = word_txns if len(word_txns) > len(table_txns) else table_txns

            # Strategy 3: text-based table extraction as last resort
            if not page_txns:
                text_tables = page.extract_tables({
                    "vertical_strategy":   "text",
                    "horizontal_strategy": "text",
                })
                for table in (text_tables or []):
                    page_txns.extend(extract_from_table(table))

            # Keep parser order stable and carry page/sequence metadata so identical
            # legitimate transactions do not collapse into one row.
            for seq, t in enumerate(page_txns):
                all_transactions.append({**t, "_page": page_idx, "_seq": seq})

    seen_ref_keys: set = set()
    unique: list = []
    for t in all_transactions:
        ref = str(t.get("reference") or "").strip()
        if ref:
            key = f"{t['_page']}|{t['date']}|{t['amount']}|{t['type']}|{ref}"
            if key in seen_ref_keys:
                continue
            seen_ref_keys.add(key)
        unique.append({k: v for k, v in t.items() if k not in {"_page", "_seq"}})

    return unique


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse-pdf")
async def parse_pdf(req: ParseRequest):
    logger.info("parse_pdf: received request, payload_len=%d", len(req.pdf_base64))
    try:
        content = base64.b64decode(req.pdf_base64)
    except Exception:
        logger.warning("parse_pdf: invalid base64 data")
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    if not content:
        logger.warning("parse_pdf: empty file after decode")
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )

    # Check page count before full parse
    try:
        with pdfplumber.open(io.BytesIO(content), password=req.password) as pdf:
            page_count = len(pdf.pages)
        if page_count > MAX_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {page_count} pages. Maximum allowed is {MAX_PAGES} pages.",
            )
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["password", "incorrect", "encrypt", "pkcs"]):
            raise HTTPException(status_code=422, detail="needsPassword")
        logger.error("parse_pdf: failed to open PDF", exc_info=True)
        raise HTTPException(status_code=400, detail="Could not open PDF. File may be corrupted.")

    try:
        transactions = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, extract_transactions, content, req.password),
            timeout=PARSE_TIMEOUT,
        )
        logger.info("parse_pdf: extracted %d transactions", len(transactions))
        return {"transactions": transactions}
    except asyncio.TimeoutError:
        logger.warning("parse_pdf: timed out after %ds", PARSE_TIMEOUT)
        raise HTTPException(
            status_code=408,
            detail=f"PDF parsing timed out after {PARSE_TIMEOUT}s. Try a smaller file.",
        )
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["password", "incorrect", "encrypt", "pkcs"]):
            if req.password:
                raise HTTPException(status_code=400, detail="Incorrect password. Please try again.")
            raise HTTPException(status_code=422, detail="needsPassword")
        logger.error("parse_pdf: extraction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="PDF parsing failed. Please try a different file.")
