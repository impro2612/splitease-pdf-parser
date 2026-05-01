import base64
import io
import re
from typing import Optional

import pdfplumber
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SplitEase PDF Parser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ParseRequest(BaseModel):
    pdf_base64: str
    password: Optional[str] = None


def parse_amount(text) -> float:
    if not text:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", str(text).replace(",", ""))
    if not cleaned:
        return 0.0
    try:
        val = float(cleaned)
        return val if val > 0 else 0.0
    except ValueError:
        return 0.0


def parse_date(text) -> str:
    if not text:
        return ""
    m = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", str(text).strip())
    if not m:
        return ""
    d, mo, y = m.group(1), m.group(2), m.group(3)
    year = f"20{y}" if len(y) == 2 else y
    try:
        from datetime import datetime
        datetime(int(year), int(mo), int(d))
    except ValueError:
        return ""
    return f"{year}-{mo.zfill(2)}-{d.zfill(2)}"


def find_column_indices(header_row: list) -> dict:
    cols = {}
    for i, cell in enumerate(header_row or []):
        text = str(cell or "").lower().strip()
        if "date" not in cols and "date" in text and "value" not in text:
            cols["date"] = i
        if "narration" not in cols and any(k in text for k in ["narration", "description", "particulars"]):
            cols["narration"] = i
        if "withdrawal" not in cols and any(k in text for k in ["withdrawal", "debit", "dr"]):
            cols["withdrawal"] = i
        if "deposit" not in cols and any(k in text for k in ["deposit", "credit", "cr"]):
            cols["deposit"] = i
    return cols


def is_header_row(row: list) -> bool:
    text = " ".join(str(c or "") for c in row).lower()
    return (
        ("withdrawal" in text or "debit" in text)
        and ("deposit" in text or "credit" in text)
        and ("date" in text or "narration" in text)
    )


def extract_from_table(table: list) -> list:
    if not table:
        return []

    col_idx = {}
    start_row = 0
    for i, row in enumerate(table):
        if row and is_header_row(row):
            col_idx = find_column_indices(row)
            start_row = i + 1
            break

    # Fallback: HDFC layout Date|Narration|Ref|ValueDt|Withdrawal|Deposit|Balance
    if not col_idx:
        col_idx = {"date": 0, "narration": 1, "withdrawal": 4, "deposit": 5}

    date_col = col_idx.get("date", 0)
    narr_col = col_idx.get("narration", 1)
    wdl_col  = col_idx.get("withdrawal", 4)
    dep_col  = col_idx.get("deposit", 5)

    transactions = []

    for row in table[start_row:]:
        if not row:
            continue

        date_cell = str(row[date_col] if date_col < len(row) else "")
        date_str  = parse_date(date_cell)
        if not date_str:
            continue

        narration  = str(row[narr_col] if narr_col < len(row) else "").replace("\n", " ").strip()
        withdrawal = parse_amount(row[wdl_col] if wdl_col < len(row) else None)
        deposit    = parse_amount(row[dep_col] if dep_col < len(row) else None)

        if withdrawal <= 0 and deposit <= 0:
            continue
        if withdrawal > 0 and deposit > 0:
            continue
        if not narration:
            continue

        transactions.append({
            "date":        date_str,
            "description": narration,
            "amount":      round(withdrawal if withdrawal > 0 else deposit, 2),
            "type":        "debit" if withdrawal > 0 else "credit",
        })

    return transactions


def extract_transactions(content: bytes, password: Optional[str]) -> list:
    open_kwargs = {"password": password} if password else {}
    transactions = []

    with pdfplumber.open(io.BytesIO(content), **open_kwargs) as pdf:
        for page in pdf.pages:
            # Try line-based table extraction first
            tables = page.extract_tables({
                "vertical_strategy":   "lines",
                "horizontal_strategy": "lines",
            })
            if tables:
                for table in tables:
                    transactions.extend(extract_from_table(table))
            else:
                # Fallback: text-based table extraction
                tables = page.extract_tables({
                    "vertical_strategy":   "text",
                    "horizontal_strategy": "text",
                })
                for table in (tables or []):
                    transactions.extend(extract_from_table(table))

    # Deduplicate
    seen = set()
    unique = []
    for t in transactions:
        key = f"{t['date']}|{t['amount']}|{t['type']}|{t['description'][:40]}"
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse-pdf")
async def parse_pdf(req: ParseRequest):
    try:
        content = base64.b64decode(req.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        transactions = extract_transactions(content, req.password)
        return {"transactions": transactions}
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["password", "incorrect", "encrypt", "pkcs"]):
            if req.password:
                raise HTTPException(status_code=400, detail="Incorrect password. Please try again.")
            raise HTTPException(status_code=422, detail="needsPassword")
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)[:300]}")
