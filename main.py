from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import os
import httpx
import re
import json
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_eR4NWqofuL8t@ep-curly-frog-a1demxkm-pooler.ap-southeast-1.aws.neon.tech/neondb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Strip sslmode from URL for asyncpg (we pass ssl="require" separately)
_db_url = DATABASE_URL.split("?")[0]
if not _db_url.startswith("postgresql://"):
    _db_url = _db_url.replace("postgresql+asyncpg://", "postgresql://")

db_pool = None

sample_docs = [
    {"name": "Q1_2026_Invoice_Acme.pdf", "file_type": "pdf", "category": "invoice", "file_size_kb": 124,
     "summary": "Invoice from Acme Corp for software development services totaling $12,500 for Q1 2026.",
     "tags": ["invoice", "acme", "q1-2026", "software"],
     "extracted_text": "INVOICE #2026-001\nAcme Corporation\nDate: January 15, 2026\nAmount Due: $12,500\nServices: Software Development Q1 2026"},

    {"name": "Service_Agreement_TechCorp.docx", "file_type": "docx", "category": "contract", "file_size_kb": 87,
     "summary": "Service agreement between TechCorp and IntelliForge for 12-month AI consulting engagement.",
     "tags": ["contract", "techcorp", "consulting", "2026"],
     "extracted_text": "SERVICE AGREEMENT\nParties: TechCorp Inc and IntelliForge Digital\nDuration: January 2026 - December 2026\nValue: $48,000 annually"},

    {"name": "AWS_Receipt_Feb2026.pdf", "file_type": "pdf", "category": "receipt", "file_size_kb": 45,
     "summary": "AWS cloud infrastructure receipt for February 2026 totaling $847.32.",
     "tags": ["receipt", "aws", "cloud", "infrastructure"],
     "extracted_text": "Amazon Web Services\nReceipt Date: March 1, 2026\nBilling Period: February 2026\nTotal: $847.32"},

    {"name": "Annual_Report_2025.pdf", "file_type": "pdf", "category": "report", "file_size_kb": 2048,
     "summary": "Annual business performance report for FY2025 covering revenue, expenses and growth metrics.",
     "tags": ["report", "annual", "2025", "finance"],
     "extracted_text": "Annual Report FY2025\nTotal Revenue: $234,500\nTotal Expenses: $156,200\nNet Profit: $78,300\nYoY Growth: 34%"},

    {"name": "NDA_Freelancer_Ravi.pdf", "file_type": "pdf", "category": "contract", "file_size_kb": 62,
     "summary": "Non-disclosure agreement with freelance developer Ravi Kumar for a 6-month engagement.",
     "tags": ["nda", "contract", "freelancer", "confidential"],
     "extracted_text": "NON-DISCLOSURE AGREEMENT\nParty: Ravi Kumar (Freelancer)\nEffective: February 1, 2026\nDuration: 6 months"},

    {"name": "Office_Rent_Invoice_March.pdf", "file_type": "pdf", "category": "invoice", "file_size_kb": 38,
     "summary": "Monthly office space rental invoice for March 2026 from Regus coworking.",
     "tags": ["invoice", "rent", "office", "march-2026"],
     "extracted_text": "Regus Business Centers\nInvoice for: Office Space - Bangalore\nMonth: March 2026\nAmount: \u20b945,000"},

    {"name": "Tax_Filing_FY2025.pdf", "file_type": "pdf", "category": "report", "file_size_kb": 512,
     "summary": "Income tax filing documentation for FY2025 including all deductions and final assessment.",
     "tags": ["tax", "filing", "2025", "compliance"],
     "extracted_text": "Income Tax Return FY2025\nAssessee: IntelliForge Digital\nGross Income: \u20b928,40,000\nTax Payable: \u20b95,84,000"},

    {"name": "Software_License_JetBrains.txt", "file_type": "txt", "category": "other", "file_size_kb": 12,
     "summary": "JetBrains software license agreement for IntelliJ IDEA and suite of developer tools.",
     "tags": ["license", "software", "jetbrains", "developer"],
     "extracted_text": "JetBrains License Agreement\nProduct: All Products Pack\nLicensee: IntelliForge Digital\nExpiry: December 31, 2026"},
]


async def analyze_document(text: str, filename: str) -> dict:
    """Gemini -> OpenRouter -> mock fallback"""
    prompt = f"""Analyze this document and return JSON:
{{
  "category": "invoice|contract|receipt|report|other",
  "summary": "2-3 sentence summary",
  "tags": ["tag1", "tag2", "tag3"],
  "key_insights": {{
    "key_dates": "any dates mentioned",
    "parties": "people/companies mentioned",
    "amounts": "any monetary amounts",
    "action_items": "any required actions"
  }}
}}

Document name: {filename}
Content: {text[:2000]}"""

    # Try Gemini first
    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=15
                )
                if resp.status_code == 200:
                    text_resp = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    match = re.search(r'\{.*\}', text_resp, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        result["_source"] = "gemini"
                        return result
        except Exception:
            pass

    # OpenRouter fallback
    if OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={"model": "anthropic/claude-3-haiku", "messages": [{"role": "user", "content": prompt}]},
                    timeout=15
                )
                if resp.status_code == 200:
                    text_resp = resp.json()["choices"][0]["message"]["content"]
                    match = re.search(r'\{.*\}', text_resp, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        result["_source"] = "openrouter"
                        return result
        except Exception:
            pass

    # Mock fallback
    return {
        "category": "other",
        "summary": f"Document '{filename}' has been processed and stored securely.",
        "tags": ["document", "processed"],
        "key_insights": {
            "key_dates": "Not detected",
            "parties": "Not detected",
            "amounts": "Not detected",
            "action_items": "Review document contents"
        },
        "_source": "mock"
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(_db_url, ssl="require", min_size=1, max_size=5)

    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE SCHEMA IF NOT EXISTS papersafe;

            CREATE TABLE IF NOT EXISTS papersafe.documents (
              id SERIAL PRIMARY KEY,
              name VARCHAR(500) NOT NULL,
              file_type VARCHAR(50),
              category VARCHAR(100),
              summary TEXT,
              extracted_text TEXT,
              tags TEXT[],
              file_size_kb INTEGER,
              status VARCHAR(50) DEFAULT 'processed',
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS papersafe.document_insights (
              id SERIAL PRIMARY KEY,
              document_id INTEGER REFERENCES papersafe.documents(id),
              insight_type VARCHAR(100),
              content TEXT,
              created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Check if we need to seed
        count = await conn.fetchval("SELECT COUNT(*) FROM papersafe.documents")
        if count == 0:
            for doc in sample_docs:
                doc_id = await conn.fetchval("""
                    INSERT INTO papersafe.documents (name, file_type, category, summary, extracted_text, tags, file_size_kb, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 'processed')
                    RETURNING id
                """, doc["name"], doc["file_type"], doc["category"], doc["summary"],
                    doc["extracted_text"], doc["tags"], doc["file_size_kb"])

                # Also analyze and create insights
                analysis = await analyze_document(doc["extracted_text"], doc["name"])
                insights = analysis.get("key_insights", {})
                for insight_type, content in insights.items():
                    await conn.execute("""
                        INSERT INTO papersafe.document_insights (document_id, insight_type, content)
                        VALUES ($1, $2, $3)
                    """, doc_id, insight_type, str(content))

    yield

    await db_pool.close()


app = FastAPI(title="PaperSafe API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DocumentCreate(BaseModel):
    name: str
    file_type: str = "txt"
    extracted_text: str = ""
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    file_size_kb: Optional[int] = None


class AnalyzeRequest(BaseModel):
    text: str
    filename: str = "document.txt"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "papersafe-api"}


@app.get("/api/documents")
async def list_documents():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, file_type, category, summary, tags, file_size_kb, status, created_at, updated_at
            FROM papersafe.documents
            ORDER BY created_at DESC
        """)
        return [dict(r) for r in rows]


@app.post("/api/documents")
async def create_document(doc: DocumentCreate):
    # AI analyze the text
    analysis = await analyze_document(doc.extracted_text or doc.name, doc.name)

    category = doc.category or analysis.get("category", "other")
    summary = analysis.get("summary", "")
    tags = doc.tags or analysis.get("tags", [])

    async with db_pool.acquire() as conn:
        doc_id = await conn.fetchval("""
            INSERT INTO papersafe.documents (name, file_type, category, summary, extracted_text, tags, file_size_kb, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'processed')
            RETURNING id
        """, doc.name, doc.file_type, category, summary, doc.extracted_text, tags, doc.file_size_kb)

        insights = analysis.get("key_insights", {})
        for insight_type, content in insights.items():
            await conn.execute("""
                INSERT INTO papersafe.document_insights (document_id, insight_type, content)
                VALUES ($1, $2, $3)
            """, doc_id, insight_type, str(content))

        row = await conn.fetchrow("""
            SELECT id, name, file_type, category, summary, tags, file_size_kb, status, created_at, updated_at
            FROM papersafe.documents WHERE id = $1
        """, doc_id)
        return dict(row)


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, name, file_type, category, summary, extracted_text, tags, file_size_kb, status, created_at, updated_at
            FROM papersafe.documents WHERE id = $1
        """, doc_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        insights = await conn.fetch("""
            SELECT id, insight_type, content, created_at
            FROM papersafe.document_insights WHERE document_id = $1
        """, doc_id)

        result = dict(row)
        result["insights"] = [dict(i) for i in insights]
        return result


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM papersafe.documents WHERE id = $1", doc_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        await conn.execute("DELETE FROM papersafe.document_insights WHERE document_id = $1", doc_id)
        await conn.execute("DELETE FROM papersafe.documents WHERE id = $1", doc_id)
        return {"message": "Document deleted"}


@app.post("/api/documents/{doc_id}/analyze")
async def reanalyze_document(doc_id: int):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, name, extracted_text FROM papersafe.documents WHERE id = $1
        """, doc_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        analysis = await analyze_document(row["extracted_text"] or row["name"], row["name"])

        await conn.execute("""
            UPDATE papersafe.documents
            SET category = $1, summary = $2, tags = $3, status = 'processed', updated_at = NOW()
            WHERE id = $4
        """, analysis.get("category", "other"), analysis.get("summary", ""), analysis.get("tags", []), doc_id)

        await conn.execute("DELETE FROM papersafe.document_insights WHERE document_id = $1", doc_id)
        insights = analysis.get("key_insights", {})
        for insight_type, content in insights.items():
            await conn.execute("""
                INSERT INTO papersafe.document_insights (document_id, insight_type, content)
                VALUES ($1, $2, $3)
            """, doc_id, insight_type, str(content))

        updated = await conn.fetchrow("""
            SELECT id, name, file_type, category, summary, tags, file_size_kb, status, created_at, updated_at
            FROM papersafe.documents WHERE id = $1
        """, doc_id)
        return {**dict(updated), "analysis": analysis}


@app.get("/api/stats")
async def get_stats():
    async with db_pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM papersafe.documents")
        by_category = await conn.fetch("""
            SELECT category, COUNT(*) as count FROM papersafe.documents GROUP BY category
        """)
        by_type = await conn.fetch("""
            SELECT file_type, COUNT(*) as count FROM papersafe.documents GROUP BY file_type
        """)
        return {
            "total_docs": total,
            "by_category": {r["category"]: r["count"] for r in by_category},
            "by_type": {r["file_type"]: r["count"] for r in by_type},
        }


@app.post("/api/analyze")
async def analyze_text(req: AnalyzeRequest):
    result = await analyze_document(req.text, req.filename)
    return result
