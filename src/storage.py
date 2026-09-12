"""Local diagnosis history and a queue for cases the model declined to call.

Two tables. `diagnoses` is a record of every prediction, confident or not — it
is what lets someone notice the same disease recurring in the same field across
a season. `review_queue` holds only the abstained cases, which is where "refer
to an expert" in the interface actually leads: without this, that phrase points
at nothing.

SQLite because this is a single-user local tool. No server, no schema
migrations to manage, one file that can be copied or deleted.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnoses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    crop          TEXT NOT NULL,
    field_name    TEXT,
    label         TEXT,              -- NULL when abstained
    display_name  TEXT NOT NULL,
    confidence    REAL NOT NULL,
    threshold     REAL NOT NULL,
    abstained     INTEGER NOT NULL,  -- 0 or 1
    top_class     TEXT NOT NULL,     -- the suppressed guess even when abstained
    all_scores    TEXT NOT NULL,     -- JSON: {display_name: probability}
    image_path    TEXT               -- copy of the photo, for the review queue
);

CREATE TABLE IF NOT EXISTS review_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id  INTEGER NOT NULL REFERENCES diagnoses(id),
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | labelled | dismissed
    true_label    TEXT,              -- filled in when status = labelled
    reviewed_at   TEXT,
    reviewer_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_crop ON diagnoses(crop);
CREATE INDEX IF NOT EXISTS idx_diagnoses_field ON diagnoses(crop, field_name);
CREATE INDEX IF NOT EXISTS idx_queue_status ON review_queue(status);
"""


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


@dataclass
class DiagnosisRecord:
    id: int
    created_at: str
    crop: str
    field_name: Optional[str]
    label: Optional[str]
    display_name: str
    confidence: float
    threshold: float
    abstained: bool
    top_class: str
    all_scores: dict
    image_path: Optional[str]


def log_diagnosis(result: dict, field_name: Optional[str] = None,
                  image_path: Optional[str] = None) -> int:
    """Record one prediction. Returns the row id.

    If the prediction was abstained, it is also enqueued for review — that queue
    is the only place an abstained case goes, so logging and enqueueing happen
    together rather than as a separate step a caller could forget.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO diagnoses
               (created_at, crop, field_name, label, display_name, confidence,
                threshold, abstained, top_class, all_scores, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, result["crop"], field_name, result["label"],
             result["display_name"], result["confidence"], result["threshold"],
             int(result["abstained"]), result["top_class"],
             json.dumps(result["all_scores"]), image_path))
        diagnosis_id = cur.lastrowid

        if result["abstained"]:
            conn.execute(
                "INSERT INTO review_queue (diagnosis_id, status) VALUES (?, 'pending')",
                (diagnosis_id,))

        return diagnosis_id


def _row_to_record(row: sqlite3.Row) -> DiagnosisRecord:
    return DiagnosisRecord(
        id=row["id"], created_at=row["created_at"], crop=row["crop"],
        field_name=row["field_name"], label=row["label"],
        display_name=row["display_name"], confidence=row["confidence"],
        threshold=row["threshold"], abstained=bool(row["abstained"]),
        top_class=row["top_class"], all_scores=json.loads(row["all_scores"]),
        image_path=row["image_path"])


def history(crop: Optional[str] = None, field_name: Optional[str] = None,
           limit: int = 100) -> List[DiagnosisRecord]:
    """Most recent diagnoses, optionally filtered to one crop or field."""
    query = "SELECT * FROM diagnoses"
    clauses, params = [], []
    if crop:
        clauses.append("crop = ?")
        params.append(crop)
    if field_name:
        clauses.append("field_name = ?")
        params.append(field_name)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect() as conn:
        return [_row_to_record(r) for r in conn.execute(query, params)]


def field_trend(crop: str, field_name: str) -> dict:
    """Counts per condition over time for one field — the thing a single photo
    cannot show: is this recurring, and is it getting worse."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT display_name, COUNT(*) as n,
                      MIN(created_at) as first_seen, MAX(created_at) as last_seen
               FROM diagnoses
               WHERE crop = ? AND field_name = ? AND abstained = 0
               GROUP BY display_name
               ORDER BY n DESC""",
            (crop, field_name)).fetchall()
    return {r["display_name"]: {"count": r["n"], "first_seen": r["first_seen"],
                                "last_seen": r["last_seen"]} for r in rows}


def pending_review(crop: Optional[str] = None) -> List[dict]:
    """Abstained cases waiting for an expert to label them."""
    query = """SELECT rq.id as queue_id, d.*
               FROM review_queue rq JOIN diagnoses d ON d.id = rq.diagnosis_id
               WHERE rq.status = 'pending'"""
    params = []
    if crop:
        query += " AND d.crop = ?"
        params.append(crop)
    query += " ORDER BY d.created_at ASC"

    with connect() as conn:
        return [dict(r) for r in conn.execute(query, params)]


def label_review(queue_id: int, true_label: str, note: str = "") -> None:
    """An expert names what the abstained case actually was.

    This updates both tables. review_queue records the review itself; diagnoses
    is corrected to the expert's label, because once a human has spoken, the
    model's original uncertain guess is no longer the record worth keeping —
    the history view should show the resolved answer, not stay stuck on
    "uncertain" forever.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        row = conn.execute(
            "SELECT diagnosis_id FROM review_queue WHERE id = ?", (queue_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No review_queue row with id={queue_id}")
        diagnosis_id = row["diagnosis_id"]

        conn.execute(
            """UPDATE review_queue
               SET status = 'labelled', true_label = ?, reviewed_at = ?, reviewer_note = ?
               WHERE id = ?""",
            (true_label, now, note, queue_id))

        conn.execute(
            """UPDATE diagnoses
               SET label = ?, abstained = 0
               WHERE id = ?""",
            (true_label, diagnosis_id))

def dismiss_review(queue_id: int, note: str = "") -> None:
    """The case wasn't usable — bad photo, wrong crop, duplicate. Removes it from
    the queue without producing a label."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            """UPDATE review_queue
               SET status = 'dismissed', reviewed_at = ?, reviewer_note = ?
               WHERE id = ?""",
            (note, now, queue_id))


def stats() -> dict:
    """Rough numbers for a dashboard: totals, abstain rate, queue backlog."""
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) n FROM diagnoses").fetchone()["n"]
        abstained = conn.execute(
            "SELECT COUNT(*) n FROM diagnoses WHERE abstained = 1").fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) n FROM review_queue WHERE status = 'pending'"
        ).fetchone()["n"]
        by_crop = conn.execute(
            "SELECT crop, COUNT(*) n FROM diagnoses GROUP BY crop").fetchall()

    return {
        "total_diagnoses": total,
        "abstained": abstained,
        "abstain_rate": abstained / total if total else 0.0,
        "pending_review": pending,
        "by_crop": {r["crop"]: r["n"] for r in by_crop},
    }


init_db()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        s = stats()
        print(f"total diagnoses : {s['total_diagnoses']}")
        print(f"abstained       : {s['abstained']} ({s['abstain_rate']:.1%})")
        print(f"pending review  : {s['pending_review']}")
        print("by crop:")
        for crop, n in s["by_crop"].items():
            print(f"  {crop:<10} {n}")
        print(f"\nusage: python src/storage.py history [crop]")
        print(f"       python src/storage.py queue [crop]")
        sys.exit(0)

    cmd = sys.argv[1]
    crop_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "history":
        for r in history(crop=crop_arg, limit=20):
            tag = "ABSTAINED" if r.abstained else f"{r.confidence:.0%}"
            print(f"{r.created_at}  {r.crop:<8} {r.display_name:<26} {tag}"
                  + (f"  [{r.field_name}]" if r.field_name else ""))
    elif cmd == "queue":
        for row in pending_review(crop=crop_arg):
            print(f"#{row['queue_id']:<4} {row['created_at']}  {row['crop']:<8} "
                  f"closest: {row['display_name']} at {row['confidence']:.0%}")
    else:
        print(f"unknown command {cmd!r}")