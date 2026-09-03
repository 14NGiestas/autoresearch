"""SQLite store: dedupe discovered share links and record fetch results."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS snaps (
                   uuid     TEXT,
                   platform TEXT,
                   source   TEXT,
                   found_at REAL,
                   fetched  INTEGER DEFAULT 0,
                   msg_count INTEGER,
                   error    TEXT,
                   PRIMARY KEY (uuid, platform))"""
        )
        self.db.commit()

    def add(self, uuid: str, platform: str, source: str) -> int:
        uuid = uuid.lower()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO snaps(uuid,platform,source,found_at) VALUES(?,?,?,?)",
            (uuid, platform, source, time.time()),
        )
        n = cur.rowcount
        self.db.commit()
        return n

    def pending(self, limit=None):
        rows = self.db.execute(
            "SELECT uuid,platform FROM snaps WHERE fetched=0"
        ).fetchall()
        return rows[:limit] if limit else rows

    def mark_fetched(self, uuid: str, platform: str, msg_count: int):
        uuid = uuid.lower()
        self.db.execute(
            "UPDATE snaps SET fetched=1,msg_count=? WHERE uuid=? AND platform=?",
            (msg_count, uuid, platform),
        )
        self.db.commit()

    def mark_error(self, uuid: str, platform: str, error: str):
        uuid = uuid.lower()
        # An attempted fetch that errors is considered "done": we don't want to
        # re-download dead/404/500 links on every run. `fetched` is set to 1 and
        # the reason is recorded in `error`; use --reaudit to re-evaluate.
        self.db.execute(
            "UPDATE snaps SET fetched=1, error=? WHERE uuid=? AND platform=?",
            (error[:200], uuid, platform),
        )
        self.db.commit()

    def stats(self):
        by_platform = dict(
            self.db.execute(
                "SELECT platform, COUNT(*) FROM snaps GROUP BY platform"
            ).fetchall()
        )
        total = self.db.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
        fetched = self.db.execute("SELECT COALESCE(SUM(fetched),0) FROM snaps").fetchone()[0]
        return by_platform, total, fetched

    def detailed_stats(self):
        """Per-platform breakdown of tracked / ok / error / pending."""
        rows = self.db.execute(
            "SELECT platform, fetched, (error IS NOT NULL AND error != '') "
            "FROM snaps"
        ).fetchall()
        agg = {}
        for plat, fetched, has_err in rows:
            a = agg.setdefault(plat, {"total": 0, "ok": 0, "error": 0, "pending": 0})
            a["total"] += 1
            if fetched:
                a["error" if has_err else "ok"] += 1
            else:
                a["pending"] += 1
        return agg
