import time
import sqlite3
import csv
from dataclasses import dataclass

DB_PATH = "/Users/jungtaeuk/ships/ships.db"
OUT_CSV = "./pixel_overlap-allITPL.csv"


DLON = 0.003
DLAT = 0.003

sql = f"""
    SELECT Date, Time,
        CAST(Lon / {DLON} AS INT) AS px,
        CAST(Lat / {DLAT} AS INT) AS py,
        COUNT(*) AS n,
        group_concat(MMSI) AS mmsis
    FROM ships_dynamic
        GROUP BY Date, Time, px, py
        HAVING n >= 2
        ORDER BY Date, Time, n DESC;
    """


@dataclass
class Progress:
    label: str
    report_every_s: float = 2.0
    t0: float = 0.0
    last_t: float = 0.0
    n: int = 0
    total: int | None = None

    def start(self, total: int | None = None):
        self.t0 = time.time()
        self.last_t = self.t0
        self.n = 0
        self.total = total
        self.step(inc=0, extra="START", force=True)

    def step(self, inc: int = 1, *, key=None, extra: str = "", force: bool = False):
        self.n += inc
        now = time.time()
        if force or (now - self.last_t) >= self.report_every_s:
            self._print(now=now, key=key, extra=extra)
            self.last_t = now

    def done(self, *, extra: str = "DONE"):
        self.step(inc=0, extra=extra, force=True)

    def _print(self, now: float | None = None, *, key=None, extra: str = ""):
        if now is None:
            now = time.time()
        elapsed = now - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0.0

        if self.total:
            pct = (self.n / self.total) * 100.0
            remain = self.total - self.n
            eta = (remain / rate) if rate > 0 else float("inf")
            eta_str = f"{eta:,.1f}s" if eta != float("inf") else "inf"
            msg = (
                f"[{self.label}] {pct:6.2f}% ({self.n}/{self.total}) | "
                f"{rate:,.1f}/s | elapsed={elapsed:,.1f}s | ETA={eta_str}"
            )
        else:
            msg = f"[{self.label}] n={self.n:,} | {rate:,.1f}/s | elapsed={elapsed:,.1f}s"

        if key is not None:
            msg += f" | key={key}"
        if extra:
            msg += f" | {extra}"

        print(msg, flush=True)


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000;")

    cur = conn.execute(sql)

    prog = Progress("write-rows", report_every_s=2.0)
    prog.start(total=None)  # 결과 row 수를 미리 모르니 total=None

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Time", "px", "py", "count", "mmsis"])

        for row in cur:
            w.writerow(row)
            prog.step(key=(row[0], row[1]))

    prog.done(extra=f"WROTE {prog.n:,} rows")
    conn.close()
    print(f"[DONE] wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()