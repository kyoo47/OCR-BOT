import subprocess, time
from datetime import datetime, timedelta

# :11 and :41 each hour, 10:00–21:59, plus 22:11
TARGETS = [(h, 11) for h in range(10, 23)] + [(h, 41) for h in range(10, 22)]
TARGETS.sort()
CMD = ["py", "screenshot_ocr.py"]
LOG = "schedule.log"

def next_target(now: datetime) -> datetime:
    for h, m in TARGETS:
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t > now:
            return t
    # none left today → first target tomorrow
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=10, minute=11, second=0, microsecond=0)

def run_once():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        p = subprocess.run(CMD, capture_output=True, text=True, timeout=180)
        line = f"[{ts}] exit={p.returncode}  {p.stdout.strip()}"
        if p.stderr:
            line += f"  [stderr] {p.stderr.strip()}"
    except Exception as e:
        line = f"[{ts}] ERROR: {e}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)

def main():
    print("OCR scheduler (precise) started. Ctrl+C to stop.")
    while True:
        now = datetime.now()
        tgt = next_target(now)
        # sleep until ~0.5s before target
        wait = (tgt - now).total_seconds()
        if wait > 0.6:
            time.sleep(wait - 0.5)
        # short, accurate wait into the target second
        while datetime.now() < tgt:
            time.sleep(0.01)
        run_once()

if __name__ == "__main__":
    main()
