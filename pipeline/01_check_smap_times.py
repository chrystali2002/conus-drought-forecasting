import earthaccess
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from config import BBOX

START_TEST = "2024-01-01"
END_TEST   = "2024-12-31"
TARGET_HOUR = 21

earthaccess.login(strategy="netrc")

results = earthaccess.search_data(
    short_name="SPL4SMGP",
    version="008",
    temporal=(START_TEST, END_TEST),
    bounding_box=BBOX,
)

print(f"Found {len(results)} granules from {START_TEST} to {END_TEST}")

if len(results) == 0:
    raise RuntimeError("No SMAP granules found. Check version, date range, or BBOX.")

def get_datetime(granule):
    t = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.fromisoformat(t.replace("Z", "+00:00"))

print("\nFirst few granule timestamps:")
for g in results[:5]:
    print(get_datetime(g))

START = pd.to_datetime(START_TEST).date()
END   = pd.to_datetime(END_TEST).date()

time_counter = Counter()
date_to_times = defaultdict(list)
selected = []

for g in results:
    dt = get_datetime(g)
    day = dt.date()
    hhmm = dt.strftime("%H:%M")

    time_counter[hhmm] += 1
    date_to_times[day].append(hhmm)

    if dt.hour == TARGET_HOUR and START <= day <= END:
        selected.append(g)

print("\nAvailable SMAP timestamps:")
for hhmm, count in sorted(time_counter.items()):
    print(f"  {hhmm}: {count} files")

unique_days = len(date_to_times)

print(f"\nUnique days represented: {unique_days}")

print("\nTimestamp coverage:")
for hhmm, count in sorted(time_counter.items()):
    pct = count / unique_days * 100
    print(f"  {hhmm}: {count}/{unique_days} days ({pct:.1f}%)")

print(
    f"\nSelected {len(selected)} granules at "
    f"{TARGET_HOUR:02d}:00 UTC within {START_TEST}–{END_TEST}"
)

print("\nExample days and available times:")
for day in sorted(date_to_times.keys())[:10]:
    print(f"  {day}: {sorted(date_to_times[day])}")

print("\nDone.")
