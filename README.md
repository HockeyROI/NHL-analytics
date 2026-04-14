# HockeyROI — NHL Analytics

Data-driven NHL analysis. Goalie performance, referee patterns, 
and the numbers the league won't show you.

📬 [Substack](https://hockeyroi.substack.com) | 
🐦 [X / Twitter](https://x.com/HockeyROI) | 
👾 [Reddit](https://reddit.com/u/HockeyROI)

---

## Goalie Analysis

Shot-level analysis examining save percentage by shot type, location, 
danger zone, distance, shooter handedness, rebounds, and rush shots — 
compared against league-wide benchmarks.

### Published Reports

📝 [Darcy Kuemper's Achilles Heel](https://hockeyroi.substack.com/p/darcy-kuempers-achilles-heel-what)

📝 [I Told You So — Forsberg Analysis](https://hockeyroi.substack.com/p/i-told-you-so-but-not-for-the-reason)

📝 [The Oilers Didn't Listen — Wedgewood & Blackwood](https://hockeyroi.substack.com/p/the-oilers-didnt-listenagain)

### Scripts

| File | Purpose |
|---|---|
| `goalie_analysis.py` | Pulls and cleans shot data by goalie |
| `league_benchmarks.py` | Compares goalie stats to league averages |
| `goalie_analysis_compare.py` | Head to head goalie comparison |
| `rebound_analysis.py` | Rebound and second chance analysis |
| `Kuemper.ipynb` | Kuemper full analysis notebook |

---

## Methodology

All goalie analysis uses NHL shot-level data broken down by:
- Shot type (wrist, snap, backhand, slap)
- Location and distance bands
- Danger zones (high, medium, low)
- Shooter handedness
- Rebound flags
- Rush shot flags

League benchmarks are calculated across all qualifying goalies 
to determine whether weaknesses are goalie-specific or universal.

---

## About HockeyROI

HockeyROI applies data-driven business frameworks to NHL analytics. 
Built by someone with 20+ years of acquisition and analysis experience, 
now pointed at hockey.

Long-term goal: bring rigorous, reproducible analytics to NHL decision-making.
