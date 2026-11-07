# 45-second pitch

Fill the numbers with `python -m src.report --pitch`, which writes `PITCH_SCRIPT.filled.md`.

## Script

Is that flat's rent fair? Tenants and landlords both want to know.

We estimated weekly rent for all 65,777 City of Sydney flats, with and without parking, as at 31 July 2026.

Old rents mislead. Copying them ran ${{lag_low_before}} a week low, because rents kept moving. Our rent index cut that gap to ${{lag_gap_after}}.

On adverts we held back, our typical miss was ${{final_median_error}} a week, and {{final_within_100_pct}}% landed within $100. Microburbs' own figure is $86.

A car space adds about ${{parking_premium}} a week. Each floor up adds about ${{floor_premium}}.

So tenants can check an advert, and landlords can price it fairly.
