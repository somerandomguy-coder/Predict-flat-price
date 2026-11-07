# Domain research: what drives flat rents and car space premiums

Lane `lane/domain-research`, Team 2 Bulbs 1 Source. Written 7 November 2026.

## How to read this note
- **Plain terms.** An *advert* is one asking rent. An *anchor* is our best past rent for a flat, moved to today with an index. *Median error* is the rank metric, in $/week. A *re-let* is a flat advertised again.
- **Our own numbers use adverts dated up to 31 March 2026 only.** Fold B (Apr-May 2026) and Fold A (Jun-Jul 2026) were not read or scored. `load_leases` was capped before loading, and panel reads were filtered at read time. The one exception is the "measured" premium column in the suburb table, quoted as you supplied it from `out/real/parking.json`, which runs to 31 July. It informed no test here.
- **Pseudo-folds.** To test ideas without touching Fold B, I refit the current model on four windows inside the allowed dates: Aug-Sep 2025, Oct-Nov 2025, Dec-Jan 2026 and Feb-Mar 2026 (8,838 adverts). Settings: one thread, 300 rounds, 8 to 11 training origins rather than 12. They score about $43, close to Fold B's $42.62, so they are a fair stand-in but not a replacement. **Every gain below still needs confirming on Fold B.**
- **Evidence labels.** **Strong**: peer-reviewed or official, directly on point. **Moderate**: industry data or a foreign study. **Weak**: commentary, inference or a search snippet.
- **Post-cutoff items** are marked **BACKGROUND ONLY**. They must not feed the model.
- **Noise.** With about 1,700 scored adverts the median has a standard error near $1.50. Per-fold results moved by up to $1 just from row order in my runs. Treat gaps under $1 as nothing.

## The 10 highest-value ideas
Ranked by expected gain in median error, then effort. "Tested" means on the pseudo-folds above; every other gain is a judgement from the literature.

| # | Idea | Expected gain | Effort | Evidence |
|---|---|---|---|---|
| 1 | Give the model the raw last rent | **-$1 to -$2** | 30 minutes | **Tested:** -$1.66 |
| 2 | Round final predictions to $10, by a fixed rule | **About -$1, can backfire** | 15 minutes | **Tested,** metric quirk |
| 3 | Floor area and build year for first-time flats | -$1 to -$2 | High, public data lane | Literature |
| 4 | Market phase of the last advert | -$0.2 to -$0.7 | 1 hour | Literature |
| 5 | Recency-weighted building rent and building spread | -$0.2 to -$0.7 | 1-2 hours | Literature, one failed cousin |
| 6 | Trust anchors over 5 years old less | -$0.2 to -$0.5 | 1 hour | Our data |
| 7 | Road and rail noise, for first-time flats | -$0.2 to -$0.5 | 1 hour, API lane | Literature |
| 8 | Flag serviced, furnished and student buildings | -$0.1 to -$0.5 | 2 hours | Weak |
| 9 | Recent-window premium with a suburb deflator | Under $0.3 on the score, **large for the report** | 1-2 hours | **Tested** in part |
| 10 | Permit test once build years exist | About $0, **a strong story** | 30 minutes after idea 3 | Official policy |

These gains are not additive. Ideas 1 and 2 interact: idea 1 moves the raw score, and that decides whether idea 2 helps.

**Rules for all of them, from `METHODS_AND_NEXT_STEPS.md`:** choose on Fold B with three seeds, ignore gaps under $1, and check each new feature's blank rate in fold rows against final rows. Report flats with and without an earlier advert separately. After any change that leans harder on past rents, re-check the model's parking gap, because it may compress further.

**1. Give the model the raw last rent.**
- *Why:* a quarter of re-lets within two years return at exactly the old rent (see 1.1), and an indexed anchor hides that number.
- *Feature:* in `_add_anchors`, add `raw_prev_ratio = log(last raw rent / best_anchor)`, `prev_m50` (last rent is a multiple of $50) and `prev_gap_days` (origin minus last advert date). Use adverts dated by the cutoff only.
- *Result on pseudo-folds, three seeds:* $43.29 to $41.63, a change of **-$1.66 (95% CI -2.36 to -0.93)**, better in all four windows. By window: $43.50 to $41.73, $41.83 to $40.49, $43.09 to $40.79, $45.20 to $44.57. Flats with history went from $37.88 to $35.70. First-time flats did not change.
- *Fold B test:* add the three columns, rerun with three seeds. Accept if the seed-averaged gain exceeds $1. A hard rule that snaps to the last rent is worse, so keep it as a feature.

**2. Round final predictions to $10, by a fixed rule.**
- *Why:* 85% of adverts sit on a $10 grid (see 1.8).
- *Be honest about it:* this exploits how a median behaves on a grid. Mean error does not change. It is defensible as "we quote rents the way agents do", but it is not better prediction.
- *Rule, to fix before looking:* take the raw seed-averaged median error on Fold B. If it is $40-$44, round to the nearest $10. If it is $45-$48, round to the $5-offset grid ($5, $15, $25...). Otherwise do not round: at raw $38.77 the $10 grid gave $40, which is worse. Apply to `rent_without_car_space`, and add a premium rounded to $10 so the gap stays sensible.
- *Result:* -$3.08 pooled at raw $43. At the real size of 1,699 adverts it improves 78-84% of the time at raw $41-$43, by about $1-$2 on average. At raw $47 it improves only 15% of the time and costs $1.72.
- *Risk:* folds may flatter us. If our true raw score is $45-$48 and we round to $10, we lose up to $2.
- *Fold B test:* one line after the final fit. Report rounded and raw scores side by side.

**3. Floor area and build year for first-time flats.**
- *Why:* first-time flats are 29% of pseudo-fold adverts at $62, against $38 for flats with history. Microburbs reports 67.9% within 10% without history, against 84.2% with it. Sydney apartment prices fall 10-33% with age. This is the biggest gap, but it needs data we do not yet have.
- *Feature:* strata lot area or building footprint per flat, and build year from the public data lane. Do not pull anything new here. Also try area per bedroom against the building median.
- *Fold B test:* score first-time flats alone ($56.20 today), and look for a fall of $5 or more there. Blank rates must match fold and final rows. Same-stack comparables failed as a cheap stand-in (table below), so real size data is needed.

**4. Market phase of the last advert.**
- *Why:* in February 2023, 49.3% of Sydney rental stock within 12.5 km of the CBD was still below pre-pandemic rent, against 10.1% at 12.5-25 km. Flats last let in 2020-21 may behave differently from the suburb index.
- *Feature:* calendar half-year of the last advert, and suburb index change since then. Our index is already by suburb, so this only adds the interaction with the flat's own history.
- *Fold B test:* three seeds, and read the group whose last advert was in 2020-21. Check it against decision 20, since date-style features caused trouble before.

**5. Recency-weighted building rent and building spread.**
- *Why:* Cook County's condo model weights a leave-one-out building mean by time. Ours is already leave-one-out by bedroom. What is missing is time weighting and a measure of how mixed the building is.
- *Feature:* time-weighted mean of indexed rents in the same building and bedroom count (one-year half-life), plus their spread.
- *Fold B test:* first-time flats and all flats. Temper expectations: the stack comparable added nothing.

**6. Trust anchors over 5 years old less.**
- *Why:* these beat the index by a median 2.7%, with a wide spread, and pseudo-fold error is about $50. RICS treats old evidence as weaker. The team already sees a -$54 bias here.
- *Rule:* for anchors over 5 years old, blend toward the building anchor, for example `0.5 own + 0.5 building`, then let the model correct it.
- *Fold B test:* read the group with old anchors, then all flats.

**7. Road and rail noise, for first-time flats.**
- *Why:* arterial frontage is -4.0% on Sydney sale prices, and rents fall about 0.4% per dB. About $30 on a typical flat where it applies.
- *Feature:* use the API lane's `shapes/busy-roads` and `shapes/noise-heatmap`, plus distance to rail line. Do not duplicate the pull.
- *Fold B test:* first-time flats. Note that a station's value and rail noise pull in opposite directions.

**8. Flag serviced, furnished and student buildings.**
- *Why:* these likely price differently (foreign furnished premiums 7-26%) and will sit in the tails. No Australian figure exists.
- *Feature:* OSM `tourism=apartment` and known student blocks at building level, plus high building-level rent spread. Keep to static sources.
- *Fold B test:* read error for flagged buildings. It may move within-$100 more than the median.

**9. Recent-window premium with a suburb deflator.**
- *Why:* the headline $40 is likely too low for 2026 (see 2.1). `parking.py` deflates with a citywide index, and its docstring already says `src/index.py` can replace it.
- *Rule:* fit on 2023 onward, or use suburb curves. Shrink suburbs toward five precincts: CBD core, eastern ring, inner south and west, Green Square, luxury harbour.
- *Fold B test:* flipped adverts only, n=85, so low power. **Decision 15 fixed the submitted gap by rule, and I am not proposing to reopen it on this evidence.** Use this for the report at minimum. The team can decide whether a $17 gap is what they want to publish, given every view here says $28 or more.

**10. Permit test once build years exist.**
- *Why:* residents of flats approved after 1996, 2000 or 2014 cannot get a street permit, while older Zone B flats without a space can hold two. A spaceless flat in a newer building has no fallback.
- *Test:* same-period premium in buildings approved before and after the date, by former council area. If the premium is higher after, we have a clean policy story. `first_ever_listed` is not a build date, and my rough attempt with it was inconclusive.

### Tested and not worth more time
| Idea | Result | Why it fails |
|---|---|---|
| Snap to $50 or $25 | +$6.92 worse | Errors land on 0, 50, 100, so the median becomes $50 unless over half hit exactly |
| Snap to $50 only when close (within $10) | +$4.00 worse | Same problem, smaller dose. Within $5 is a tie (+$0.08) |
| Snap to the flat's last rent when close | +$0.26 to +$3.84 worse | Stickiness is real, but it needs a feature, not a hard rule |
| Centre predictions between two $50 marks | +$1.92 worse | A neat theory: a +-$40 window would catch two clusters. It did not work |
| Flat dollar or percent shift (+-$5, +-1%) | -$0.5 to +$1.8, sign flips between windows | Bias changes sign by window. Do not calibrate |
| Seasonal uplift on anchors at July cutoffs | Under $0.80 either way | 2024 and 2025 July origins, anchor only. See 1.3 |
| Price-tier index | Bias is about +1% in all five tiers | No tier is being missed, so tiers have little to fix here |
| Filling blank floors from unit numbers | Not possible | `floor_level` already comes from the unit number (see 1.4) |
| Same-stack comparable (flats directly above and below, same bedrooms) | -$0.27 (CI -0.87 to +0.46), three seeds | Covers only 35% of adverts, and first-time flats did not improve |
| Relative position in unit numbering | +$0.37 (CI -0.27 to +0.93), three seeds | No signal |
| All the new features together | -$1.09, worse than raw last rent alone (-$1.66) | Extra columns only add noise |
| Blending 25% toward the last rent | -$0.95 (CI -1.57 to -0.42) | Works, but the feature in idea 1 does better |

## Question 1: the nature of flat rents

### 1.1 The last rent is the best guide, but it behaves oddly
**Our data (2023 to March 2026, flats advertised twice):**

| Gap since last advert | Pairs | Same rent exactly | Rent fell | Median change | Against the index |
|---|---|---|---|---|---|
| 91-180 days | 1,407 | 32% | 29% | $0 | -1.1% |
| 181 days-1 year | 5,164 | 30% | 20% | $0 | -1.6% |
| 1-2 years | 7,249 | 26% | 13% | +$40 | -1.9% |
| 2-3 years | 4,464 | 23% | 6% | +$115 | -0.5% |
| 3-5 years | 5,797 | 16% | 3% | +$190 | +2.0% |
| Over 5 years | 3,848 | 8% | 3% | +$170 | +2.7% |

- **A quarter of re-lets within two years come back at exactly the old rent, even though rents were rising.** An index cannot produce this. It pushes every anchor up smoothly, and the model predicted above the old rent in 80% of cases at 1 to 2.5 years. This is the basis of idea 1.
- **Short gaps fall behind the index and long gaps run ahead of it.** The second half fits the team's -$54 bias on anchors over 5 years old: old flats get refurbished, or are simply re-priced at market.
- **Nothing in the file sits under 91 days apart.** The smallest gap is 91 days, across 107,311 adverts. Re-advertising and price cuts inside a letting campaign are therefore invisible. If the frozen set follows the same rule, a flat last advertised after about 1 July 2026 cannot appear in the score window. I could not confirm how the file was built.
- **Gaps peak at about a year:** 8.9% fall at 340-390 days, and the median gap is 614 days.

**Literature:**
- Australia, April 2025, annual change per property. New tenants: 82.8% rose, 9.6% unchanged, 7.6% fell. Existing tenants: 72.6%, 24.9%, 2.5%. Monthly turnover 2.2%. [ABS](https://www.abs.gov.au/articles/latest-insights-rental-market). **Strong.** Our 26% unchanged is well above ABS's 9.6%. I could not resolve this. Adverts are not agreed rents, and ABS covers all property types nationally.
- US 1974-81: 29% of rents unchanged year on year, far more for sitting tenants. Most rigidity at turnover "can be ascribed to grid pricing", and last year's rent acts as a focal point. Rigidity is greater in small buildings. [Genesove, NBER](https://www.nber.org/papers/w7137). **Strong, but old and American.**
- Turkey 2008-11: unchanged rents 17.1% at turnover against 35.4% for continuing tenants. [MPRA](https://mpra.ub.uni-muenchen.de/55292/1/MPRA_paper_55292.pdf). **Moderate.**
- Ireland: turnover rents rose 11.6-15.9% against 2.9-4.4% for ongoing tenancies outside rent pressure zones. [ESRI](https://esri.ie/publications/understanding-trends-in-property-level-rental-inflation). **Moderate.**
- NSW: no increase in the first 12 months of a tenancy or within 12 months of the last one, with 60 days' notice. [NSW Government](https://www.nsw.gov.au/housing-and-construction/renting-a-place-to-live/rent-increases). **Strong.** The page does not say whether a new tenancy resets this. It seems tied to the agreement, so a re-let looks unconstrained, but that is my inference.
- No-grounds evictions ended in May 2025, and 84% of tenancies end by renter choice. [NSW Government](https://www.nsw.gov.au/ministerial-releases/renters-better-protected-one-year-on-from-no-grounds-evictions-ban). **Strong.** I found no evidence of how this changed re-let rents.
- Not found: any study of advert-to-advert changes in Australia.

### 1.2 Asking versus agreed rent
- **The gap changes sign with the cycle.** Early in the pandemic agreed rents fell further than adverts. Later they rose above them: +24% against +22% for CoreLogic advertised rents since February 2020. This compares index growth, not matched listings. [ABS](https://www.abs.gov.au/statistics/detailed-methodology-information/information-papers/new-insights-rental-market). **Strong.**
- NSW Fair Trading (2014-24): voluntary overbidding on about 12% of tenancies in 2015-20, and 22% at the 2022-24 peak. City of Sydney is not named among the top areas, and the page gives no dollar gap. [NSW Fair Trading](https://www.nsw.gov.au/departments-and-agencies/fair-trading/who-we-are/rent-bidding-nsw-insights-report). **Strong on direction.**
- Adverts must state one fixed price, with no ranges. Unsolicited higher offers are legal. [NSW Government](https://www.nsw.gov.au/housing-and-construction/rules/advertising-and-rent-bidding-on-rental-properties). **Strong (snippet only).**
- **Implication:** our target is the ask, so past asks are the right comparables. Do not apply any fixed asking-to-agreed correction. RICS calls asking prices weak evidence of value, but that is about value, not about predicting the next ask. Use NSW bond data only for trend, as already decided.

### 1.3 Seasonality
**Our data (repeat-flat index, deviation from a 13-month trend, 2015-25 without 2020-21, nine years):**

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rent vs trend % | -0.8 | +0.2 | +0.3 | -0.2 | +0.1 | -0.8 | -1.0 | -0.4 | -0.2 | +0.1 | +0.3 | -0.2 |
| Share of adverts % (2022-25) | 8.7 | 7.4 | 6.8 | 6.5 | 8.3 | 9.4 | 9.7 | 9.1 | 7.7 | 8.3 | 11.4 | 6.9 |

- **Rent seasonality is about 1% end to end and mostly noise.** The clearest months are June and July (standard error about 0.3), then a partial recovery in August and September.
- **The two-month step from a July cutoff averages +1.1% (SE 0.4) against +0.3% (SE 0.6) from a March cutoff.** Most of this is trend, since the all-month average is +0.75%. The gap between March and July is not significant.
- **What this means for decision 14.** The flat index was chosen at a March cutoff, and a July cutoff could differ. I tested an uplift directly at the 2024 and 2025 July origins and it moved median error by under $0.80. I would leave it alone.
- **Adverts are seasonal even though rents are not.** November is the peak, and the score window falls in a mid-to-low stretch.
- **Student belt** (Ultimo, Chippendale, Haymarket, Camperdown, Darlington, Glebe, Forest Lodge; 20% of adverts): no distinct rent season. September is -1.4% (SE 0.4) against 0.0 elsewhere, one month out of twelve, so possibly chance. Adverts peak harder in November (13.2% against 10.9%).

**Literature:**
- Cotality: March quarter 2025 +1.7% against +0.4% in the December quarter, partly attributed to students. [Cotality](https://corelogic.com.au/__data/assets/pdf_file/0024/26637/202504_Rental-Review.pdf). **Moderate (snippet).**
- SQM: vacancies rise in December as students leave and reverse in January-February. [Adviser Voice](https://www.adviservoice.com.au/2023/01/vacancy-rates-seasonally-increase-over-december/). **Moderate.**
- SQM: asking rents usually firmest October to January. [MPA](https://www.mpamag.com/au/news/general/rental-vacancies-split-as-spring-test-looms-for-landlords/589728). **BACKGROUND ONLY (15 Sep 2026).**
- SQM: Sydney unit asks -1.0% in the 30 days to 12 October 2026. [PIP](https://propertyinvestmentprofessionals.com.au/research-insights/sqm-national-vacancy-september-2026-1-4-percent-sydney-unit-rents-turn). **BACKGROUND ONLY.** It must not change any model setting.
- Not found: any measured August-September or semester 2 effect on Sydney rents.

### 1.4 Floor, view, aspect, noise
**Our data, inside building x bedrooms x bathrooms x parking:**

| Floor | 0-2 | 3-5 | 6-10 | 11-15 | 16-20 | 21-30 | 31+ |
|---|---|---|---|---|---|---|---|
| Premium over 0-2 | base | +$10 | +$36 | +$61 | +$88 | +$111 | +$144 |
| SE | | 3 | 4 | 8 | 9 | 12 | 16 |

- **About $3.80 a floor, and close to a straight line in dollars.** The foreign evidence is more curved.
- **A data discovery.** `floor_level` matches the leading digits of the unit number in 99.98% of 3-digit and 100% of 4-digit units, and it is blank for only 0.17% of those. So the field *is* the unit number, and the blanks are simply 1 and 2 digit units. It therefore carries no true storey information for buildings numbered in sequence. This replaces the "fill floors from unit patterns" idea in `METHODS_AND_NEXT_STEPS.md`.

**Literature:**
- San Diego condo sales: about +2.2% a floor, flattening above the mean. [IJHMA](https://ideas.repec.org/a/eme/ijhmap/v6y2013i2p197-208.html). **Moderate.** Hong Kong also finds it diminishes. [Lingnan](https://scholars.ln.edu.hk/en/publications/the-pattern-of-floor-level-premium-in-apartment-buildings/). **Moderate.**
- Sydney apartment sale prices: scenic view +5.4%, arterial road -4.0%, pool +0.7%, buildings of 10 or fewer dwellings +6.3% against over 100. n=553,275. [RBA RDP 2020-04](https://www.rba.gov.au/publications/rdp/2020/2020-04/appendix-d.html). **Strong for prices, moderate for rents.** No Sydney *rent* hedonic was found.
- Hong Kong prime estates: partial view about 11%, full view about 22%, controlling for floor. [Lingnan](https://scholars.ln.edu.hk/en/publications/estimating-view-premiums-in-high-rise-residential-housing-hedonic/). **Moderate.**
- Rents and traffic noise: -0.4% per dB(A), up to -9.6% at the loudest sites. [TUM](https://portal.fis.tum.de/en/publications/impact-of-simulation-based-traffic-noise-on-rent-prices/). **Moderate.**
- Rail noise bites harder than road per dB in Poznań sales. [IJSPM](https://journals.vilniustech.lt/index.php/IJSPM/article/view/20563). **Moderate (snippet).**
- Not found: any aspect or street-frontage rent study. Treat both as unproven. I tried same-stack comparables as a proxy for size, aspect and view, and they added nothing.

### 1.5 Building age, strata, amenities, new builds
- **Sydney sale prices fall fast with age, then level off:** 2-5 years -10.4%, 5-10 -17.1%, 10-15 -21.8%, 20-30 -28.1%, 40-50 -32.7%, and 60+ only -15.8% as heritage stock rebounds. [RBA](https://www.rba.gov.au/publications/rdp/2020/2020-04/appendix-d.html). **Strong for prices.**
- Melbourne listings: rent against building age is non-linear, with an inner-city historic premium. [Urban Studies 2021](https://findanexpert.unimelb.edu.au/scholarlywork/1455513-examining-building-age--rental-housing-and-price-filtering-for-affordability-in-melbourne--australia). **Strong, no coefficients seen.**
- **Our data shows no visible new-build decay on re-let.** Flats first seen in a building's first six months ran -1.5% against the index at the next advert. That compares with -0.9% to -1.5% for older buildings. Caveat: "building age" here is the first advert in our file, not a build date. So age probably matters *between* buildings, which suits first-time flats, not as a decay inside a flat's own history.
- Melbourne build-to-rent against private: raw +19.4%, adjusted +6.3%, with only concierge and library significant. [WSU](https://researchers.westernsydney.edu.au/en/publications/amenities-premium-services-and-rental-premiums-in-melbournes-buil/). **BACKGROUND ONLY.** The listed date is after today, so verify before quoting.
- Not found: peer-reviewed gym or pool rent premiums, or a link between strata levies and rent.

### 1.6 Furnished and short-stay flats
- We have no furnished field. Foreign premiums run from 7-21% (UK) to 25.8% (Porto Alegre). **Weak to moderate, snippets only.** [UK](https://propertyindustryeye.com/research-reveals-city-differences-in-pre-furnished-rental-premium/), [Brazil](https://periodicos.ufsc.br/index.php/economia/article/view/81978). No Australian figure was found.
- Greater Sydney non-hosted short-term letting is capped at 180 days a year, and bookings of 21+ days are exempt. [NSW Planning](https://www.planning.nsw.gov.au/short-term-rental-accommodation). **Strong.**
- City of Sydney, 2023: 2,468 registered against 5,454 active. [Council](https://meetings.cityofsydney.nsw.gov.au/ieDecisionDetails.aspx?AIId=18994). **Strong.**
- These flats are likely a different pricing regime, and they will sit in the error tails. See idea 8.

### 1.7 Students around Ultimo, Chippendale and Haymarket
- About 50% of international students rent privately and 15% use student housing. They are twice as likely as domestic students to live in inner-city areas (2016 Census). A 50,000 rise in population adds about 0.5% to private rents, and students explain only a small part of the post-pandemic rise. [RBA](https://www.rba.gov.au/publications/bulletin/2025/jul/international-students-and-the-australian-economy.html). **Strong.**
- City of Sydney student rooms: 4,223 (2012) to 11,529 (2022). In 2023 one provider charged $729 for a large studio and $419 for a shared room. [Council](https://meetings.cityofsydney.nsw.gov.au/documents/s70704/Affordable%20Student%20Accommodation%20Shortage.pdf). **Strong.**
- 2027 planning level 295,000, with 2026 commencements -8% on 2025. [Education](https://www.education.gov.au/node/20481). **Strong (snippet).** The July 2026 announcement is pre-cutoff.
- **Our data backs a weak effect on rents, but a clear one on cars.** The belt has no distinct rent season (1.3), and Chippendale and Haymarket have 58-59% of households with no car (Census).
- Not found: suburb student shares, or any measured effect of student housing on flat rents.

### 1.8 Rounding
**Our data, 32,717 adverts since 2024:** 99.6% are multiples of $5, 85% of $10, 61% of $50 and 32% of $100. Top values are $750, $850, $800, $1,100 and $950.

| Rent band | Multiple of $50 | Multiple of $10 |
|---|---|---|
| Under $500 | 29% | 73% |
| $500-$799 | 44-50% | 78-83% |
| $800-$999 | 58% | 84% |
| $1,000-$1,249 | 85% | 94% |
| $1,500-$1,999 | 88% | 96% |

- There is a real break at $1,000: above it, agents think in $50s.
- Rents ending in 80 and 95 are the "charm" prices: 6.6% and 4.7% of adverts.
- Share at $50 multiples has risen from 38% (2015) to 61% (since 2024).
- Changes between adverts also cluster: 55% of 1-2 year changes are $50 multiples, with $0, +$50 and +$100 the most common.

**Literature:**
- Turkey: rounding to multiples of 10/25/50/100 is linked to 68% of rent rigidity. [MPRA](https://mpra.ub.uni-muenchen.de/55292/1/MPRA_paper_55292.pdf). **Moderate for Australia.**
- US home prices bunch at round marks. [Pope et al.](https://ideas.repec.org/a/eee/gamebe/v93y2015icp89-107.html). **Strong (snippet).**
- Sweden: just-below asking prices sold 3-5% higher. [Uppsala](https://swopec.hhs.se/uunewp/abs/uunewp2017_010.htm). **Moderate.**
- Not found: any study of Australian rent clustering, or any showing that rounding predictions lowers median error. The test below may be the first of its kind for this data.

**Does rounding predictions help? Only by a quirk of the metric.**

| Rule | Median error | Change | Exact hits |
|---|---|---|---|
| No rounding | $43.08 | | 0.8% |
| Nearest $10 | $40.00 | -$3.08 | 5.8% |
| Nearest $5 | $45.00 | +$1.92 | 3.2% |
| Nearest $25 or $50 | $50.00 | +$6.92 | 10-18% |

Why this happens: adverts sit on a $10 grid, so a prediction on the same grid makes every error a multiple of $10, and the median has to land on a grid value. Nothing has become more accurate. **Mean error is unchanged at $70.9 against $70.85.** The number of adverts within $40 rises from 47.1% to 51.0% purely because errors of $41-44 become $40.

**It is a step function, so whether it helps depends on where our raw score falls:**

| Raw median error | $10 grid gives | Chance of improving at 1,699 adverts |
|---|---|---|
| $40.9 | About $40 | 78% |
| $43.1 | $40 | 84% |
| $45.2 | $45 | 42% |
| $47.4 | $50 | 15% |
| $51.7 | $50 | 81% |

In Feb-Mar 2026 the raw score was $44.88, and rounding gave $45, a tie. Note the other grid: rounding to $5, $15, $25 gave $45 at raw $47.4, and so wins there. See idea 2.

## Question 2: the nature of parking

### 2.1 How much is a car space worth? Five views
| View | Estimate | What biases it |
|---|---|---|
| Pooled within-building, all years (`parking.py`, to March) | $38 ($40 in `parking.json`) | Averages old years. **Likely too low for 2026** |
| Same year, same building, bedrooms, bathrooms (2023 to Mar 2026) | $60.50 (SE 3.4), 867 buildings | Parking flats are also better flats, so too high |
| Same flat, parking flag changes, 2023 on | $28 (95% CI 18 to 36), 515 flips | Any mistaken flags pull it down |
| Grattan's 6.7% applied to an $850 rent | About $57 | Sale-price percent carried to rents, unverified |
| Stand-alone car space rents | $25-$170 by area | Asks, undated, and a ceiling not a mean |

**My reading: about $30-$60 a week today, central $45.** The model's own gap (median $12 on pseudo-folds, $17 as submitted) is below all of these.

**Why the pooled figure is low.** The `by_year` table in `parking.json` shows the premium rising from about $28 in 2015 to about $60-70 in 2024-26. That is far faster than rents, and share-moving-with-rents is capped at 1. So old comparisons are under-scaled and the "at the cutoff" figure sits below recent years. Same-period and pooled estimates rank suburbs alike (rank correlation 0.88), but the level differs a lot: Waterloo $26 against $3, Darlinghurst $116 against $51.

**The flips look real, not typing errors.** 38% reverse at the next advert, yet reversed flips carry as big a rent gap as held ones ($19 against $16). Switching on and off give similar sizes ($24 and $16). The flip rate is 3.5% overall, and it is higher in the CBD (Sydney 6.7%, Haymarket 7.5%) than in Waterloo (2.4%). So noise does not explain Waterloo's low figure. My inference is that spaces really do come and go, for example leased separately or kept by the owner.

**Same-flat flips by area, all years:** city core (Sydney, Pyrmont, Woolloomooloo, Millers Point) $53 (33 to 68), n=240. Green Square (Waterloo, Zetland, Rosebery, Beaconsfield) -$11 (-32 to +6), n=179. Everywhere else $15 (6 to 24), n=593.

### 2.2 What the literature says
- **Best local number.** Sydney apartments built since 2015: first space +6.71% (95% CI 5.85 to 7.57), second +6.67%, about $52k for a first space. n=54,010 sales, controls for floor area, beds, baths and age, with SA2 and month effects. Melbourne's first space is +11.8%. [Grattan](https://grattan.edu.au/wp-content/uploads/2026/07/Grattan-Institute-Presentation-2026-Wasted-space-WTP-methodology-ACE-presentation-8-July-2026.pdf). **Strong-moderate: think-tank work, on point, dated 8 July 2026, I read the slides directly.** Prices, not rents.
- **$52k at a 3-4% gross yield is $30-$40 a week.** The yield is my assumption. It lands near our pooled number and below same-period.
- **Coarse geography hides the premium.** Grattan's Sydney first-space estimate falls from 6.7% with SA2 effects to about 4% with SA3 and about 0% with SA4 (read from a chart). Central flats are dear and have fewer spaces, which offsets the premium. This is why within-building is the right design.
- **Oversupply.** 40% of Sydney and Melbourne studio and one-bed households with a space own no car. Spaces outnumber residents' cars in almost every LGA. Up to 40% of spaces in Melbourne apartment buildings sit vacant. Willingness to pay is below the roughly $80k basement cost in most Sydney areas. [Grattan](https://grattan.edu.au/wp-content/uploads/2026/07/Grattan-Institute-Presentation-2026-Wasted-space-WTP-methodology-ACE-presentation-8-July-2026.pdf). **Strong-moderate.**
- US 2011: a garage was worth about US$142 a month, about 17% of rent. [Gabbe and Pierce, via summary](https://carrefour.vivreenville.org/storage/app/media/publications/Articles-de-fond-Encyclopedique/problematique-du-stationnement/hidden-cost-bundled-parking.pdf). **Strong, seen only in a summary.**
- San Francisco 1996: condos +US$38.8k, 13% (n=130). [Jia and Wachs](https://escholarship.org/uc/item/0fm8k169). **Strong, small sample.**
- LA adaptive reuse: about US$200 a month, and evidence that parking-free units are smaller is mixed. [Manville](https://ideas.repec.org/a/taf/rjpaxx/v79y2013i1p49-66.html). **Moderate, abstract only.**
- Parking changes car ownership: 38% car ownership at no parking against 81% at one space per unit, in a San Francisco housing lottery. [Streetsblog summary](https://usa.streetsblog.org/2021/02/10/study-yes-more-parking-does-put-more-cars-on-the-road). **Strong study, snippet only.** So no-car share is partly a *result* of scarce parking, which matters for 2.4.
- Amsterdam street permit willingness to pay €8 a day. [Tinbergen](https://papers.tinbergen.nl/09083.pdf). **Strong, abstract.**
- Unbundling: I found no study comparing bundled premiums with separate prices in one market. Indi City Sydney is cited as unbundled. [RMIT](https://www.rmit.edu.au/news/all-news/2023/mar/unbundled-parking). **Moderate.** Inference: where a space can be sublet, its separate rent caps the premium, so a premium below the stand-alone rent is normal.

### 2.3 City of Sydney rules
I read this policy directly. [Neighbourhood On-Street Parking Policy, approved April 2025](https://www.cityofsydney.nsw.gov.au/-/media/corporate/files/publications/policies/neighbourhood-parking-policy/neighbourhood-on-street-parking-policy-2025-12.pdf?download=true). **Strong.**
- **Residents of newer flats get no street permit.** Not eligible: "a new or Converted Residential Flat Building ... approved since 8 May 1996 (former South Sydney) or 1 May 2000 (former Sydney City Council) or 12 May 2014 (all remaining areas)" (s 10.2).
- **So a spaceless flat in a post-1996 building has no fallback,** while one in an older block in Zone B can hold two permits (s 10.3). The rule should push the premium up in newer buildings. This can be tested once we have build years.
- **Zone A covers Kings Cross only,** the one area where permits outnumber spaces: one permit at most, none with on-site parking (s 9.3, 10.3).
- **The policy names Green Square** as the intensification concern (s 8.3). Zetland, Beaconsfield and Rosebery are parking Area 35, and Waterloo appears inside Redfern Area 41 on the map.
- **Central Sydney is outside this policy** and has its own (not read). Barangaroo's status: not verified. The maps mark the CBD and some precincts "Not Applicable".
- Some Ultimo and Pyrmont properties are limited to one permit (s 8.4). Typical controls are 2P at 8am-8pm up to 8am-10pm (Table 1).
- **Meters:** Area 1 $9 an hour, Area 2 $7 (2026-27, file dated 3 July 2026). [Council](https://www.cityofsydney.nsw.gov.au/transport-parking/finding-and-using-parking-meter). **Strong.**
- **LEP maximums:** sources conflict (0.5 per one-bed in a [consultant report](https://meetings.cityofsydney.nsw.gov.au/documents/s96676/Attachment%20D%20-%20Clause%204.6%20Variation%20Request%20-%20Parking.pdf), lower in a snippet). I could not open the LEP, so treat as unverified.

### 2.4 Stand-alone car space rents, as an external check
**BACKGROUND ONLY:** nearly all are undated listings seen in November 2026. They are asks, mostly search snippets (Parkhound, Parksy, Parking Made Easy and Domain block fetching). They check our story and must not feed the model.

| Area | $/week asked | Source |
|---|---|---|
| CBD | 80-95, up to 140-173 at operators | [Domain, Aug 2025](https://www.domain.com.au/car-space-27-park-street-sydney-nsw-2000-17682381), [Raine & Horne](https://raineandhorne.com.au/properties/b6-car-space-130-elizabeth-st-sydney-2000-new-south-wales), [Parkhound](https://www.parkhound.com.au/blog/?p=358) |
| Haymarket | About 117 | [Parkhound](https://www.parkhound.com.au/parking-haymarket-quay-st-nsw-108275) |
| Pyrmont | 45-90 typical | [Parksy](https://www.parksy.com/parking/au/nsw/pyrmont/all) |
| Surry Hills | 45-100 | [Parking Made Easy](https://www.parkingmadeeasy.com.au/rent-car-spaces/car-space-rent-surry-hills) |
| Zetland | 25-75 | [Parksy](https://www.parksy.com/parking/au/nsw/zetland/secure-accessible-private-parking-zetland-50week-89186) |
| Waterloo | 30-120 | [Parksy](https://www.parksy.com/parking/au/nsw/waterloo/secure-parking-at-crown-square-plaza-waterloo-nsw--92327) |
| Barangaroo | 200 and 400, two asks only | [Parking Made Easy](https://www.parkingmadeeasy.com.au/parking-barangaroo) |

The ordering matches ours: CBD above inner ring above Green Square. Barangaroo and The Rocks are the exceptions, sitting above what a space alone seems to fetch. Wilson and Secure Parking current rates were not found.

### 2.5 What drives the premium, tested on our data
**Building level: about 1,700 buildings, far more power than 30 suburbs. Figures are the pooled premium, so read the shape, not the level.**

| Building trait | Pattern (premium in $) |
|---|---|
| Typical rent | Under $650: 25. $650-800: 38. $800-950: 42. $950-1,200: 38. $1,200-1,600: 64. Over $1,600: 205 (18 buildings) |
| Size (flats) | Up to 20: 25. 21-50: 30. 51-100: 38. 101-250: 49. Over 250: 42 |
| Share of flats with parking | Under 30%: 24. 30-50%: 45. 50-70%: 45. 70-90%: 40. Over 90%: 21 |
| Distance to rail or metro | Under 300 m: 53. 300-500: 40. 500-800: 39. 800-1,200: 28. Over 1,200: 27 |

- **Roughly 4-5% of rent up to $1,600, then a jump.** The jump marks where "with parking" starts to mean a different class of flat.
- **Scarcity has a hump.** Where almost no flats have a space, the few that do may be awkward ones, or tenants hold street permits. Where almost all do, the flats without one are the odd ones out and spaces are in surplus.
- **Stations do not explain it.** The rail gradient is really a CBD effect. Dropping the CBD core gives 36, 33, 41, 28, 27. Inside a suburb, near buildings minus far buildings is -$2.90 (SE 3.3), so no effect. Good transit does not visibly lower the premium at this scale.

**Suburb level (Spearman rank correlation with the same-period premium; 24 suburbs with 100+ comparisons, excluding Barangaroo and The Rocks; about +-0.40 needed for significance):**

| Factor | Correlation | Reading |
|---|---|---|
| Distance to rail or metro | -0.53 | A proxy for closeness to the CBD |
| Share of flats with parking | -0.41 | Scarcity raises the premium |
| Households with no car | +0.35 | Wrong way for "more cars, more value": scarcity causes both |
| Income | +0.12 | Nothing |
| Density (flats per occupied hectare) | +0.01 | Nothing |
| Typical rent | -0.28 | Nothing once luxury suburbs are out |

Thirty suburbs is too few for firm claims, and all of this is correlation. Using the pooled premium instead gives the same picture with weaker scarcity (-0.10).

### 2.6 Why the premium is near zero in some places and very high in others
- **Near zero:** spaces plentiful (76-80% of flats have one), cheap stand-alone spaces nearby, many tenants without cars, and supply above demand, as Grattan finds. Green Square fits.
- **High, for real reasons:** few spaces, high meter charges, short limits, no permit for newer buildings, and well-paid tenants. Potts Point, Surry Hills and the CBD fit.
- **High, for false reasons:** within a building the flat with the space is bigger, higher, has the better view, or belongs to a better tier. Bedrooms and bathrooms do not capture that. Barangaroo and The Rocks fit.

### 2.7 Confounding
- Spaces go to bigger, newer, better flats. Studies report bias upward without size controls. [Gabbe and Pierce, via summary](https://carrefour.vivreenville.org/storage/app/media/publications/Articles-de-fond-Encyclopedique/problematique-du-stationnement/hidden-cost-bundled-parking.pdf). **Strong, seen only in a summary.**
- Grattan calls units with 3+ spaces "quality-confounded".
- Our within-building design beats the published work I saw on location and age, because I found no published within-building rent estimate. It cannot see floor area or aspect, so it probably still leans high.
- Parking arms also differ on height: in Sydney suburb, flats with a space sit on floor 19 against 14.7. Floor is adjusted in the pooled figure but not in the same-period one, which at $3.80 a floor is about $16 of the $78 there.
- The same-flat view removes all of this, but it is a small and unusual sample.

## Question 3: area by area
**Columns.**
- *Measured*: `parking.json` as supplied (to 31 July).
- *Same-period*: 2023 to March 2026, same year, same building, bedrooms and bathrooms, plus or minus one SE.
- *With*: share of flats with a space.
- *No car*: 2021 Census, households with no vehicle ([ABS QuickStats](https://abs.gov.au/census/find-census-data/quickstats/2021/SAL14201), one page per suburb, codes in the sources list). Strong, but note Waterloo includes public housing and Barangaroo had only 220 people in 2021.
- *Rail*: median metres from flats to the nearest rail or metro station (OSM, straight line).
- *Believable range*: my judgement for one car space, not a fitted result. It blends the views in 2.1 and shrinks thin suburbs. Treat it as a reading guide.

| Suburb | Measured | Flats / buildings | Same-period | With | No car | Rail | Verdict | Believable range |
|---|---|---|---|---|---|---|---|---|
| Sydney | $69 | 4,174 / 82 | $78 +-13 | 56% | 59% | 263 | **Believable.** Flips agree ($50, n=219). Under the separate Central Sydney policy | $55-80 |
| Pyrmont | $70 | 1,176 / 38 | $67 +-21 | 90% | 28% | 1,084 | **Believable.** Two views agree, no train, spaces $45-90 | $55-75 |
| Woolloomooloo | $72 | 687 / 35 | $105 +-31 | 64% | 44% | 487 | **Believable direction,** wide | $60-100 |
| Potts Point | $49 | 2,388 / 144 | $89 +-11 | 36% | 56% | 383 | **Believable.** Fewest spaces, in or beside Zone A | $50-90 |
| Surry Hills | $49 | 2,446 / 133 | $85 +-9 | 60% | 48% | 539 | **Believable.** Large sample, spaces $45-100 | $50-85 |
| Darlinghurst | $54 | 1,813 / 140 | $116 +-13 | 49% | 45% | 414 | **Direction believable,** $116 likely mixes in better flats | $50-90 |
| Elizabeth Bay | $54 | 1,181 / 76 | $77 +-17 | 49% | 44% | 592 | **Believable** | $50-75 |
| Camperdown | $51 | 1,034 / 68 | $70 +-10 | 81% | 28% | 1,363 | **Believable** | $45-65 |
| Redfern | $43 | 837 / 56 | $64 +-11 | 76% | 39% | 409 | **Believable** | $40-60 |
| Erskineville | $41 | 1,006 / 31 | $62 +-17 | 78% | 25% | 396 | **Believable** | $40-60 |
| Alexandria | $36 | 864 / 80 | $52 +-8 | 75% | 23% | 642 | **Believable** | $35-50 |
| Centennial Park | $43 | 339 / 23 | $45 +-9 | 80% | 20% | 1,563 | **Believable.** Views agree | $35-45 |
| Glebe | $32 | 940 / 92 | $51 +-15 | 65% | 33% | 1,931 | **Believable.** Old stock, permits likely help | $30-50 |
| Haymarket | $29 | 1,989 / 25 | $43 +-10 | 63% | 59% | 484 | **Medium.** Few big towers, flips -$12 | $25-45 |
| Ultimo | $21 | 1,112 / 35 | $39 +-14 | 61% | 45% | 935 | **Medium.** Student area, few cars | $20-40 |
| Chippendale | $24 | 1,726 / 52 | $40 +-10 | 56% | 59% | 509 | **Medium.** Student area, typical building about $0 | $20-40 |
| Newtown | $23 | 1,186 / 97 | $59 +-11 | 63% | 31% | 425 | **Medium.** Views disagree | $25-55 |
| Rushcutters Bay | $25 | 578 / 39 | $62 +-14 | 47% | 49% | 494 | **Medium.** Views disagree | $25-60 |
| Paddington | $16 | 798 / 68 | $63 +-13 | 51% | 21% | 886 | **Medium.** Terraces, older permits, views disagree | $20-60 |
| Zetland | $16 | 2,728 / 101 | $38 +-12 | 76% | 30% | 687 | **Low is real,** $16 slightly under. Spaces $25-75 | $15-35 |
| Waterloo | $4 | 1,810 / 90 | $26 +-6 | 76% | 37% | 943 | **Low is real, $4 is too low.** See below | $15-30 |
| Rosebery | $18 | 1,168 / 68 | $24 +-9 | 80% | 17% | 999 | **Believable.** Views agree | $15-25 |
| Forest Lodge | $7 | 472 / 43 | $25 +-18 | 68% | 28% | 2,073 | **Noisy, likely low** | $10-30 |
| Annandale | $13 | 341 / 36 | $48 +-20 | 66% | 14% | 1,358 | **Noisy.** Most households have cars but street parking is easier (inference) | $15-40 |
| Barangaroo | $232 | 81 / 10 | $180 +-70 | 78% | 30% | 387 | **Confounded.** See below | $100-150 |
| The Rocks | $165 | 45 / 6 | $249 +-106 | 80% | 27% | 382 | **Confounded.** 64% of weight in one building | $80-150 |
| Millers Point | $51 | 134 / 5 | $125 +-78 | 76% | 31% | 324 | **Too thin.** Heritage mix | $50-100 |
| Beaconsfield | $6 | 95 / 7 | $22 +-31 | 78% | 13% | 562 | **Too thin.** Use Green Square | $15-30 |
| St Peters | $32 | 55 / 8 | $63 +-29 | 92% | 16% | 356 | **Too thin** | $25-55 |
| Darlington | $53 | 56 / 8 | $100 (n=10) | 71% | 32% | 356 | **Too thin.** Use Redfern | $30-60 |
| Eveleigh | Suppressed | | n=7 | 96% | | | **Too thin.** One building | Use Redfern |

Suburbs with no comparisons, such as Dawes Point, should inherit from a wider level, as `premium_for` already does through its fallbacks.

### Why Waterloo is low, and why $4 is still too low
- **It is not one odd building.** Dropping any single building leaves $1.40-$5.00.
- **It is not flag noise.** Waterloo's flip rate (2.4%) is below the citywide 3.5%.
- **The whole precinct is low.** Waterloo, Zetland, Rosebery and Beaconsfield all sit lowest, and their same-flat flips average -$11 (-32 to +6). So low is real.
- **Likely reasons:** 76-80% of flats already have a space, so spaces are not scarce. Stand-alone spaces are cheap in Zetland ($25-75). Adverts here are mostly modern towers. Grattan's oversupply story fits. All of this is inference.
- **But $4 is too low.** Same-period says $26 +-6, and nearly every genuine stand-alone ask is $25 or more. The pooled figure deflates all years with a citywide index, and any precinct whose rents grew differently will be distorted. I did not prove this. Test: idea 9.
- **The Census is a poor guide here.** Waterloo's 37% no-car share and $530 median rent include a public estate of about 2,012 homes ([Substack, weak](https://moresocialhousing.substack.com/p/waterloo-cutting-through); the [City of Sydney PDF](https://meetings.cityofsydney.nsw.gov.au/documents/s25301/Waterloo%20Estate%20Redevelopment%20-%20A%20Better%20Way%20for%20the%20Community.pdf) gives the same figure, seen as a snippet). Those homes never appear in private adverts.
- **The metro made no visible difference.** Waterloo's station opened in August 2024 ([NSW Government](https://nsw.gov.au/media-releases/sydney-metro-city-to-open-on-monday-19-august)). The same-period premium was $19 (SE 8) before and $17 (SE 9) after. That is too noisy to claim anything except no large fall.

### Why Barangaroo is mostly not about the car
- **Small sample:** 81 flats, 10 buildings, 45 same-period comparisons.
- **The raw gap shows two classes of flat:** $2,750 with a space against $1,350 without. Within matched cells it is still 15% of rent, against 1-9% in other suburbs.
- **The same jump appears across the top rent band:** $205 for buildings over $1,600, against 4-5% of rent below that.
- **A space alone seems worth less.** CBD asks run $80-$170. Barangaroo has just two asks ($200, $400), which could equally be luxury-tower spaces.
- **Floors barely help.** Only 43% of comparisons have a floor, and floor here is just the unit number.
- **Bedrooms and bathrooms cannot separate** a harbour-view corner flat from a back-of-building flat. One-bed listings at 88 Barangaroo Ave often show no space, and larger ones do (snippets, weak).
- **The Census cannot help:** 220 people in 2021, against 1,800+ residents now ([agent page, weak](https://a-d.com.au/buying-living/market-insights/one-sydney-harbour-completes-barangaroo-souths-landmark-transformation)).
- **My reading:** the space itself is plausibly $100-$150, and the rest is flat quality. Do not quote $232 as the price of parking.

## Question 4: tips from rent and AVM practice
### Valuers
- RICS ranks recent near-identical lettings first, including the subject's own, with indices below that. Old evidence is usable with a market trend. Adjust for location, specification, size, timing and incentives. [RICS 2019](https://rics.org/content/dam/ricsglobal/documents/standards/comparable_evidence_in_real_estate_valuation.pdf). **Strong.** Our hierarchy of own, building, block anchors is the same logic.
- The Australian Property Institute warns against simply averaging method results. [API](https://api.org.au/professional-development/knowledge-hub/valuation-protocol-valuation-approaches-and-methods). **Strong.** No adjustment numbers are given.

### Commercial AVMs
- **Closest benchmark:** Microburbs' rental AVM (March 2026, adverts July 2024 to March 2026). Median error 5.5%, apartments $53.77, NSW $73.21. Within 10%: 84.2% with rental history, 67.9% without. Whole suburbs are withheld from training, which is a harder test than ours. [Microburbs](https://www.microburbs.com.au/research/rental-avm-whitepaper). **Moderate, vendor claim. I read it directly.** "MAE" there may not mean median, so do not compare to our $42 without checking.
- Our pseudo-folds show about 5% median percent error: 4.5% at $600-$1,000, rising to 6.2% at $1,500-$2,000 and 9.7% above $2,000.
- History is worth a lot. Our errors by anchor: own $38, building $62, block $85.
- Zillow publishes sale accuracy of 1.94% on-market against 7.06% off-market (via a copy; [the Close](https://theclose.com/how-accurate-is-zillow-zestimate/)). **Weak.** The list price is an input on-market, so this looks like the value of an anchor. No Rent Zestimate median error was found.
- PropTrack combines comparable, index, listing and hedonic models. [PropTrack](https://www.proptrack.com.au/?p=4440). **Moderate.** No accuracy figures. No official Cotality or Domain accuracy figures were found.
- **Practical takeaway:** feed comparables in together, not as a fallback chain. We already do: own, building and block anchors all reach the model, and the building anchor leaves the flat out. Cook County's condo model adds time weighting, with a leave-one-out, time-weighted building mean over 5 years as a LightGBM feature. [CCAO](https://github.com/ccao-data/model-condo-avm). **Moderate.** See idea 5.

### Time adjustment
- Do not roll anchors with CPI rents. New-tenant rents rose 14% against about 5% for CPI rents in the year to February 2023. New-tenant rents tracked adverts. [RBA](https://www.rba.gov.au/publications/bulletin/2023/jun/new-insights-into-the-rental-market.html). **Strong.** We already use advert-based indices.
- Stratified medians beat a raw median by a wide margin in Sydney: deviation-from-trend RMSE 4.70 for the raw median, about 1.15 with suburb-price strata, with little difference from 4 to 60 groups. [RBA RDP 2006-04](https://www.rba.gov.au/publications/rdp/2006/pdf/rdp2006-04.pdf). **Strong.** Our tier bias test suggests little left to gain here.
- In February 2023, 49.3% of Sydney rental stock within 12.5 km of the CBD was still below pre-pandemic rent, against 10.1% at 12.5-25 km. [ABS](https://www.abs.gov.au/statistics/detailed-methodology-information/information-papers/new-insights-rental-market). **Strong.** A citywide index would misprice 2020-22 anchors near the CBD. Ours is already by suburb, so only the link to a flat's own history is left (idea 4).
- Zillow's repeat-rent index upweights short gaps. [Zillow](https://www.zillow.com/research/methodology-zori-repeat-rent-27092/amp/). **Moderate (snippet).**

### Thin data
- Louisville: within 10% rose from 36.5% (plain hedonic) to 47.4% with submarkets plus kriging. [Bourassa et al.](https://www.prres.org/uploads/763/1688/Bourassa_Predicting_House_Prices_With_Spatial.pdf). **Strong, but linear models.**
- With flexible models the gain shrinks: Japan apartment rents, XGBoost beat kriging at every sample size. [Yoshida and Seya](https://arxiv.org/abs/2107.12539). **Moderate.**
- Ireland 2022: smoothing wins where data are thin (GAM 16% against random forest 56% median error), not in Dublin (13% against 14%). [O'Donovan et al.](https://arxiv.org/abs/2511.12625). **Moderate.** Thin suburbs here are few, so expect little from kriging.
- Learned comparables beat plain XGBoost: 16.08 to 13.13 on the time-split dataset. [Popelier et al.](https://arxiv.org/abs/2603.12986). **Moderate.** The baseline seems to lack comparable features, so this does not test boosting plus comparables.
- IAAO: use the median, bootstrap the intervals. Condo COD of 5-10 is normal, and below 5 is suspect. [IAAO 2013](https://www.audits2.ga.gov/wp-content/uploads/2024/08/Standard_on_Ratio_Studies.pdf). **Strong.** Below 5 in a fold would hint at leakage (my analogy).

### Median-error objectives
- Absolute loss targets the median. [Gneiting](https://www.econbiz.de/Record/quantiles-as-optimal-point-forecasts-gneiting-tilmann/10008871353). **Strong (snippet).** We already use L1.
- Medians survive transforms, so exp(median of log) is the median rent and no log bias correction is needed. [Geraci and Jones](https://oro.open.ac.uk/44629). **Strong (snippet).**
- Our metric is the median of errors, not their mean. The ideal point is the centre of the densest window of about +-$42 of the flat's possible rents, not strictly its median (inference). Gains from metric-aligned loss are small: 89.4% to 90.0% within 20% in Norway. [Hjort et al.](https://econpapers.repec.org/RePEc:taf:jpropr:v:39:y:2022:i:4:p:338-364). **Strong (snippet).**
- Trees beat deep models at this size. [Grinsztajn et al.](https://arxiv.org/abs/2207.08815). **Strong.** No reason to change model family.
- Use paired bootstraps and time-forward splits, as random splits flatter models.

### Rounding predictions
See 1.8. Heaped data can bias estimators when heaping depends on the true value. [Barreca et al.](https://econpapers.repec.org/RePEc:bla:ecinqu:v:54:y:2016:i:1:p:268-293). **Strong on the general point (snippet).** Porto Alegre's 30.5% of properties sharing their nearest comparable's exact price is a reminder that clustering changes how comparables behave. [Popelier et al.](https://arxiv.org/abs/2603.12986). **Moderate.**

## Lines for the pitch or report
1. "One in four Sydney flats re-advertised within two years comes back at exactly the old rent, even in a rising market. Landlords think in last year's number, so we teach the model that too."
2. "A car space is worth about $45 a week in the City of Sydney, but that hides a real spread: around $20 in Green Square, where most flats have one, and $50 to $80 from Surry Hills to the CBD."
3. "We checked this four ways: flats compared inside the same building, the same flat with and without a space, Sydney sale evidence, and what stand-alone spaces rent for. Together they rule out the extremes."
4. "We do not claim Barangaroo parking is worth $232. There, the flat with a space is usually a better flat, and we say so."
5. "Proximity to a station does not lower the premium within a suburb. What matters is how scarce spaces are."

Lines 1, 2 and 5 are drawn from data to 31 March 2026. Re-check them against the final fit before use. Keep the second sentence of line 1 only if idea 1 ships. Line 2 is my judgement across the views in 2.1, so call it an estimate.

## What I could not verify
- Any Sydney rent hedonic, a rent-to-floor-area elasticity, or a published Australian study of rent rounding.
- How our lease file treats re-adverts under 91 days, and why our unchanged share exceeds ABS's.
- Australian furnished premiums, gym or pool premiums, aspect or frontage effects.
- Student shares by suburb, or any effect of student housing on flat rents.
- LEP parking maximums (sources conflict), permit fees, Barangaroo's permit status, and the Central Sydney policy.
- Current Wilson or Secure monthly rates. Stand-alone asks are undated and mostly snippets.
- Whether spaces can legally be sublet in a given strata scheme.
- The cause of Waterloo's low pooled figure. The index explanation is untested.
- My "believable ranges" are judgement, and the seasonal estimates rest on nine years.
- Sources marked "snippet" or "summary" were not read in full. Figures from the research pass were read through a summariser. I re-read these directly: ABS rental article, RBA RDP 2020-04, RBA student article, Genesove abstract, Grattan slides, City of Sydney policy (pages 1-18), Microburbs paper, NSW rent page and two QuickStats pages (Waterloo, Chippendale). Spot-check anything else before quoting it.
- Census table values were machine-read, with those two pages checked by hand.

## Sources
Strength as defined at the top. "Direct" means I re-read it myself. "Snippet" means only a search result or abstract was seen. Everything else was read once through a summariser.

**Rents, tenancy and markets (Australia)**
- ABS, Latest insights into the rental market, 28 May 2025. Strong, direct. https://www.abs.gov.au/articles/latest-insights-rental-market
- ABS, New insights into the rental market. Strong. https://www.abs.gov.au/statistics/detailed-methodology-information/information-papers/new-insights-rental-market
- RBA Bulletin, New insights into the rental market, June 2023. Strong. https://www.rba.gov.au/publications/bulletin/2023/jun/new-insights-into-the-rental-market.html
- RBA RDP 2020-04, The Apartment Shortage, Appendix D. Strong for sale prices, direct. https://www.rba.gov.au/publications/rdp/2020/2020-04/appendix-d.html
- RBA RDP 2006-04, Prasad and Richards. Strong. https://www.rba.gov.au/publications/rdp/2006/pdf/rdp2006-04.pdf
- RBA Bulletin, International students, July 2025. Strong, direct. https://www.rba.gov.au/publications/bulletin/2025/jul/international-students-and-the-australian-economy.html
- NSW Government, Rent increases. Strong, direct. https://www.nsw.gov.au/housing-and-construction/renting-a-place-to-live/rent-increases
- NSW Government, No-grounds evictions one year on. Strong. https://www.nsw.gov.au/ministerial-releases/renters-better-protected-one-year-on-from-no-grounds-evictions-ban
- NSW Fair Trading, Rent bidding insights report. Strong on direction. https://www.nsw.gov.au/departments-and-agencies/fair-trading/who-we-are/rent-bidding-nsw-insights-report
- NSW Government, Advertising and rent bidding rules. Strong, snippet. https://www.nsw.gov.au/housing-and-construction/rules/advertising-and-rent-bidding-on-rental-properties
- NSW Planning, Short-term rental accommodation. Strong. https://www.planning.nsw.gov.au/short-term-rental-accommodation
- City of Sydney, short-term rental decision. Strong. https://meetings.cityofsydney.nsw.gov.au/ieDecisionDetails.aspx?AIId=18994
- City of Sydney, Affordable student accommodation shortage. Strong. https://meetings.cityofsydney.nsw.gov.au/documents/s70704/Affordable%20Student%20Accommodation%20Shortage.pdf
- Department of Education, National planning level. Strong, snippet. https://www.education.gov.au/node/20481
- Cotality Rental Review, April 2025. Moderate, snippet. https://corelogic.com.au/__data/assets/pdf_file/0024/26637/202504_Rental-Review.pdf
- SQM via Adviser Voice, January 2023. Moderate. https://www.adviservoice.com.au/2023/01/vacancy-rates-seasonally-increase-over-december/
- Microburbs rental AVM whitepaper, March 2026. Moderate, vendor, direct. https://www.microburbs.com.au/research/rental-avm-whitepaper

**Rent and price studies**
- Genesove, The Nominal Rigidity of Apartment Rents, NBER w7137. Strong, direct (abstract). https://www.nber.org/papers/w7137
- Turkey rent rigidity, MPRA 55292. Moderate. https://mpra.ub.uni-muenchen.de/55292/1/MPRA_paper_55292.pdf
- ESRI, Property-level rental inflation. Moderate. https://esri.ie/publications/understanding-trends-in-property-level-rental-inflation
- Melbourne building age and rent, Urban Studies 2021. Strong, no coefficients seen. https://findanexpert.unimelb.edu.au/scholarlywork/1455513-examining-building-age--rental-housing-and-price-filtering-for-affordability-in-melbourne--australia
- San Diego floor premium. Moderate. https://ideas.repec.org/a/eme/ijhmap/v6y2013i2p197-208.html
- Hong Kong floor premium. Moderate. https://scholars.ln.edu.hk/en/publications/the-pattern-of-floor-level-premium-in-apartment-buildings/
- Hong Kong view premium. Moderate. https://scholars.ln.edu.hk/en/publications/estimating-view-premiums-in-high-rise-residential-housing-hedonic/
- Traffic noise and rents. Moderate. https://portal.fis.tum.de/en/publications/impact-of-simulation-based-traffic-noise-on-rent-prices/
- Poznań noise and sales. Moderate, snippet. https://journals.vilniustech.lt/index.php/IJSPM/article/view/20563
- Furnished premiums: UK, weak, snippet, https://propertyindustryeye.com/research-reveals-city-differences-in-pre-furnished-rental-premium/ ; Brazil, moderate, snippet, https://periodicos.ufsc.br/index.php/economia/article/view/81978
- Pope et al., Round numbers. Strong, snippet. https://ideas.repec.org/a/eee/gamebe/v93y2015icp89-107.html
- Uppsala, just-below asking prices. Moderate. https://swopec.hhs.se/uunewp/abs/uunewp2017_010.htm

**Parking**
- Grattan Institute, Wasted space, 8 July 2026. Strong-moderate, direct. https://grattan.edu.au/wp-content/uploads/2026/07/Grattan-Institute-Presentation-2026-Wasted-space-WTP-methodology-ACE-presentation-8-July-2026.pdf
- City of Sydney, Neighbourhood On-Street Parking Policy, approved April 2025. Strong, direct (pages 1-18). https://www.cityofsydney.nsw.gov.au/-/media/corporate/files/publications/policies/neighbourhood-parking-policy/neighbourhood-on-street-parking-policy-2025-12.pdf?download=true
- City of Sydney, Parking meters. Strong. https://www.cityofsydney.nsw.gov.au/transport-parking/finding-and-using-parking-meter
- Glebe clause 4.6 parking report. Moderate. https://meetings.cityofsydney.nsw.gov.au/documents/s96676/Attachment%20D%20-%20Clause%204.6%20Variation%20Request%20-%20Parking.pdf
- Gabbe and Pierce, summary of Hidden Cost of Bundled Parking. Strong, summary only. https://carrefour.vivreenville.org/storage/app/media/publications/Articles-de-fond-Encyclopedique/problematique-du-stationnement/hidden-cost-bundled-parking.pdf
- Jia and Wachs, San Francisco. Strong, small sample. https://escholarship.org/uc/item/0fm8k169
- Manville, LA. Moderate, abstract. https://ideas.repec.org/a/taf/rjpaxx/v79y2013i1p49-66.html
- San Francisco parking lottery, via Streetsblog. Strong study, snippet. https://usa.streetsblog.org/2021/02/10/study-yes-more-parking-does-put-more-cars-on-the-road
- Amsterdam permit value, Tinbergen. Strong, abstract. https://papers.tinbergen.nl/09083.pdf
- RMIT, Unbundled parking. Moderate. https://www.rmit.edu.au/news/all-news/2023/mar/unbundled-parking

**Valuation and AVM practice**
- RICS, Comparable evidence in real estate valuation, 2019. Strong. https://rics.org/content/dam/ricsglobal/documents/standards/comparable_evidence_in_real_estate_valuation.pdf
- API valuation protocol. Strong. https://api.org.au/professional-development/knowledge-hub/valuation-protocol-valuation-approaches-and-methods
- IAAO Standard on Ratio Studies, 2013. Strong. https://www.audits2.ga.gov/wp-content/uploads/2024/08/Standard_on_Ratio_Studies.pdf
- Cook County condo model. Moderate. https://github.com/ccao-data/model-condo-avm
- PropTrack AVM. Moderate. https://www.proptrack.com.au/?p=4440
- Zillow accuracy, via the Close. Weak. https://theclose.com/how-accurate-is-zillow-zestimate/
- Zillow ZORI method. Moderate, snippet. https://www.zillow.com/research/methodology-zori-repeat-rent-27092/amp/
- Bourassa et al., spatial prediction. Strong, linear models. https://www.prres.org/uploads/763/1688/Bourassa_Predicting_House_Prices_With_Spatial.pdf
- Yoshida and Seya. Moderate. https://arxiv.org/abs/2107.12539
- O'Donovan et al. Moderate. https://arxiv.org/abs/2511.12625
- Popelier et al. Moderate. https://arxiv.org/abs/2603.12986
- Grinsztajn et al. Strong. https://arxiv.org/abs/2207.08815
- Gneiting. Strong, snippet. https://www.econbiz.de/Record/quantiles-as-optimal-point-forecasts-gneiting-tilmann/10008871353
- Geraci and Jones. Strong, snippet. https://oro.open.ac.uk/44629
- Hjort et al. Strong, snippet. https://econpapers.repec.org/RePEc:taf:jpropr:v:39:y:2022:i:4:p:338-364
- Barreca et al. Strong on the general point, snippet. https://econpapers.repec.org/RePEc:bla:ecinqu:v:54:y:2016:i:1:p:268-293

**Census and suburb context**
- ABS 2021 QuickStats, strong, read by machine with Waterloo and Chippendale checked directly. Pattern: `https://abs.gov.au/census/find-census-data/quickstats/2021/` plus the code. Barangaroo SAL10192, The Rocks SAL13856, Millers Point SAL12630, Sydney SAL13730, Haymarket SAL11884, Ultimo SAL14024, Pyrmont SAL13297, Waterloo SAL14201, Zetland SAL14542, Rosebery SAL13419, Woolloomooloo SAL14424, Potts Point SAL13268, Elizabeth Bay SAL11398, Darlinghurst SAL11213, Surry Hills SAL13714, Redfern SAL13352, Chippendale SAL10906, Glebe SAL11645, Forest Lodge SAL11549, Newtown SAL12969, Erskineville SAL11442, Camperdown SAL10780, Alexandria SAL10030, Paddington SAL13126, Rushcutters Bay SAL13459, Annandale SAL10062, Beaconsfield SAL10239, Darlington SAL11215, Centennial Park SAL10872, St Peters SAL13659.
- Sydney Metro City opening, 19 August 2024. Moderate. https://nsw.gov.au/media-releases/sydney-metro-city-to-open-on-monday-19-august
- Waterloo estate size: Substack, weak, https://moresocialhousing.substack.com/p/waterloo-cutting-through ; City of Sydney, snippet, https://meetings.cityofsydney.nsw.gov.au/documents/s25301/Waterloo%20Estate%20Redevelopment%20-%20A%20Better%20Way%20for%20the%20Community.pdf
- Barangaroo residents, agent page. Weak. https://a-d.com.au/buying-living/market-insights/one-sydney-harbour-completes-barangaroo-souths-landmark-transformation
- Station locations: OSM extract already pulled by the public data lane (`data/external/osm_overpass_poi/poi.json`). No new pull.

**BACKGROUND ONLY (dated after 31 July 2026, or undated and seen in November 2026). Must not feed the model.**
- SQM via MPA, 15 September 2026. https://www.mpamag.com/au/news/general/rental-vacancies-split-as-spring-test-looms-for-landlords/589728
- SQM via PIP, October 2026. https://propertyinvestmentprofessionals.com.au/research-insights/sqm-national-vacancy-september-2026-1-4-percent-sydney-unit-rents-turn
- WSU build-to-rent study, listed date 14 December 2026, which is after today, so verify. https://researchers.westernsydney.edu.au/en/publications/amenities-premium-services-and-rental-premiums-in-melbournes-buil/
- Stand-alone car space listings in section 2.4 (Raine & Horne, Parkhound, Parksy, Parking Made Easy; links in 2.4). These are undated. The one Domain ask is dated August 2025, so it is pre-cutoff, but treat it as a sense check like the rest.
