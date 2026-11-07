# Method comparison

Real data, Fold B and earlier windows only. "Error" is median absolute error in $/week. Fold A was not scored or read: its rows were dropped from the panel before it was saved.

## Result
**Nothing beat the current model by a margin we can trust.** The one recommended change is averaging 5 seeds. That removes seed luck; it is not a measured accuracy gain.

| | Fold B error | Fold B within $100 | Inner error | Inner within $100 |
|---|---|---|---|---|
| Published model (seed 7) | $42.62 | 81.0% | $44.09 | |
| Same model, mean of 10 single seeds | $43.05 | 81.3% | $44.03 | |
| **Recommended: same model, 5-seed average** | **$42.51** | **81.4%** | **$43.97** | **79.2%** |

Against seed 7, the 5-seed average is -$0.12 (95% interval -$1.34 to +$0.93). Its Fold B interval is $40.36 to $44.67.

## Noise floor
| Source | Size |
|---|---|
| Seed spread, 5 seeds, single model | SD $0.71, range $42.58 to $44.23 |
| Seed spread, 10 seeds, single model | SD $0.51, mean $43.05 |
| Spread of a 3-seed average | SD $0.30 |
| Spread of a 5-seed average | SD $0.22 |
| Bootstrap of Fold B error, resampling adverts | SD $1.18, interval $40.76 to $45.38 |
| Bootstrap, resampling buildings | SD $1.30, interval $40.34 to $45.58 |
| Bootstrap of within $100 | 79.4% to 82.6% |
| Paired gap between two configs on Fold B | interval about ±$1.2 |
| Paired gap pooled over 4 windows (8,756 adverts) | interval about ±$0.5 |

Seed 7 sits $0.43 below the 10-seed mean, so part of the published $42.62 is luck. A second warning: 800 rounds scored $42.18 on 3 seeds and $43.02 on 5. Any 3-seed lead under about $1 should be read that way.

## Windows
- **Fold B:** Apr-May 2026, 2,182 scored adverts.
- **Inner:** I1 Feb-Mar 2026, I2 Dec-Jan, I3 Oct-Nov 2025. Each trains only on origins before its own cutoff, and "inner" is the mean of the three.
- **Pooled:** all four windows together, which gives the tightest paired interval.

## Results
Gaps are paired against the same-seed reference; negative is better. Rows at 5 seeds use the 5-seed reference and rows at 3 or 2 seeds use the 3-seed base ($42.29, inner $43.93). Fold B gap intervals resample adverts.

| Step | Config | Seeds | Fold B | Within $100 | Inner | Fold B gap | Pooled gap | Verdict |
|---|---|---|---|---|---|---|---|---|
| 2 | Blend 50% log_ratio, 50% direct | 5 | $42.46 | 81.6% | $43.67 | -$0.04 [-1.15, +1.02] | -$0.07 [-0.56, +0.40] | Null |
| 2 | Blend 25% log_ratio, 75% direct | 5 | $42.28 | 81.5% | $43.63 | -$0.23 [-1.49, +1.07] | -$0.09 [-0.74, +0.53] | Null |
| 2 | Direct only | 5 | $42.63 | 81.1% | $44.03 | | | Null |
| 3 | 800 rounds | 5 | $43.02 | 81.5% | $43.81 | +$0.51 [-0.30, +1.40] | $0.00 [-0.38, +0.35] | Null |
| 3 | Learning rate 0.02, 800 rounds | 3 | $42.10 | 81.3% | $43.68 | -$0.18 [-0.96, +1.01] | -$0.10 [-0.58, +0.36] | Null |
| 3 | 31 leaves, 400 rounds | 3 | $41.64 | 81.0% | $44.14 | -$0.64 [-1.41, +0.62] | +$0.01 [-0.59, +0.57] | Null, Fold B only |
| 3 | min data 50 | 3 | $42.29 | 81.7% | $43.74 | $0.00 [-0.81, +1.15] | -$0.10 [-0.56, +0.36] | Null |
| 3 | min data 100 | 3 | $43.35 | 81.0% | $43.92 | +$1.06 [-0.13, +2.06] | +$0.31 [-0.26, +0.74] | Worse on B |
| 3 | L2 = 10 | 3 | $42.59 | 81.3% | $43.71 | +$0.30 [-0.64, +1.43] | -$0.02 [-0.55, +0.42] | Null |
| 3 | Feature fraction 0.5 | 3 | $42.76 | 81.3% | $44.03 | +$0.47 [-0.89, +1.42] | +$0.07 [-0.48, +0.50] | Null |
| 3 | Combined: lr 0.02, 31 leaves, min 100, L2 10, ff 0.5 | 3 | $42.59 | 81.1% | $44.53 | +$0.30 [-0.91, +1.43] | +$0.49 [-0.11, +1.03] | Worse |
| 4 | 6 origins | 3 | $43.56 | 79.9% | $42.83 | +$1.28 [+0.17, +3.09] | -$0.34 [-1.15, +0.30] | Split |
| 4 | 8 origins | 3 | $43.92 | 80.4% | $43.04 | +$1.63 [+0.09, +3.21] | -$0.05 [-0.93, +0.46] | Split |
| 4 | 18 origins | 3 | $41.89 | 80.9% | $44.53 | -$0.40 [-1.67, +1.05] | +$0.42 [-0.33, +1.03] | Split |
| 4 | 24 origins | 3 | $42.20 | 81.3% | $44.30 | -$0.09 [-1.44, +1.52] | +$0.33 [-0.40, +1.03] | Split |
| 4 | 12 origins, 12-month half-life | 5 | $43.13 | 80.8% | $43.36 | +$0.63 [-0.36, +1.95] | -$0.14 [-0.71, +0.39] | Split |
| 4 | 24 origins, rows before May 2023 at 0.25 | 3 | $42.14 | 81.0% | $44.73 | -$0.15 [-1.23, +1.53] | +$0.61 [-0.15, +1.28] | Null |
| 5 | CatBoost, depth 6, 300 rounds | 2 | $42.14 | 80.7% | $43.97 | -$0.14 [-1.39, +1.60] | -$0.06 [-0.83, +0.74] | Null |
| 5 | LightGBM + CatBoost, half each | 5+2 | $42.79 | 81.5% | $43.82 | +$0.28 [-0.98, +1.45] | -$0.11 [-0.63, +0.39] | Null |
| 5 | Median regression on log anchors | 1 | $49.71 | 77.9% | $50.01 | | | About $7 worse |
| 6 | Shrink toward anchor, 0.8 | 3 | $43.42 | 80.3% | $43.23 | | | Split |
| 6 | Stretch away from anchor, 1.05 | 3 | $41.97 | 81.3% | $44.52 | | | Split |
| 6 | First-time flats to the direct model | 5 | $42.10 | 81.5% | $43.75 | -$0.41 [-1.30, +0.43] | -$0.13 [-0.53, +0.17] | Under $1, keep simple |

All 67 result rows, one per config and round count, are in `experiments.csv` (Fold A columns blank) and `out/real/methods/`.

**What the steps show:**
- **Seeds and blends:** averaging helps by removing luck. Blending direct adds nothing measurable.
- **Tuning:** the surface is flat. No setting wins on both Fold B and inner by more than noise, so tuning is not where the error is.
- **History length:** Fold B prefers more history and inner windows prefer less, and pooled they cancel. Choosing on Fold B alone would have picked 18 origins, which is $0.60 worse on inner. Fewer than 12 origins also costs 1 to 1.6 points of within $100 on Fold B.
- **Rows before 2023:** they do not help, whether full weight or down-weighted. Adverts per window jump from 628 to 1,847 at the 2023 break.
- **Shrinkage:** it moves Fold B and inner in opposite directions, like history length.

### First-time flats (Fold B 556 adverts)
| Option | First-time error, Fold B | First-time error, inner | Verdict |
|---|---|---|---|
| Joint model | $58.48 | $62.45 | Reference |
| Separate log_ratio model | $59.69 | $66.20 | Worse |
| Separate direct model | $60.18 | $64.66 | Worse |
| Weight first-time rows x2 | $60.43 | $64.63 | Worse |
| Weight first-time rows x4 | $58.62 | $65.28 | Worse |
| Route to joint direct model | $57.15 | $62.39 | Only option better on both, by under $1.50 and $0.10 |

Repeat flats sit near $39. The first-time gap needs new information such as building age or size, not a new model on these features.

### Bias
Tested beyond the brief. Median bias is not stable: +$6.57 on Fold B, -$12.67 on I1, -$13.94 on I2, -$0.32 on I3. Correcting a window with the previous window's bias made it worse (Fold B $45.49 at full correction, against $42.29). Do not add a correction.

## Parking gap
Scored on flipped adverts. Gaps are against the current two-pass model, 3-seed average.

| Gap method | Median gap | Fold B flipped (n=85) | Gap on Fold B | Pooled flipped (n=317) | Gap pooled | Fold B all adverts |
|---|---|---|---|---|---|---|
| Two-pass, anchors fixed (current) | $15.90 | $55.93 | | $57.80 | | $42.29 |
| 75% two-pass, 25% within-building | $22.70 | $53.51 | -$2.42 [-5.8, +9.2] | $55.77 | -$2.03 [-3.7, +3.9] | $42.40 |
| 50% each | $28.00 | $56.82 | +$0.89 [-11.6, +16.6] | $56.75 | -$1.05 [-5.1, +5.6] | $42.43 |
| Within-building only | $36.20 | $64.45 | +$8.52 [-10.1, +23.1] | $61.73 | +$3.93 [-3.2, +11.6] | $42.59 |
| Anchors rebuilt at test only | $72.20 | $62.09 | +$6.16 [-16.9, +34.2] | $63.80 | +$6.00 [-2.8, +23.0] | $42.43 |
| Anchors rebuilt at train and test | $50.90 | $56.52 | +$0.59 [-18.5, +17.6] | $55.54 | -$2.26 [-8.0, +7.6] | $42.60 |

**The sample cannot separate these.** Every interval spans zero by a wide margin, on the 85 Fold B adverts and on the 317 pooled ones.

- **Rebuilding anchors does fix the compression, but overshoots.** Test-only double-counts, because the model's own step stacks on the anchor step ($72). Rebuilding in training as well gives $51, still above the measured $36 to $40.
- **Rebuilding costs a little elsewhere.** Train-and-test within $100 on Fold B is 80.9% against 81.4%. That is inside the noise, but there is no gain to pay for it.
- **Flip direction differs.** On the 5-seed model, the 17 adverts that gained parking score $108 two-pass and $94 within-building. The 68 that lost it score $47 and $59. Seventeen adverts prove nothing, but flips may partly be advert noise, not real parking changes.

### Decision 15 needs a team call
Replaying `pick_premium` on 10 seeds (Fold B flipped, n=85):

| | Mean | SD | Min | Max |
|---|---|---|---|---|
| Rule margin | $3.48 | $4.42 | -$3.49 | $9.60 |

- **The $9.60 behind decision 15 is the highest of the 10 seeds.** Seven of 10 seeds choose within-building.
- **On the 5-seed model the rule's margin is $2.41, so it selects within-building.** If seed averaging goes into `local/run_model.py`, the premium source flips automatically, with nobody choosing it.
- **The rule and the submission measure different things.** The rule scores within-building at the midpoint of the two passes, but `compose_prices` keeps the flat's own pass. On flipped adverts the rule's midpoint figure equals the 50% blend above, not within-building only.
- **What the flip costs on the submission path (5-seed model):** Fold B moves from $42.51 and 81.39% to $43.00 and 81.67%, and flipped error from $54.21 to $64.81. Those are inside the noise.

## Recommendation
Change only this, and use it for the final fit:

```python
objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, cat_smooth=10,
deterministic=True, force_row_wise=True
num_boost_round = 400
method = "log_ratio"       # log(rent / best_anchor)
n_origins = 12             # no recency weights
seeds = (7, 11, 23, 42, 101)   # NEW: fit once per seed, average weekly rent per pass, then compose_prices
```

- Features are unchanged. Average each pass in dollars before `compose_prices`, then round as now.
- `local/run_model.py` and `Model` still fit one booster, so the loop has to be added. It is not on this branch.
- **Parking:** keep the two-pass gap. Nothing here justifies moving it, and a change would need to gain over $1. That goes against what `pick_premium` would output on the averaged model, so the team must either hold the source at two-pass or accept the flip. Decision 15 says not to change the rule after seeing a result, which is why this is flagged rather than decided here.
- **Expect** about $42.5 on Fold B, with an interval of about $40 to $45. Inner windows suggest $43 to $45.

## How this was run
- **Seeds:** noise floor 10 and blends 5 and 10. The search, origins, step 6 and parking used 3; finalists used 5; CatBoost used 2; the rule replay used 10. All are noted above. No result comes from a killed or partial run: rows are only written after every seed and window finishes, and the saved predictions were checked for all four windows.
- **Reduced from the plan because of time and shared CPU:**
  - Search checkpoints stop at 600 rounds, or 1,600 at learning rate 0.02.
  - CatBoost is one config, learning rate 0.1, with no tuning.
  - The median regression fits a fixed sample of 8,000 rows per window.
  - Steps 4 to 7 ran at current parameters, not tuned ones; tuning found nothing to carry over.
- **Bias carry rows** cover I2, I1 and B only, because I3 has no earlier window here.
- **`was_leased` itself was not tested.** The panel lacks it, so step 4 tests age of rows only.
- **Checks:** the harness reproduces the published $42.62 and 81.03% exactly at seed 7. The rebuilt anchors match the panel exactly at each row's own parking. `src/features.py` is unchanged.
- **Scripts:** `local/methods_*.py`. Panel and predictions are in `out/real/` and stay untracked.
