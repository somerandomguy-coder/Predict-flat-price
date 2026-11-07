# Foundation models against trees

**Verdict: no single model beats trees reliably, but a blend does, by about $1 to $2. It is worth one Fold A check, not a switch yet.**

- **Nothing alone is a clear winner.** TabICL on 5,000 rows led LightGBM by $1.65 on Fold B, but only $0.39 on a second window (the holdout below). TabM and RealMLP were also within noise of trees.
- **A blend beat the current model on both windows.** CatBoost + TabICL + RealMLP, equal weights: $41.17 against $43.00 on Fold B (gap -$1.83, interval -$3.02 to -$0.60), and $43.69 against $45.13 on the holdout (gap -$1.44, -$3.11 to -$0.09).
- **The blend is not just averaging.** Against a LightGBM + CatBoost blend it gained $0.96 on Fold B (-$1.96 to +$0.18, inside the noise) and $1.78 on the holdout (-$3.01 to -$0.23).
- **Discount that gain.** 23 blends were compared on these same windows, and this one was picked after seeing them, so the true gain is probably nearer $1.
- **The real LightGBM baseline is $43.00, not $42.62.** That is the 3-seed mean. Row order alone moved one seed by $0.52 ($42.62 to $43.14 on identical data), so $42.62 was a good draw.
- **Time-series models: no.** Holding the index flat beat Chronos-2 and Chronos-Bolt over 13 origins.
- **First-time flats did not clearly improve.** No improvement has an interval clear of zero.

**Is it worth the complexity?** Possibly, if it survives Fold A. It needs torch and two more models, and about 40 minutes of CPU for TabICL on a full prediction run (estimated from the timings below, not measured). The team should score the blend on Fold A once, and re-check the parking premium, before changing the submission. Fold A was not read here, and the submission was not touched.

## Results (Fold B, 2,210 adverts)
Same 33 features, same rows, scored with `src.validate.score` through the two-pass prices. Each figure is a mean over 3 seeds. Intervals are paired bootstraps over test adverts (2,000 draws), so they show test-row noise, not all sources of doubt. Log target throughout unless stated.

**Single models, against LightGBM on all rows**

| Model | Training rows | Median error | Seed SD | Within $100 | Gap to LightGBM (95% interval) | First-time median error |
|---|---|---|---|---|---|---|
| LightGBM | all 27,985 | $43.00 | 0.19 | 81.1% | reference | $59.02 |
| CatBoost | all | $42.25 | 0.08 | 81.3% | -$0.74 (-1.99, +0.28) | $57.35 |
| XGBoost | all | $43.42 | 0.25 | 81.5% | +$0.42 (-0.71, +1.29) | $58.62 |
| TabM, L1 | all | $41.90 | 0.22 | 81.7% | -$1.09 (-2.47, +0.39) | $58.41 |
| RealMLP, L1 | all | $42.00 | 0.27 | 81.0% | -$0.99 (-2.27, +0.33) | $56.24 |
| TabICLv2, median | 5,000 | $41.34 | 0.33 | 81.0% | -$1.65 (-3.13, -0.26) | $57.38 |
| TabICLv2, median | 2,000 | $42.64 | 0.91 | 80.4% | -$0.35 (-1.84, +1.00) | $56.76 |
| TabPFN v2, median | 2,000 | $42.75 | 0.67 | 80.2% | -$0.24 (-1.78, +1.19) | $58.32 |
| Comparables, 50 nearest | all | $48.47 | 1 run | 77.5% | +$5.64 (+3.26, +8.61) | $60.36 |

TabICL and TabPFN used one ensemble member; their defaults use more. First-time flats are about 550 adverts, so their intervals are wide.

**Same subsample, for a fair comparison.** Foundation models use small data far better than trees:

| Rows | LightGBM | CatBoost | XGBoost | TabICLv2 | TabPFN v2 | Foundation against LightGBM |
|---|---|---|---|---|---|---|
| 2,000 | $46.53 | $45.84 | $47.94 | $42.64 | $42.75 | TabICL -$3.89 (-5.41, -2.37); TabPFN -$3.78 (-5.33, -2.02) |
| 5,000 | $45.01 | $44.07 | $45.19 | $41.34 | not run | TabICL -$3.66 (-5.05, -2.28) |
| 10,000 | $44.01 | $43.01 | $44.67 | killed, no result | not run | none |

**Target form and output.** Median error on Fold B.

| Model | Rows | Log ratio | Rent | Log ratio, mean output |
|---|---|---|---|---|
| LightGBM | all | $43.00 | $43.48 | |
| CatBoost | all | $42.25 | $43.30 | |
| XGBoost | all | $43.42 | $43.89 | |
| TabICLv2 | 5,000 | $41.34 | $44.16 | $42.07 |
| TabPFN v2 | 2,000 | $42.75 | $45.29 | $43.70 |

Log ratio won for every model: by $0.47 to $1.04 for trees on all rows, and by $2.54 to $2.81 for the foundation models. Median output beat mean output in all six foundation-model comparisons across both windows, by $0.31 to $0.95.

**The holdout: a second window.** Fold B has been looked at many times. So the same models were also trained on the 11 older origins and scored on the last training origin (1,777 adverts, February-March 2026). It uses no Fold A rows.

| Model | Fold B | Holdout | Gap to LightGBM, Fold B | Gap to LightGBM, holdout |
|---|---|---|---|---|
| LightGBM | $43.00 | $45.13 | reference | reference |
| CatBoost | $42.25 | $45.74 | -$0.74 (-1.99, +0.28) | +$0.62 (-1.03, +1.91) |
| LightGBM + CatBoost | $42.13 | $45.46 | -$0.87 (-1.67, -0.18) | +$0.33 (-0.89, +1.17) |
| TabICLv2, 5,000 | $41.34 | $44.74 | -$1.65 (-3.13, -0.26) | -$0.39 (-2.30, +1.14) |
| TabPFN v2, 2,000 | $42.75 | $44.35 | -$0.24 (-1.78, +1.19) | -$0.78 (-2.74, +1.21) |
| TabM | $41.90 | $44.78 (2 seeds) | -$1.09 (-2.47, +0.39) | -$0.05 (-2.27, +1.52) |
| RealMLP | $42.00 | $45.68 | -$0.99 (-2.27, +0.33) | +$0.55 (-1.53, +2.13) |
| **CatBoost + TabICL + RealMLP** | **$41.17** | **$43.69** | **-$1.83 (-3.02, -0.60)** | **-$1.44 (-3.11, -0.09)** |
| All five (2 seeds on holdout) | $41.16 | $43.50 | -$1.84 (-2.88, -0.63) | -$1.33 (-2.81, +0.10) |

Blend weights were fixed at equal. Weights chosen on the holdout did no better on Fold B: 0.6 on TabICL with CatBoost gave $41.26, against $41.18 at equal weights. The plain tree blend did not carry over to the holdout, so the ensemble lane should check that too.

**Rent index forecast.** 13 origins, 33 suburbs plus an all-suburb series, each forecast seeing only months up to its origin.

| Forecast | Index error at 2 months (log points) | Anchor x forecast, all origins | Anchor x forecast, Fold B |
|---|---|---|---|
| **Hold flat** | **1.25** | **$57.68** | $52.10 |
| Chronos-Bolt small | 1.37 | $57.78 | $51.42 |
| Chronos-2 | 1.50 | $57.86 | $52.30 |
| Same months last year | 1.60 | $58.09 | $52.56 |
| 6-month drift | 1.60 | $58.19 | $54.60 |

Adding the forecast as a LightGBM feature gave -$0.72 for Chronos-Bolt (-1.71, +0.14) and -$0.43 for Chronos-2 (-1.34, +0.46): under $1 and inside the noise. Chronos-Bolt was $0.68 ahead on the Fold B origin alone, but flat won across all origins. Keep the index flat, which agrees with decision 14. The index is smoothed, which may favour flat a little, but the anchor columns use real adverts and agree.

**Context choice.** The most recent 2,000 rows were much worse than a random 2,000 for TabICL ($46.99 against $42.64) and for trees, with bias near +$19. Use a random spread of origins.

## What each model needs
Local runs: CPU only, 1 to 2 threads, so times are rough. They are per fit, on about 2,300 test rows for TabPFN and TabICL. A full run of 65,777 flats in two passes is 57 times that many rows.

| Model | Package tested | Licence (code / weights) | Gate | Time here | Memory | Median or L1 |
|---|---|---|---|---|---|---|
| TabPFN v2 | tabpfn 9.2.0, `ModelVersion.V2` | Apache-2.0 / Apache-2.0 plus attribution | None. Downloaded with no token | 210s at 2,000 rows, 1 member | Under 1.5 GB | Yes: `output_type="main"` gives mean, median, quantiles |
| TabPFN 2.5, 2.6, 3, 3.5 | Not run | Apache-2.0 / non-commercial | **Skipped.** `ensure_license_accepted` needs a licence click and token | | | Yes |
| TabICLv2 | tabicl 2.2.0 | BSD-3 / BSD-3 on the model page; weights scope not confirmed | None | 41s at 5,000 rows, 1 member | Over 1.5 GB at 5,000 rows, over 2.8 GB at 10,000 | Yes: `output_type=["mean","median"]` |
| TabM | tabm 0.0.3 | Apache-2.0 | None, no weights | 185s, capped, about 20 epochs | Under 2.8 GB | L1 per member, median of members |
| RealMLP | pytabkit 1.8.0 | Apache-2.0 | None, no weights | 65s at 64 epochs | Under 2.8 GB | `train_metric_name="pinball(0.5)"` |
| Chronos-2, Chronos-Bolt | chronos-forecasting 2.3.2 | Apache-2.0 | None | Seconds | Small | Median quantile |
| LightGBM, CatBoost, XGBoost | 4.7.0, 1.2.10, 3.4.1 | MIT, Apache-2.0, Apache-2.0 | None | 7s, 30s, 11s on all rows | Small | L1 objective |

Privacy check: tabpfn 9.2.0 has no telemetry code. Its network calls are weight downloads and the licence check for gated versions, which were not triggered. `tabpfn-client` was not installed.

## What failed, or was not done
**Killed runs are not results.**
- TabICL at 10,000 rows hit the 2.8 GB job cap within seconds. It has no score, so the size trend stops at 5,000.
- The first TabICL 5,000 attempt hit a 1.5 GB cap. It was rerun in full under 2.8 GB, and only the rerun is reported.
- The first two notebook tests ran on the system Python by mistake, so the install cell started `pip` there. One hit the 1.5 GB cap and one was stopped by hand. Nothing was installed. The cell now installs only on Colab, and the test uses the separate venv.
- TabM crashed on one seed because a column was constant in that split. That was fixed, and all three TabM seeds are from one configuration.

**Not a fair limit of the model.**
- TabM stopped at a 180-second cap, after about 20 epochs, and may have had more to give. RealMLP ran 64 epochs, not its default 256. Neither was tuned.
- TabPFN ran only at 2,000 rows with one member, because it is slow on CPU.
- With the recent rows, TabICL sees the same rows every seed, so those three runs were identical.

**Not run.** TabPFN 2.5 and later (gated, skipped as told), TabDPT, Mitra, LimiX, FT-Transformer, ModernNCA, TabR, AutoGluon, TimesFM, and Moirai 2 (research-only licence). Model predictions used as stacking features, rather than a blend, were not tried.

**Limits of this evidence.**
- Trees here use one fixed setting each. The tuning lane may narrow the gap.
- The holdout is one window, close in time to Fold B, and its rows are also training rows for Fold B. It was new to this comparison, but it is not a third independent fold.
- TabM's holdout has 2 seeds.
- For TabICL and TabPFN, most adverts had only their scored parking pass computed, to save time. This cannot change a score (checked on LightGBM and CatBoost, identical), but it means the premium for all flats was not measured. A submission would need both passes, and the premium rule (decision 15) was set on LightGBM.
- The index trial loads the whole lease file, then drops everything after 31 May straight away. Fold A months are never used, but they pass through memory in that step.

## What to run on Colab
`colab/foundation_models.ipynb`, on a GPU runtime, reading only `features_foldB.parquet`. Setup is in `colab/README.md`. Colab is hosted compute, so this assumes the team is happy to keep these rows in the private Drive folder, as it already does.

**Tested:** the notebook ran end to end on CPU through nbconvert, with all seven models on a tiny sample (300 training rows, 120 adverts; those scores mean nothing). An earlier build also ran LightGBM on all rows through the notebook's code: $43.14, the row-order effect in the verdict. `features_foldB.parquet` matched the panel on all 30,195 rows and every feature.

**Unrun:** the Drive mount, the GPU path and the Colab package install have never been run, so a version clash or memory limit there is possible.

Questions it answers, in priority order:
1. Does TabICL keep improving at 10,000 rows and all rows, with 4 members? This is the biggest open question, because 2,000 to 5,000 rows gained $1.30.
2. Does TabM improve with a longer training budget?
3. Does TabPFN v2 at 10,000 rows with 4 members catch TabICL?
4. Do blends improve further? The notebook prints fixed 50/50 blends with LightGBM.

If TabICL or TabM moves more than $1 at full size, repeat the holdout check before trusting it. The notebook does not run the holdout yet; `local/run_foundation.py run --val` does. Fold A stays for one final check by the team.

## What suits this problem
- **Small data.** This is where in-context models are strongest, and the trial agrees: on the same 2,000 or 5,000 rows they are far ahead of trees. The catch is that we have 28,000 rows, and trees use them.
- **Median metric.** A model that outputs a distribution can give its median directly. That is a free fix: the median beat the mean by $0.31 to $0.95 from the same fit, in every comparison. Anyone using these models with the default `predict()` gets the mean and loses that.
- **Heavy tail and anchors.** log(rent / anchor) beat rent for every model at every size, on seed means. The anchor removes the flat's level, so the model only has to learn a small, well-behaved ratio.
- **Comparables.** Our anchors already are same-flat and same-building comparables, so an explicit retrieval model has little left to find. A plain 50-neighbour comparables model lost clearly. Retrieval-style deep models (TabR, ModernNCA) were not run.

## Literature
A review of papers and model pages. Nothing here was tested by us unless the section above says so. Elo is comparable only inside one table.

**No published source tests our setup:** a tree model trained with L1 on a residual target, scored on median absolute error, against a foundation model. Leaderboard Elo comes from RMSE, so a benchmark win does not transfer to our metric.

| Model | Reported against tuned trees | Limits | CPU | Note |
|---|---|---|---|---|
| TabPFN v2 | TabArena regression: tuned and ensembled rank 13.0, below CatBoost 11.0 (partly imputed) | 10k rows, 500 features | Slow; needs an override above 1,000 rows | The only open-weights TabPFN |
| TabPFN 2.5 to 3.5 | 3.5 Elo 2103 against AutoGluon 2056, in the vendor's report | 100k to 1M rows | GPU | Non-commercial weights. The 3.5 licence covers competitions only if results are not used commercially |
| TabICLv2 | Elo 1672 on TabArena regression. TALENT MAE rank 4.63 against CatBoost 12.38 | Good to about 100k rows | Yes, and fast | Weak on the closest public analogue below, though not on our data |
| TabDPT | Elo 1748, but TabICLv2's authors question its regression score | 2,048 context by retrieval | Unverified | Not run |
| Mitra | v1 rank 15.04 on regression. v2 reports Elo 1986, own run | v1 10k; v2 32k | v2 not practical | Not run |
| TabM | Tuned Elo 1463, below CatBoost 1513. Best tabular deep model on TabReD per its repo | None | Slow but works | Temporal benchmark evidence is good |
| RealMLP | Tuned Elo 1636 against CatBoost 1479. First on BeyondArena temporal | None | Works | Best-supported licence-clean option |
| FT-Transformer | TALENT MAE rank 13.04 against CatBoost 12.38 | Quadratic in features | Slow | Not run |
| ModernNCA, TabR | Mid-table, weak on TabReD, recovered with better validation splits | Not stated | GPU mostly | Not run |
| AutoGluon 1.6 | `extreme` Elo 2002 | Best under 100k rows | `best` preset has no foundation models | Time-aware folds unverified. Not run |

Time series, on fev-bench MASE skill against seasonal naive (pinned commit): Chronos-2 35.5, TimesFM 2.5 30.2, Moirai 2 27.3 (CC-BY-NC, research only), Chronos-Bolt 26.5. On FRED-MD monthly macro series, Chronos-2 beat AR(1) by about 5% at one month, TimesFM did not, and worst-case series were up to 48% worse. Some models trained on FRED-MD. Nothing was found on short monthly rent series.

**Temporal shift, the closest evidence.** On `sberbank_housing` (temporal, 27,195 rows, RMSE): TabPFN-2.6 0.243, RealMLP 0.247, XGBoost 0.250, LightGBM 0.251, TabDPT 0.261, TabICLv2 0.285. That is one dataset, and the top four overlap. BeyondArena's lead for tuned models on temporal data is mostly a dataset-size effect.

**Where it is hype**
- Headline wins ("state of the art on all 7 benchmarks", 96% win rate) are vendor-run, and four of the benchmarks use an API-only variant we could not use anyway.
- The benchmarks are not independent of the vendors. BeyondArena's lead and senior authors are at Prior Labs. Its temporal finding is still unfavourable to foundation models, which adds credibility.
- TabArena has 13 regression datasets, is IID by design and scores RMSE.
- Contamination is open: pretraining on benchmark data is possible for some models.
- Stacking evidence is thin. Foundation-model-only pools are nearly redundant, and no regression study of foundation models plus trees under temporal validation was found.
- Time-series gains shrink towards flat when the signal is weak, which is what we saw.
- The TabArena live page did not render, so all figures come from papers.

Sources: TabPFN [repo](https://github.com/PriorLabs/TabPFN), [models](https://docs.priorlabs.ai/models), [3.5 licence](https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE); [TabICL](https://github.com/soda-inria/tabicl); [pytabkit](https://github.com/dholzmueller/pytabkit); [TabM](https://github.com/yandex-research/tabm); [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference); [Chronos-2](https://huggingface.co/amazon/chronos-2); [fev-bench](https://huggingface.co/spaces/autogluon/fev-bench); arXiv [2506.16791](https://arxiv.org/abs/2506.16791) (TabArena), [2606.30410](https://arxiv.org/abs/2606.30410) (BeyondArena), [2609.17895](https://arxiv.org/abs/2609.17895) (TabPFN-3.5), [2602.11139](https://arxiv.org/abs/2602.11139) (TabICLv2), [2406.19380](https://arxiv.org/abs/2406.19380) (TabReD), [2502.20260](https://arxiv.org/abs/2502.20260), [2605.18696](https://arxiv.org/abs/2605.18696), [2407.00890](https://arxiv.org/abs/2407.00890).
