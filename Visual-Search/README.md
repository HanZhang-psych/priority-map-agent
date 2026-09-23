# Visual-Search: the priority field fitted to human saccades

The priority-map model of visual search (paper: Results, "A priority map
for visual search"; Methods, "The priority-map model") — the search
counterpart of the reach-avoid agent (`Reach-Avoid/model/goal_es2.py`),
fitted to human first fixations.

Data: the pooled oculomotor-suppression studies from Drennan & Gaspelin
(2026, *Cognition*), "What can a half-million saccades tell us about
distractor suppression?" — 12 eye-tracking studies, N = 354. The
per-study fixation reports are NOT part of this repo (OSF:
https://osf.io/q27ph/); point `pool_data.py` at their folder.

## Model (4 fitted weights)

```
F_i = w_S*S_i + w_G*G_i + w_H*H_i
P(saccade -> i) = softmax over the current choice set
```

| Weight | Meaning | Agent counterpart |
| --- | --- | --- |
| w_S | goal-independent sensory color-salience gain | sensory/obstacle gain |
| w_G | unified green-circle template evidence gain | goal gain block |
| w_H | target history-field expression weight | history gain |
| eta_H | target trace accrual rate | memory update rate |

Scope: first saccades only, launched from the display center; positions
only (no latency); one population-level fit (no individual differences);
bare field -> softmax readout (no motor-repetition, no lapse); no
attention window because all first-saccade items are equidistant from
fixation. Practice trials are excluded altogether (the memory traces
start cold at each subject's first experimental trial); the likelihood
scores kept trials. Trials with other distractor types (abrupt onsets,
singleton-onsets, motion singletons) are excluded entirely — the
analysis is color-singleton present vs. absent, as in the source paper
— with dropped counts printed at pooling time.
Gaspelin & Luck 2018 E4 is excluded (block-alternating singleton colors
break the fixed-task-set assumption), as in the source paper's
suppression/priming analyses.

## Files

| File | Purpose |
| --- | --- |
| `pool_data.py` | Fixation reports → `dataset/saccades.csv` + `dataset/events.csv` |
| `front_end.py` | Display rendering and item geometry |
| `build_contexts.py` | Reconstructs every unique display and precomputes item evidence (`dataset/senses.npz` + `dataset/saccades_ctx.csv`) |
| `data.py` | Shared tensors (choice sets, distances, visited, trial-ordered events) + the subject split |
| `tutorial_visual_search.ipynb` | Teaching notebook: builds, trains, evaluates the final model on the real data |

## Reproduce

```bash
python pool_data.py --data_dir "<...>/search_data/Data Files" --out_dir dataset
python build_contexts.py
jupyter lab tutorial_visual_search.ipynb
```

## Diagnostics reported in the notebook

- Observed vs model-implied first-saccade rates (target / singleton /
  per-item nonsingleton baseline) — the oculomotor suppression effect.
- Held-out fit quality and learned weights.
- Target-capture ANOVA for the target statistical-learning manipulation.
