"""Pool the oculomotor-suppression studies into one saccade-level table.

Input: the per-study fixation reports from Drennan & Gaspelin (2026),
"What can a half-million saccades tell us about distractor suppression?"
(--data_dir; not part of this repo). Output: saccades.csv, one row per
scoreable saccade (indices 1-5), plus events.csv, one row per trial with
the target/singleton locations that drive the history traces.

Decisions:
- Saccades 1-5, each a conditional-choice trial from the current fixation
  (the landing position of the previous saccade; screen center for #1).
- Trials truncate at the first target fixation (later saccades are
  responding, not searching). Saccades landing on the currently fixated
  item (corrective) are dropped and counted.
- Practice trials are excluded ALTOGETHER (no saccades, no event rows):
  the memory traces start cold at each subject's first experimental
  trial.
- Trials with other distractor types - abrupt onsets, singleton-onsets,
  motion singletons - are excluded entirely (the analysis is
  color-singleton present vs. absent, as in the source paper); the
  number of dropped trials is printed per study.
- Gaspelin & Luck 2018 E4 (studyID 12 here) alternates singleton colors
  each block, which disrupts the fixed task set the model assumes; it is
  excluded, as in the source paper's suppression/priming analyses.
- Item coordinates per (study, location) are the median landing position
  of fixations classified to that location - self-consistent with the
  original nearest-object assignment.

Usage:
  python pool_data.py --data_dir ".../search_data/Data Files" --out_dir dataset
"""

import argparse
import os

import numpy as np
import pandas as pd

EXCLUDE_STUDYNAMES = {"Gaspelin_2018"}
CENTER = (960.0, 540.0)


def load_study(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    need = ["studyName", "subjNum", "block", "practice", "trial", "saccindex",
            "currloc", "curritem", "normX", "normY", "targLoc", "singLoc",
            "singPres", "setsize", "keepTrial", "ACC", "targCol", "singCol"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} missing column {c}")
    if "singType" in df.columns:
        # only color-singleton-present and singleton-absent trials are
        # analyzed (as in the source paper); trials with other
        # distractor types (abrupt onsets, singleton-onsets, motion
        # singletons) are excluded entirely
        ok = df.singType.isin(["sing", "absent", "Abs"]) | df.singType.isna()
        dropped = df[~ok]
        if len(dropped):
            n_tr = dropped.groupby(
                ["subjNum", "block", "trial"]).ngroups
            kinds = sorted(dropped.singType.dropna().unique())
            print(f"  excluded {n_tr} other-distractor trials "
                  f"(types: {', '.join(kinds)})")
        df = df[ok].copy()
        df["color_sing"] = df.singType.eq("sing")
        if "singSal" in df.columns:
            # Stilwell varied singleton salience; the canonical
            # reconstruction (salient red-on-green) only matches the
            # high-salience displays, so low-salience singleton
            # trials are excluded (absent trials are kept)
            low = df.color_sing & df.singSal.eq("low")
            if low.any():
                n_tr = df[low].groupby(
                    ["subjNum", "block", "trial"]).ngroups
                print(f"  excluded {n_tr} low-salience singleton trials")
                df = df[~low].copy()
    else:
        df = df.copy()
        df["color_sing"] = df.singPres.eq("P")
    return df


def item_coordinates(df):
    """Median landing position per (setsize, location)."""
    fx = df[(df.saccindex >= 1) & df.normX.notna() & (df.currloc >= 1)]
    coords = fx.groupby(["setsize", "currloc"])[["normX", "normY"]].median()
    return {(int(s), int(l)): (r.normX, r.normY) for (s, l), r in coords.iterrows()}


def process_study(df, study_ord):
    sacc_rows, event_rows, n_corrective = [], [], 0
    coords = item_coordinates(df)
    if coords:
        radius = float(np.median([np.hypot(x - CENTER[0], y - CENTER[1])
                                  for x, y in coords.values()]))
    else:
        # no gaze coordinates in this study (first saccades only): synthetic
        # iso-eccentric ring; saccade-1 distances are equal by design, so
        # only the (shared) envelope constant is affected
        radius = 240.0
        for s in df.setsize.dropna().unique():
            s = int(s)
            for j in range(1, s + 1):
                a = 2 * np.pi * (j - 1) / s
                coords[(s, j)] = (CENTER[0] + radius * np.cos(a),
                                  CENTER[1] + radius * np.sin(a))
    dnorm = 2.0 * radius
    df = df.copy()
    df["block"] = df["block"].fillna(0)
    df = df.sort_values(["subjNum", "block", "trial", "saccindex"])
    for (subj, block, trial), tr in df.groupby(["subjNum", "block", "trial"], sort=False):
        head = tr.iloc[0]
        setsize = int(head.setsize)
        if setsize < 4 or setsize > 6:
            continue
        targ = int(head.targLoc)
        sing = int(head.singLoc) if (head.singPres == "P"
                                     and head.color_sing) else 0
        if head.practice == "Y":
            continue                      # practice excluded altogether
        keep = int((tr.keepTrial == 1).any())
        event_rows.append(dict(study=study_ord, subj=subj, block=block,
                               trial=trial, setsize=setsize, targLoc=targ,
                               singLoc=sing))
        if not keep:
            continue
        cur_x, cur_y, cur_loc = CENTER[0], CENTER[1], 0
        for _, r in tr[tr.saccindex >= 1].iterrows():
            k = int(r.saccindex)
            if k > 5 or pd.isna(r.currloc):
                break
            dest = int(r.currloc)
            if dest < 1 or dest > setsize:
                break
            if dest == cur_loc:
                n_corrective += 1
                if pd.isna(r.normX):
                    break
                cur_x, cur_y = r.normX, r.normY
                continue
            d = [np.hypot(coords[(setsize, j)][0] - cur_x,
                          coords[(setsize, j)][1] - cur_y) / dnorm
                 if (setsize, j) in coords else np.nan
                 for j in range(1, setsize + 1)]
            if any(np.isnan(d)):
                break
            sacc_rows.append(dict(
                study=study_ord, subj=subj, block=block, trial=trial,
                saccindex=k, setsize=setsize, choice=dest, fixloc=cur_loc,
                targLoc=targ, singLoc=sing,
                targCol=str(head.targCol).lower(),
                singCol=str(head.singCol).lower(),
                **{f"d{j}": d[j - 1] for j in range(1, setsize + 1)}))
            if dest == targ or pd.isna(r.normX):
                break
            cur_x, cur_y, cur_loc = r.normX, r.normY, dest
    return sacc_rows, event_rows, n_corrective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="dataset")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_sacc, all_ev, ncorr = [], [], 0
    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".txt"))
    for i, f in enumerate(files):
        df = load_study(os.path.join(args.data_dir, f))
        name = df.studyName.iloc[0]
        if name in EXCLUDE_STUDYNAMES:
            print(f"skip {f} ({name}: block-alternating colors)")
            continue
        s, e, c = process_study(df, name)
        all_sacc += s
        all_ev += e
        ncorr += c
        print(f"{f}: {len(s)} saccades, {len(e)} trials")
    sacc = pd.DataFrame(all_sacc)
    ev = pd.DataFrame(all_ev)
    sacc.to_csv(f"{args.out_dir}/saccades.csv", index=False)
    ev.to_csv(f"{args.out_dir}/events.csv", index=False)
    print(f"\ntotal: {len(sacc)} saccades ({ncorr} corrective dropped), "
          f"{len(ev)} trials, {ev.groupby(['study','subj']).ngroups} subjects")
    print(sacc.saccindex.value_counts().sort_index())


if __name__ == "__main__":
    main()
