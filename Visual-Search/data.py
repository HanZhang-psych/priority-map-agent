"""Shared data loading for the search model.

build_tensors() turns dataset/saccades_ctx.csv + dataset/events.csv
into the tensors every fit and analysis uses: per-saccade choice sets,
choices, item roles, distances, and the
per-subject trial-ordered event maps that drive the memory traces.
"""

import numpy as np
import pandas as pd
import torch

NLOC = 6


def load_frames(saccades="dataset/saccades_ctx.csv",
                events="dataset/events.csv"):
    sacc = pd.read_csv(saccades, low_memory=False)
    ev = pd.read_csv(events, low_memory=False)
    return sacc, ev


def build_tensors(sacc, ev):
    # chunked CSV parsing can leave key columns with mixed int/str values,
    # which silently breaks the merge - force numeric
    for df in (sacc, ev):
        df["block"] = pd.to_numeric(df["block"], errors="coerce").fillna(0.0)
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df["subj"] = df["subj"].astype(str)
    ev = ev.sort_values(["study", "subj", "block", "trial"]).reset_index(drop=True)
    ev["skey"] = ev.study.astype(str) + "|" + ev.subj.astype(str)
    subjects = {s: i for i, s in enumerate(ev.skey.unique())}
    ev["si"] = ev.skey.map(subjects)
    ev["ti"] = ev.groupby("si").cumcount()
    S, T = len(subjects), int(ev.ti.max()) + 1
    eT = torch.zeros(S, T, NLOC)
    eD = torch.zeros(S, T, NLOC)
    eT[ev.si.values, ev.ti.values, ev.targLoc.values - 1] = 1.0
    has_sing = ev.singLoc.values > 0
    eD[ev.si.values[has_sing], ev.ti.values[has_sing],
       ev.singLoc.values[has_sing] - 1] = 1.0

    key = ["study", "subj", "block", "trial"]
    n_before = len(sacc)
    sacc = sacc.merge(ev[key + ["si", "ti"]], on=key, how="inner")
    if len(sacc) != n_before:
        raise RuntimeError(f"merge lost {n_before - len(sacc)} saccades")
    # model scope: first fixations only
    sacc = sacc[sacc.saccindex == 1].reset_index(drop=True)

    N = len(sacc)
    d = torch.ones(N, NLOC)
    valid = torch.zeros(N, NLOC, dtype=torch.bool)
    for j in range(1, NLOC + 1):
        col = f"d{j}"
        if col in sacc.columns:
            v = sacc[col].notna().values & (sacc.setsize.values >= j)
            d[v, j - 1] = torch.tensor(sacc.loc[v, col].values,
                                       dtype=torch.float32)
            valid[:, j - 1] = torch.tensor(v)
    rows = torch.arange(N)
    fix = torch.tensor(sacc.fixloc.values)
    fixated = fix > 0
    valid[rows[fixated], fix[fixated] - 1] = False
    choice = torch.tensor(sacc.choice.values) - 1

    return sacc, dict(eT=eT, eD=eD, si=torch.tensor(sacc.si.values),
                      ti=torch.tensor(sacc.ti.values), d=d, valid=valid,
                      choice=choice)
