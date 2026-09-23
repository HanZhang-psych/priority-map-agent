"""Reconstruct displays and precompute the model's item evidence.

One pass: read dataset/saccades.csv (from pool_data.py), normalize
colors, assign a context id to every unique (setsize, targLoc,
singLoc, targCol, singCol, fixation) combination, render each unique
display once (feature search: target circle among heterogeneous
nontarget shapes, singleton in the opposite color), and compute one row
of item evidence per display item.

The fitted model is item-based: it consumes A[ctx, item, field] directly
and adds target history at the same item index. Pixel maps remain here
only to support visualization and shape/color footprint construction.

Reconstruction assumptions: per-trial set size, colors, and item
distances come from the data files; shapes, item size, and background
are paper-sourced (the OSF trial files carry no display parameters).
All displays use the canonical green-target / red-singleton scheme
(see normalize_colors).

Channel units: P is unsigned item color distinctiveness
in [0, 1]; T is a unified template-match field in [-1, 1], combining
signed color match with signed shape match so a red square is more
negative than a red circle; S_T is kept as a diagnostic target-shape
similarity value in [0, 1].

Output: dataset/senses.npz -- A [ctx, 6, 3] (item fields: bottom-up
color salience P, unified template-match evidence T, diagnostic
target-shape evidence S_T), GREY [1] (the color constant, for the
record). Plus dataset/saccades_ctx.csv (saccades with ctx ids).

Usage: python build_contexts.py
"""

import numpy as np
import pandas as pd

from front_end import COLORS, IMG, ITEM_R, item_positions, render, shape_for

def normalize_colors(sacc):
    """Canonical color scheme: each subject's colors were fixed for
    their whole session, so the model only ever sees match-vs-mismatch
    structure - every display is reconstructed with a GREEN target
    color and a RED singleton. (Recorded cost: Stilwell 2023's
    within-study singleton-salience color manipulation is invisible to
    this reconstruction.)"""
    sacc = sacc.copy()
    sacc["targCol"] = "green"
    sacc["singCol"] = np.where(sacc.singLoc.values > 0, "red", "none")
    return sacc


def shape_kernels(r_px):
    """Binary footprint kernels at item scale for each shape."""
    n = int(3.2 * r_px) | 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    ks = {}
    ks["circle"] = ((xx - c) ** 2 + (yy - c) ** 2 < r_px ** 2)
    ks["square"] = (np.abs(xx - c) < 0.95 * r_px) & (np.abs(yy - c) < 0.95 * r_px)
    ks["diamond"] = (np.abs(xx - c) + np.abs(yy - c)) < 1.3 * r_px
    ks["triangle"] = ((yy - c > -0.9 * r_px)
                      & (np.abs(xx - c) < 0.95 * r_px
                         * (1 - (yy - c + 0.9 * r_px) / (2.0 * r_px))))
    ks["cross"] = (((np.abs(xx - c) < 0.45 * r_px) & (np.abs(yy - c) < 1.1 * r_px))
                   | ((np.abs(yy - c) < 0.45 * r_px) & (np.abs(xx - c) < 1.1 * r_px)))
    return {k: v.astype(float) for k, v in ks.items()}


def shape_match_map(img, template_shape="circle"):
    """PIXEL-DERIVED target-shape evidence with transparent background.

    The rendered image is binarized into object footprints, connected
    components are scored against the same-scale target-shape kernel,
    and every component is painted back with its target-shape similarity.
    The score discounts generic compact-object overlap, so a square can
    carry partial circle evidence without looking almost target-like.
    The result is a crisp footprint map in [0, 1]: the true target shape
    is strongest, similar nontargets still carry partial evidence, and
    empty background remains at 0.

    Remaining scope limit: the trial data never record item shapes, so
    displays are reconstructed with the template at the target's
    location; this channel makes the EVIDENCE pathway realistic, not
    the display's provenance."""
    from front_end import BG, ITEM_R
    scale = img.shape[0] // IMG        # supports hi-res renders
    r_px = ITEM_R / 1.5 * IMG * scale

    dev = np.sqrt(((img - np.array(BG)) ** 2).sum(-1))
    dev = (dev > 0.15).astype(float)   # binarize: shape, not color amplitude
    mask = dev.astype(bool)
    out = np.zeros_like(dev)
    yy, xx = np.mgrid[0:dev.shape[0], 0:dev.shape[1]]
    seen = np.zeros_like(mask, dtype=bool)
    ks = shape_kernels(r_px)
    kshape = next(iter(ks.values())).shape
    kc = kshape[0] // 2

    def score_component(comp, cx, cy, kern):
        x0, x1 = int(round(cx)) - kc, int(round(cx)) - kc + kshape[1]
        y0, y1 = int(round(cy)) - kc, int(round(cy)) - kc + kshape[0]
        patch = np.zeros(kshape, dtype=bool)
        sx0, sx1 = max(0, x0), min(mask.shape[1], x1)
        sy0, sy1 = max(0, y0), min(mask.shape[0], y1)
        if sx0 >= sx1 or sy0 >= sy1:
            return 0.0
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = comp[sy0:sy1, sx0:sx1]
        inter = np.logical_and(patch, kern).sum()
        union = np.logical_or(patch, kern).sum()
        return inter / max(union, 1)

    for y_start, x_start in zip(*np.nonzero(mask & ~seen)):
        if seen[y_start, x_start]:
            continue
        stack = [(int(y_start), int(x_start))]
        seen[y_start, x_start] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for yn in range(max(0, y - 1), min(mask.shape[0], y + 2)):
                for xn in range(max(0, x - 1), min(mask.shape[1], x + 2)):
                    if mask[yn, xn] and not seen[yn, xn]:
                        seen[yn, xn] = True
                        stack.append((yn, xn))
        pts = np.asarray(pts)
        cy, cx = pts.mean(axis=0)
        comp = np.zeros_like(mask, dtype=bool)
        comp[pts[:, 0], pts[:, 1]] = True
        scores = {sh: score_component(comp, cx, cy, kern.astype(bool))
                  for sh, kern in ks.items()}
        relative = scores[template_shape] / max(max(scores.values()), 1e-9)
        out[comp] = np.clip((relative - 0.60) / 0.40, 0.0, 1.0) ** 2.0

    return out / out.max() if out.max() > 0 else out


def item_shape_similarity(shape, template_shape="circle"):
    """Template-shape similarity for an item label, in [0, 1]."""
    r_px = ITEM_R / 1.5 * IMG
    ks = {k: v.astype(bool) for k, v in shape_kernels(r_px).items()}
    target = ks[template_shape]
    other = ks.get(shape, ks["circle"])
    inter = np.logical_and(other, target).sum()
    union = np.logical_or(other, target).sum()
    relative = inter / max(union, 1)
    return float(np.clip((relative - 0.60) / 0.40, 0.0, 1.0) ** 2.0)


def signed_shape_match_map(items, template_shape="circle"):
    """Signed shape match in [-1, 1] as hard item footprints."""
    m = np.zeros((IMG, IMG))
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    for it in items:
        sim = item_shape_similarity(it.get("shape", "circle"), template_shape)
        m[item_footprint_mask(it, xx, yy)] = 2 * sim - 1
    return m


def template_axis(targCol):
    rgb = np.array(COLORS.get(targCol, COLORS["green"]))
    rg = rgb[0] - rgb[1]
    by = rgb[2] - (rgb[0] + rgb[1]) / 2
    n = np.hypot(rg, by)
    return (rg / n, by / n) if n > 1e-6 else (1.0, 0.0)


_GREY_C = None


def color_distinctiveness_constant():
    """Canonical red-vs-green singleton salience full-scale."""
    pos, _ = item_positions(6)
    items = [dict(x=pos[k][0], y=pos[k][1],
                  color=("red" if k + 1 == 5 else "green"),
                  shape=shape_for(k + 1, 2)) for k in range(6)]
    vals = item_color_distinctiveness(items)
    return max(float(vals.max()), 1e-9)


def item_color_distinctiveness(items):
    """Bottom-up item salience from color distinctiveness.

    The background is transparent: it is not a competing color. Each
    item is compared with the mean item color in opponent coordinates,
    so a singleton color is more salient than the majority color."""
    vecs = []
    for it in items:
        r, g, b = COLORS[it["color"]]
        vecs.append([r - g, b - (r + g) / 2])
    vecs = np.asarray(vecs, dtype=float)
    return np.linalg.norm(vecs - vecs.mean(0, keepdims=True), axis=1)


def item_footprint_mask(it, xx, yy, scale=1):
    """Rendered footprint for one item, matching front_end.render()."""
    to_px = lambda v: (v + 0.75) / 1.5 * IMG * scale
    r_px = ITEM_R / 1.5 * IMG * scale
    cx, cy = to_px(it["x"]), to_px(it["y"])
    shape = it.get("shape", "circle")
    if shape == "diamond":
        return (np.abs(xx - cx) + np.abs(yy - cy)) < 1.3 * r_px
    if shape == "square":
        return (np.abs(xx - cx) < 0.95 * r_px) & (np.abs(yy - cy) < 0.95 * r_px)
    if shape == "triangle":
        return ((yy - cy > -0.9 * r_px)
                & (np.abs(xx - cx) < 0.95 * r_px
                   * (1 - (yy - cy + 0.9 * r_px) / (2.0 * r_px))))
    if shape == "cross":
        return (((np.abs(xx - cx) < 0.45 * r_px) & (np.abs(yy - cy) < 1.1 * r_px))
                | ((np.abs(yy - cy) < 0.45 * r_px) & (np.abs(xx - cx) < 1.1 * r_px)))
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r_px ** 2


def sensory_salience_map(items):
    """Paint item color distinctiveness as hard item footprints.

    Background remains 0. The map is not blurred: P(x) is an item-level
    salience signal, so the display keeps crisp object support instead
    of introducing smoothing rims around every shape.
    """
    m = np.zeros((IMG, IMG))
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    vals = item_color_distinctiveness(items) / color_distinctiveness_constant()
    for val, it in zip(vals, items):
        m[item_footprint_mask(it, xx, yy)] = val
    return m


def paint_history_footprints(items, vals):
    """Paint selection history using each item's actual shape footprint."""
    m = np.zeros((IMG, IMG))
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    for val, it in zip(vals, items):
        m[item_footprint_mask(it, xx, yy)] = val
    return m


def target_color_evidence_map(items, targCol):
    """Salience-weighted signed target-color evidence.

    Each item's unsigned color distinctiveness supplies the magnitude.
    Its opponent-color direction, projected onto the target-color axis,
    supplies the sign. Thus more-salient opposite-color items produce
    stronger negative evidence, while background remains zero.
    """
    m = np.zeros((IMG, IMG))
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    u = np.array(template_axis(targCol))
    bg = np.asarray(COLORS[targCol], dtype=float)
    sal = item_color_distinctiveness(items) / color_distinctiveness_constant()
    for s, it in zip(sal, items):
        rgb = np.asarray(COLORS[it["color"]], dtype=float)
        vec = np.array([rgb[0] - rgb[1], rgb[2] - (rgb[0] + rgb[1]) / 2])
        base = np.array([bg[0] - bg[1], bg[2] - (bg[0] + bg[1]) / 2])
        diff = vec - base
        n = np.hypot(diff[0], diff[1])
        direction = float(np.dot(diff / n, u)) if n > 1e-9 else 1.0
        m[item_footprint_mask(it, xx, yy)] = s * direction
    return m


def template_match_map(items, targCol, template_shape="circle"):
    """Unified signed match to the search template, e.g. green circle.

    Color contributes signed salience-weighted target-color evidence.
    Shape contributes signed target-shape evidence: exact template
    shape is positive and poor shape matches are negative. Both
    components are painted as hard item footprints. The average gives one
    interpretable template map in [-1, 1].
    """
    color = target_color_evidence_map(items, targCol)
    signed_shape = signed_shape_match_map(items, template_shape)
    return 0.5 * (color + signed_shape)


def grey_color_constant():
    """Compatibility shim: color evidence now shares P's full-scale unit."""
    return color_distinctiveness_constant()


def item_centers(setsize):
    """The sensed pixels: each item's center, as (row, col)."""
    pos, _ = item_positions(setsize)
    return [(int(round((p[1] + 0.75) / 1.5 * IMG)),
             int(round((p[0] + 0.75) / 1.5 * IMG))) for p in pos]


def display_item_evidence(setsize, targLoc, singLoc, targCol, singCol):
    """One display -> the model's item evidence A [6, 3].

    Columns are P, T, S_T:
    - P is goal-independent bottom-up color salience, with the
      background treated as absent.
    - T is unified search-template evidence in [-1, 1], combining
      signed color and signed shape match.
    - S_T is retained as a diagnostic shape-similarity map in [0, 1]."""
    pos, _ = item_positions(setsize)
    items = [dict(x=pos[k][0], y=pos[k][1],
                  color=(singCol if (k + 1) == singLoc and singCol != "none"
                         else targCol),
                  shape=shape_for(k + 1, targLoc)) for k in range(setsize)]
    img = render(items)
    pmap = sensory_salience_map(items)                       # background = 0
    tmap = template_match_map(items, targCol)                 # background = 0
    sm = shape_match_map(img)                                # background = 0
    chans = np.stack([pmap, tmap, sm])
    A = np.zeros((6, 3), dtype=np.float32)
    for j, (py, px) in enumerate(item_centers(setsize)):
        A[j] = chans[:, py, px]
    return A


def main():
    sacc = pd.read_csv("dataset/saccades.csv", low_memory=False)
    sacc = normalize_colors(sacc)
    keys = (sacc[["setsize", "targLoc", "singLoc", "targCol", "singCol",
                  "fixloc"]].drop_duplicates().reset_index(drop=True))
    keys["ctx"] = np.arange(len(keys))
    n = len(keys)
    print(f"{n} unique contexts")
    A = np.zeros((n, 6, 3), dtype=np.float32)
    cache = {}
    for _, k in keys.iterrows():
        dk = (int(k.setsize), int(k.targLoc), int(k.singLoc),
              k.targCol, k.singCol)
        if dk not in cache:
            cache[dk] = display_item_evidence(*dk)
        A[int(k.ctx)] = cache[dk]
    np.savez_compressed("dataset/senses.npz", A=A,
                        GREY=np.array([grey_color_constant()]))
    merged = sacc.merge(keys, on=["setsize", "targLoc", "singLoc",
                                  "targCol", "singCol", "fixloc"], how="left")
    assert merged.ctx.notna().all()
    merged.to_csv("dataset/saccades_ctx.csv", index=False)
    print(f"saved senses.npz + saccades_ctx.csv "
          f"({n} contexts, {len(cache)} unique displays)")


if __name__ == "__main__":
    main()
