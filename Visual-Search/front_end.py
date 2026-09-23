"""Display rendering and item geometry for the visual-search tutorial."""

import numpy as np

# display geometry (normalized units: ring radius 0.5 as in the fits)
IMG = 256               # rendered resolution (square)
ITEM_R = 0.055          # item radius in ring-diameter units
ECC = 0.5               # ring radius

COLORS = {
    "green": (0.10, 0.75, 0.20),
    "red": (0.85, 0.10, 0.10),
    "pink": (0.95, 0.45, 0.70),
    "teal": (0.05, 0.65, 0.65),
    "blue": (0.15, 0.30, 0.90),
    "gray": (0.55, 0.55, 0.55),
}
BG = (0.35, 0.35, 0.35)


def render(items, scale=1):
    """items: list of dicts {x, y, color, shape} in centered coords
    (units of ring diameter). Returns [IMG*scale, IMG*scale, 3];
    scale > 1 renders at higher resolution (e.g. for shape matching)."""
    N = IMG * scale
    img = np.ones((N, N, 3)) * np.array(BG)
    yy, xx = np.mgrid[0:N, 0:N]
    to_px = lambda v: (v + 0.75) / 1.5 * N
    r_px = ITEM_R / 1.5 * N
    for it in items:
        cx, cy = to_px(it["x"]), to_px(it["y"])
        shape = it.get("shape", "circle")
        if shape == "diamond":
            mask = (np.abs(xx - cx) + np.abs(yy - cy)) < 1.3 * r_px
        elif shape == "square":
            mask = (np.abs(xx - cx) < 0.95 * r_px) & (np.abs(yy - cy) < 0.95 * r_px)
        elif shape == "triangle":
            mask = ((yy - cy > -0.9 * r_px)
                    & (np.abs(xx - cx) < 0.95 * r_px * (1 - (yy - cy + 0.9 * r_px)
                                                        / (2.0 * r_px))))
        elif shape == "cross":
            mask = (((np.abs(xx - cx) < 0.45 * r_px) & (np.abs(yy - cy) < 1.1 * r_px))
                    | ((np.abs(yy - cy) < 0.45 * r_px) & (np.abs(xx - cx) < 1.1 * r_px)))
        else:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r_px ** 2
        img[mask] = COLORS[it["color"]]
    return img


NONTARGET_SHAPES = ["square", "triangle", "cross", "diamond"]


def item_positions(setsize):
    """The setsize item slots: an iso-eccentric ring (units of ring
    diameter, so center-to-item distance is 0.5)."""
    ang = [2 * np.pi * j / setsize - np.pi / 2 for j in range(setsize)]
    return [(ECC * np.cos(a), ECC * np.sin(a)) for a in ang], ang


def shape_for(slot, targ_slot, template_shape="circle"):
    """Feature-search displays: the target is the template shape among
    HETEROGENEOUS nontarget shapes (the target is never a shape
    singleton - inclusion criterion of the source studies)."""
    if slot == targ_slot:
        return template_shape
    return NONTARGET_SHAPES[slot % 4]
