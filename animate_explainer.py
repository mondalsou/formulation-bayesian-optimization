#!/usr/bin/env python3
"""
Bayesian Optimization Explainer
60-second animated explainer, 1920×1080, 30 fps
Audience: pharma/materials chemists with no ML background
"""

import os, math, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

# ─── Configuration ────────────────────────────────────────────────────────────
FPS       = 30
DURATION  = 60
TOTAL_F   = FPS * DURATION          # 1800 frames
W, H      = 19.2, 10.8             # figure size in inches (×100 dpi = 1920×1080)
OUT_DIR   = "/sessions/admiring-compassionate-gauss/mnt/outputs/frames"
GIF_DIR   = "/sessions/admiring-compassionate-gauss/mnt/outputs/gif_frames"
MP4_OUT   = "/sessions/admiring-compassionate-gauss/mnt/outputs/bo_explainer.mp4"
GIF_OUT   = "/sessions/admiring-compassionate-gauss/mnt/outputs/bo_teaser.gif"

# ─── Palette ──────────────────────────────────────────────────────────────────
BG        = '#EEF7F9'
TEAL_D    = '#1A7A8A'
TEAL_M    = '#2EB8C8'
TEAL_L    = '#A8DDE5'
TEAL_VL   = '#D4EFF4'
ACCENT    = '#F5A623'
ACC_L     = '#FDDFA0'
TEXT      = '#112A33'
TEXT_M    = '#2E5A65'
WHITE     = '#FFFFFF'
GRAY      = '#CDE3E8'

# ─── Easing ───────────────────────────────────────────────────────────────────
def ease_in_out(t): return t*t*(3-2*t)
def ease_out(t):    return 1-(1-t)**3
def ease_in(t):     return t**3
def clip01(t):      return max(0.0, min(1.0, t))

def fade(t, start, end):
    """Fade in from t=start, fully in at t=end"""
    return clip01((t-start)/(end-start+1e-9))

# ─── Drawing helpers ──────────────────────────────────────────────────────────
def text_shadow(ax, x, y, s, **kw):
    txt = ax.text(x, y, s, **kw)
    txt.set_path_effects([pe.withStroke(linewidth=4, foreground=BG)])
    return txt

def draw_rounded_box(ax, x, y, w, h, color, alpha=1.0, radius=0.3, lw=0, ec='none'):
    fancy = FancyBboxPatch((x-w/2, y-h/2), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha,
                           transform=ax.transData, zorder=3)
    ax.add_patch(fancy)
    return fancy

def draw_beaker(ax, cx, cy, scale=1.0, color=TEAL_M, alpha=1.0):
    """Simple beaker icon"""
    # body trapezoid
    bx = [cx-0.4*scale, cx+0.4*scale, cx+0.3*scale, cx-0.3*scale]
    by = [cy-0.55*scale, cy-0.55*scale, cy+0.35*scale, cy+0.35*scale]
    ax.fill(bx, by, color=color, alpha=alpha, zorder=4)
    # neck / opening
    ax.fill([cx-0.35*scale, cx+0.35*scale, cx+0.38*scale, cx-0.38*scale],
            [cy+0.35*scale, cy+0.35*scale, cy+0.55*scale, cy+0.55*scale],
            color=color, alpha=alpha, zorder=4)
    # liquid
    ax.fill([cx-0.38*scale, cx+0.38*scale, cx+0.28*scale, cx-0.28*scale],
            [cy-0.52*scale, cy-0.52*scale, cy-0.05*scale, cy-0.05*scale],
            color=TEAL_VL, alpha=alpha*0.9, zorder=5)

def draw_pill(ax, cx, cy, w=1.2, h=0.5, color=ACCENT, alpha=1.0):
    """Tablet/pill icon"""
    r = h/2
    ax.add_patch(mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=color, edgecolor=WHITE, linewidth=2, alpha=alpha, zorder=4))

def draw_target(ax, cx, cy, scale=1.0, alpha=1.0):
    for r, c in [(0.6, TEAL_L), (0.38, TEAL_M), (0.2, TEAL_D)]:
        ax.add_patch(Circle((cx, cy), r*scale, facecolor=c, edgecolor='none',
                             alpha=alpha, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.07*scale, facecolor=ACCENT,
                         edgecolor='none', alpha=alpha, zorder=5))

def draw_checklist(ax, cx, cy, scale=1.0, n_checked=3, n_total=4, alpha=1.0):
    bw, bh = 1.4*scale, 1.6*scale
    draw_rounded_box(ax, cx, cy, bw, bh, WHITE, alpha=alpha, radius=0.15)
    for i in range(n_total):
        y = cy + bh/2 - 0.25*scale - i*0.35*scale
        # checkbox
        col = TEAL_D if i < n_checked else GRAY
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-0.55*scale, y-0.11*scale), 0.22*scale, 0.22*scale,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=col, edgecolor='none', alpha=alpha, zorder=5))
        # line
        ax.plot([cx-0.25*scale, cx+0.55*scale], [y, y],
                color=TEAL_L if i < n_checked else GRAY,
                lw=2.5*scale, alpha=alpha, zorder=5, solid_capstyle='round')

def draw_notebook(ax, cx, cy, scale=1.0, alpha=1.0):
    bw, bh = 1.3*scale, 1.6*scale
    draw_rounded_box(ax, cx, cy, bw, bh, WHITE, alpha=alpha, radius=0.12)
    # spine
    ax.add_patch(mpatches.Rectangle((cx-bw/2, cy-bh/2), 0.22*scale, bh,
        facecolor=TEAL_M, alpha=alpha, zorder=5))
    # lines on page
    for i in range(5):
        y = cy + 0.45*scale - i*0.22*scale
        ax.plot([cx-0.15*scale, cx+0.55*scale], [y, y],
                color=TEAL_L, lw=2*scale, alpha=alpha*0.7, zorder=5)

def draw_loop_arrow(ax, cx, cy, r=1.5, start_angle=30, span=300,
                    color=TEAL_M, lw=3, alpha=1.0, arrowsize=0.25):
    """Circular arrow"""
    theta = np.linspace(np.radians(start_angle),
                        np.radians(start_angle+span), 120)
    xs = cx + r*np.cos(theta)
    ys = cy + r*np.sin(theta)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=5, solid_capstyle='round')
    # arrowhead at end
    end_angle = start_angle + span
    dx =  np.sin(np.radians(end_angle)) * arrowsize
    dy = -np.cos(np.radians(end_angle)) * arrowsize
    ex = cx + r*np.cos(np.radians(end_angle))
    ey = cy + r*np.sin(np.radians(end_angle))
    ax.annotate('', xy=(ex+dx, ey+dy), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=20), zorder=5,
                alpha=alpha)

def label_pill(ax, x, y, text, color=TEAL_D, alpha=1.0, fontsize=22):
    draw_rounded_box(ax, x, y, len(text)*0.27+0.5, 0.45, color, alpha=alpha, radius=0.2)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=WHITE, fontweight='bold', alpha=alpha, zorder=6)

# ─── Background + title card ──────────────────────────────────────────────────
def draw_bg(ax):
    ax.set_facecolor(BG)
    # subtle top banner
    ax.add_patch(mpatches.Rectangle((0, 10.0), 19.2, 0.8,
        facecolor=TEAL_D, alpha=0.15, zorder=0))

def scene_title(ax, step_num, title, alpha=1.0):
    if step_num:
        ax.text(0.6, 10.25, f"Step {step_num}", ha='left', va='center',
                fontsize=20, color=TEAL_D, fontweight='bold', alpha=alpha, zorder=10)
    ax.text(0.6 if step_num else 9.6, 10.25, title,
            ha='left' if step_num else 'center', va='center',
            fontsize=22 if step_num else 28, color=TEXT, alpha=alpha,
            fontweight='bold' if not step_num else 'normal', zorder=10)

# ─── Caption bar ─────────────────────────────────────────────────────────────
def caption_bar(ax, text, alpha=1.0):
    ax.add_patch(mpatches.Rectangle((0, 0), 19.2, 0.7,
        facecolor=TEAL_D, alpha=0.9*alpha, zorder=20))
    ax.text(9.6, 0.35, text, ha='center', va='center',
            fontsize=22, color=WHITE, alpha=alpha, zorder=21)

# ─── Scene 1: Problem setup ───────────────────────────────────────────────────
def scene1(ax, t):
    """0–6 s: Lab bench + many formulations"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.3))
    a2 = ease_out(fade(t, 0.25, 0.7))
    a3 = ease_out(fade(t, 0.5, 1.0))

    # many small beakers scattered = "too many recipes"
    np.random.seed(42)
    positions = [(3.5+np.random.uniform(-2.8,2.8), 5.5+np.random.uniform(-2.2,2.2))
                 for _ in range(18)]
    for i,(bx,by) in enumerate(positions):
        ba = a1 * ease_out(clip01((t - i*0.018)))
        draw_beaker(ax, bx, by, scale=0.55, alpha=ba*0.75)

    # main big beaker
    draw_beaker(ax, 9.6, 5.4, scale=1.3, alpha=a1)

    # dots representing formulations
    xs = np.linspace(10.5, 18.0, 14)
    ys = [5 + 1.5*math.sin(i*0.9) for i in range(14)]
    for i,(px,py) in enumerate(zip(xs, ys)):
        pa = a2 * ease_out(clip01((t - 0.3 - i*0.025)))
        ax.scatter(px, py, s=160, color=TEAL_M, alpha=pa*0.8,
                   edgecolors=TEAL_D, linewidths=1.5, zorder=5)

    # Big text
    big_kw = dict(ha='center', va='center', fontsize=52, color=TEXT,
                  fontweight='bold', zorder=10)
    ax.text(9.6, 8.5, "Too many possible recipes.", alpha=a2, **big_kw)
    ax.text(9.6, 7.4, "Too few experiments.", alpha=a3, **big_kw)

    # subtext
    ax.text(9.6, 1.5, "Testing every combination is too slow and expensive.",
            ha='center', va='center', fontsize=30, color=TEXT_M, alpha=a3, zorder=10)
    caption_bar(ax, "The challenge: exploring a huge space with limited resources", a3)

# ─── Scene 2: Constraints ─────────────────────────────────────────────────────
def scene2(ax, t):
    """6–13 s: Define feasible space"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.25))
    a2 = ease_out(fade(t, 0.2, 0.6))
    a3 = ease_out(fade(t, 0.55, 1.0))

    # axis box (the 2D space)
    bx0, bx1 = 1.5, 11.5
    by0, by1 = 1.5,  9.0
    bw = bx1-bx0; bh = by1-by0
    # full experimental space (light)
    ax.add_patch(mpatches.Rectangle((bx0, by0), bw, bh,
        facecolor=TEAL_VL, edgecolor=GRAY, linewidth=2, alpha=a1*0.6, zorder=2))
    # feasible zone (green-teal box)
    fx0, fy0 = bx0+1.2, by0+1.5
    fw, fh  = bw-2.5, bh-3.0
    ax.add_patch(mpatches.Rectangle((fx0, fy0), fw, fh,
        facecolor=TEAL_L, edgecolor=TEAL_D, linewidth=3,
        linestyle='--', alpha=a2*0.55, zorder=3))

    # axis labels
    ax.text((bx0+bx1)/2, by0-0.35, "Ingredient A (%)", ha='center', va='top',
            fontsize=26, color=TEXT_M, alpha=a1)
    ax.text(bx0-0.35, (by0+by1)/2, "Ingredient B (%)", ha='center', va='center',
            fontsize=26, color=TEXT_M, alpha=a1, rotation=90)
    ax.plot([bx0,bx1],[by0,by0], color=TEXT_M, lw=2, alpha=a1)
    ax.plot([bx0,bx0],[by0,by1], color=TEXT_M, lw=2, alpha=a1)

    # "Feasible range" label
    label_pill(ax, fx0+fw/2, fy0+fh+0.4, "Feasible range", alpha=a2)

    # sliders on the right
    sx = 12.8
    slider_data = [
        ("Surfactant",  0.3, 1.0),
        ("API load",    0.5, 0.8),
        ("pH",          0.2, 0.7),
        ("Viscosity",   0.4, 0.9),
    ]
    for i, (name, lo, hi) in enumerate(slider_data):
        sy = 8.0 - i*1.4
        sa = a2 * ease_out(clip01((t - 0.2 - i*0.08)))
        # track
        ax.plot([sx, sx+5.5], [sy, sy], color=GRAY, lw=5, alpha=sa,
                solid_capstyle='round', zorder=4)
        # active range
        ax.plot([sx+lo*5.5, sx+hi*5.5], [sy, sy], color=TEAL_M, lw=5,
                alpha=sa, solid_capstyle='round', zorder=5)
        # knobs
        for pos in [lo, hi]:
            ax.scatter(sx+pos*5.5, sy, s=220, color=TEAL_D, zorder=6, alpha=sa)
        ax.text(sx-0.15, sy, name, ha='right', va='center',
                fontsize=22, color=TEXT, alpha=sa)

    # headline
    ax.text(9.6, 9.6, "Step 1 · Define feasible space (safe, practical ranges)",
            ha='center', va='center', fontsize=36, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)
    ax.text(9.6, 0.35, "Set realistic ingredient ranges and constraints",
            ha='center', va='center', fontsize=26, color=TEXT_M, alpha=a3, zorder=10)
    caption_bar(ax, "Step 1 — Feasible range: what's safe and practical to test", a3)

# ─── Scene 3: Initial experiments ────────────────────────────────────────────
def scene3(ax, t):
    """13–20 s: Starter experiments"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.3))
    a3 = ease_out(fade(t, 0.6, 1.0))

    bx0, bx1 = 2.5, 12.0
    by0, by1 = 1.5,  9.2
    ax.add_patch(mpatches.Rectangle((bx0, by0), bx1-bx0, by1-by0,
        facecolor=TEAL_VL, edgecolor=TEAL_L, linewidth=2, alpha=a1*0.5))
    ax.text((bx0+bx1)/2, by0-0.35, "Formulation space", ha='center',
            fontsize=24, color=TEXT_M, alpha=a1)

    # diverse starter points (Latin hypercube-like)
    pts = [(3.5,2.5),(6.5,8.2),(10.5,2.8),(4.5,6.5),(9.0,6.8),
           (7.5,4.0),(11.0,8.0),(3.2,8.5),(8.5,3.2)]
    for i,(px,py) in enumerate(pts):
        pa = a1 * ease_out(clip01((t - i*0.055)))
        # beaker icon at each point
        draw_beaker(ax, px, py, scale=0.55, alpha=pa)
        ax.scatter(px, py+0.55*0.55, s=60, color=TEAL_D, alpha=pa, zorder=6)

    label_pill(ax, 7.25, 9.7, "Starter experiments", alpha=a1)

    # right panel: lab notebook + checklist
    draw_notebook(ax, 15.5, 5.5, scale=1.4, alpha=a1)
    draw_checklist(ax, 17.5, 5.5, scale=1.2, n_checked=3, n_total=5, alpha=a1)

    ax.text(9.6, 10.1,
            "Step 2 · Run a small, diverse starter set of experiments",
            ha='center', va='center', fontsize=34, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)
    ax.text(9.6, 0.35,
            "Cover the space broadly — don't cluster all runs in one region",
            ha='center', va='center', fontsize=26, color=TEXT_M, alpha=a3)
    caption_bar(ax, "Step 2 — Starter experiments: diverse, spread across the space", a3)

# ─── Scene 4: Build model ─────────────────────────────────────────────────────
def _build_surfaces():
    """Build prediction + uncertainty grids once"""
    np.random.seed(7)
    x = np.linspace(0,1,80)
    y = np.linspace(0,1,80)
    X,Y = np.meshgrid(x,y)
    # synthetic "true" surface
    Z = (np.sin(X*5)*np.cos(Y*4)*0.5 +
         np.exp(-((X-0.65)**2+(Y-0.7)**2)/0.05)*1.2 +
         np.exp(-((X-0.3)**2+(Y-0.3)**2)/0.08)*0.7)
    Z = gaussian_filter(Z, 3)
    Z = (Z-Z.min())/(Z.max()-Z.min())

    # uncertainty: high away from observed points
    obs = np.array([(0.1,0.1),(0.5,0.9),(0.9,0.1),(0.2,0.6),(0.8,0.7)])
    U = np.ones_like(Z)
    for (ox,oy) in obs:
        U -= 0.55*np.exp(-((X-ox)**2+(Y-oy)**2)/0.04)
    U = np.clip(U, 0, 1)
    return Z, U, obs

_Z, _U, _OBS = _build_surfaces()

def scene4(ax, t):
    """20–30 s: Prediction + uncertainty"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.3))
    a2 = ease_out(fade(t, 0.25, 0.7))
    a3 = ease_out(fade(t, 0.6, 1.0))

    bx0, bx1 = 0.8, 10.0
    by0, by1 = 1.2,  9.3
    ex = bx1-bx0; ey = by1-by0

    # prediction heatmap
    cmap_pred = LinearSegmentedColormap.from_list(
        'pred', [TEAL_VL, TEAL_L, TEAL_M, TEAL_D, '#0A3A45'])
    ax.imshow(_Z, origin='lower', extent=[bx0,bx1,by0,by1],
              cmap=cmap_pred, aspect='auto', alpha=a1*0.9, zorder=2)

    # uncertainty overlay – transparent → opaque amber
    import matplotlib.colors as mcolors
    _ar,_ag,_ab,_ = mcolors.to_rgba(ACCENT)
    cmap_unc = LinearSegmentedColormap.from_list(
        'unc', [(_ar,_ag,_ab,0.0), (_ar,_ag,_ab,0.65)])
    ax.imshow(_U, origin='lower', extent=[bx0,bx1,by0,by1],
              cmap=cmap_unc, aspect='auto', alpha=a2, zorder=3)

    # observed points
    for (ox,oy) in _OBS:
        px = bx0 + ox*ex; py = by0 + oy*ey
        pa = a1 * ease_out(clip01((t - 0.1)))
        ax.scatter(px, py, s=220, color=WHITE, edgecolors=TEAL_D,
                   linewidths=2.5, zorder=7, alpha=pa)

    # axis labels
    ax.text((bx0+bx1)/2, by0-0.35, "Ingredient A", ha='center',
            fontsize=22, color=TEXT_M, alpha=a1)
    ax.text(bx0-0.4, (by0+by1)/2, "Ingredient B", ha='center',
            fontsize=22, color=TEXT_M, alpha=a1, rotation=90)

    # legend pills
    label_pill(ax, 5.4, 9.8, "What we expect", color=TEAL_D, alpha=a1)
    label_pill(ax, 5.4, 9.15, "Where we are unsure", color='#C47A00', alpha=a2)

    # right panel: explanation cards
    cards = [
        (13.5, 7.4, TEAL_D,  "Prediction surface",
         "Where performance is likely\nhigh vs. low"),
        (13.5, 4.3, '#B06000', "Uncertainty map",
         "Where the model needs\nmore information"),
    ]
    for cx,cy,cc,title,body in cards:
        ca = a1 if cc == TEAL_D else a2
        draw_rounded_box(ax, cx, cy, 5.8, 2.5, WHITE, alpha=ca*0.95, radius=0.3)
        draw_rounded_box(ax, cx, cy+0.75, 5.8, 0.65, cc, alpha=ca, radius=0.2)
        ax.text(cx, cy+0.75, title, ha='center', va='center',
                fontsize=24, color=WHITE, fontweight='bold', alpha=ca)
        ax.text(cx, cy-0.25, body, ha='center', va='center',
                fontsize=22, color=TEXT, alpha=ca, linespacing=1.4)

    ax.text(9.6, 10.3,
            "Step 3 · Learn from data  (prediction + uncertainty)",
            ha='center', va='center', fontsize=34, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)
    caption_bar(ax, "Step 3 — Model learns: what to expect AND where it's uncertain", a3)

# ─── Scene 5: Next experiment ─────────────────────────────────────────────────
def scene5(ax, t):
    """30–40 s: Suggest next best point"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0,   0.3))
    a2 = ease_out(fade(t, 0.3, 0.7))
    a3 = ease_out(fade(t, 0.65, 1.0))

    bx0, bx1 = 0.8, 10.0
    by0, by1 = 1.2,  9.3
    ex = bx1-bx0; ey = by1-by0

    cmap_pred = LinearSegmentedColormap.from_list(
        'pred', [TEAL_VL, TEAL_L, TEAL_M, TEAL_D, '#0A3A45'])
    ax.imshow(_Z, origin='lower', extent=[bx0,bx1,by0,by1],
              cmap=cmap_pred, aspect='auto', alpha=0.85, zorder=2)
    import matplotlib.colors as mcolors
    _ar2,_ag2,_ab2,_ = mcolors.to_rgba(ACCENT)
    cmap_unc = LinearSegmentedColormap.from_list(
        'unc2', [(_ar2,_ag2,_ab2,0.0), (_ar2,_ag2,_ab2,0.55)])
    ax.imshow(_U, origin='lower', extent=[bx0,bx1,by0,by1],
              cmap=cmap_unc, aspect='auto', alpha=0.75, zorder=3)

    for (ox,oy) in _OBS:
        ax.scatter(bx0+ox*ex, by0+oy*ey, s=220, color=WHITE,
                   edgecolors=TEAL_D, linewidths=2.5, zorder=7)

    # NEXT BEST point — balances high predicted value & uncertainty
    nx, ny = 0.62, 0.68
    px = bx0 + nx*ex; py = by0 + ny*ey
    # pulsing glow
    pulse = 0.5 + 0.5*math.sin(t*math.pi*6)
    for r, al in [(0.55,0.12),(0.38,0.22),(0.22,0.35)]:
        ax.add_patch(Circle((px,py), r, facecolor=ACCENT,
                             alpha=a2*al*(0.7+0.3*pulse), zorder=8))
    ax.scatter(px, py, s=400, color=ACCENT, edgecolors=WHITE,
               linewidths=3, zorder=9, alpha=a2)
    # star-like cross hairs
    for dx,dy in [(0.7,0),(-0.7,0),(0,0.7),(0,-0.7)]:
        ax.plot([px,px+dx],[py,py+dy], color=ACCENT, lw=2.5, alpha=a2*0.7)

    # label
    label_pill(ax, px+1.6, py+0.5, "Next best test", color='#B06000', alpha=a2, fontsize=20)

    # right: "why this point?" card
    draw_rounded_box(ax, 15.0, 6.5, 7.8, 3.5, WHITE, alpha=a2*0.95, radius=0.3)
    ax.text(15.0, 8.0,
            "Why this point?",
            ha='center', va='center', fontsize=28, color=TEAL_D,
            fontweight='bold', alpha=a2)
    for i,(icon,desc) in enumerate([
        ("✓", "High predicted performance"),
        ("?", "High uncertainty → most to learn"),
        ("→", "Best learning value per run"),
    ]):
        iy = 7.0 - i*0.85
        ax.text(11.5, iy, icon, ha='center', va='center',
                fontsize=30, color=ACCENT, fontweight='bold', alpha=a2)
        ax.text(12.0, iy, desc, ha='left', va='center',
                fontsize=22, color=TEXT, alpha=a2)

    # convergence hint arrow
    draw_target(ax, 17.5, 3.5, scale=1.0, alpha=a3)
    ax.text(17.5, 2.25, "Target", ha='center', fontsize=22,
            color=TEXT_M, alpha=a3)

    ax.text(9.6, 10.3,
            "Step 4 · Suggest the next most informative experiment",
            ha='center', va='center', fontsize=34, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)
    caption_bar(ax,
        "Step 4 — Next best test: balances expected gain with information value", a3)

# ─── Scene 6: Loop ───────────────────────────────────────────────────────────
def scene6(ax, t):
    """40–50 s: Run → Update → Repeat"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.25))
    a2 = ease_out(fade(t, 0.2, 0.6))
    a3 = ease_out(fade(t, 0.55, 1.0))

    # Loop diagram: 4 nodes in a circle
    # icon replaced with short ASCII symbol (no emoji — font fallback)
    nodes = [
        (9.6, 8.2,  TEAL_D,   "Lab\nexperiment",  "[lab]"),
        (13.8,5.5,  TEAL_M,   "Measure\nresult",   "[data]"),
        (9.6,  2.8, TEAL_D,   "Update\nmodel",     "[model]"),
        (5.4,  5.5, ACCENT,   "Suggest\nnext",     "[ >> ]"),
    ]
    r_node = 1.05

    # draw arcs between nodes
    arc_pairs = [(0,1),(1,2),(2,3),(3,0)]
    for i,(ni,nj) in enumerate(arc_pairs):
        x0,y0 = nodes[ni][:2]; x1,y1 = nodes[nj][:2]
        aa = a1 * ease_out(clip01((t - i*0.08)))
        ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(
                        arrowstyle='->', color=TEAL_M, lw=3,
                        mutation_scale=22,
                        connectionstyle='arc3,rad=0.35'),
                    alpha=aa, zorder=4)

    # draw nodes
    for i,(nx,ny,nc,label,icon) in enumerate(nodes):
        na = a1 * ease_out(clip01((t - i*0.06)))
        ax.add_patch(Circle((nx,ny), r_node, facecolor=nc,
                             edgecolor=WHITE, linewidth=3, alpha=na, zorder=5))
        ax.text(nx, ny+0.22, icon, ha='center', va='center',
                fontsize=16, color=WHITE, alpha=na, zorder=6, style='italic')
        ax.text(nx, ny-0.45, label, ha='center', va='center',
                fontsize=20, color=WHITE, fontweight='bold',
                alpha=na, zorder=6, linespacing=1.3)

    # new data points appearing
    new_pts = [(3.5,8.8),(15.5,8.5),(16.5,5.5),(3.2,3.0),(15.8,3.2)]
    for i,(px,py) in enumerate(new_pts):
        pa = a2 * ease_out(clip01((t - 0.4 - i*0.06)))
        draw_beaker(ax, px, py, scale=0.55, color=ACCENT, alpha=pa*0.8)

    # label "Update and repeat"
    label_pill(ax, 9.6, 1.3, "Update and repeat", color=TEAL_D, alpha=a3, fontsize=24)

    ax.text(9.6, 10.3,
            "Step 5 · Run, update, repeat",
            ha='center', va='center', fontsize=36, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)
    ax.text(9.6, 9.55,
            "Each experiment makes the model smarter",
            ha='center', va='center', fontsize=26, color=TEXT_M, alpha=a2, zorder=10)
    caption_bar(ax, "Step 5 — Update and repeat: every run sharpens the next suggestion", a3)

# ─── Scene 7: Outcome ────────────────────────────────────────────────────────
def scene7(ax, t):
    """50–60 s: Convergence + final message"""
    draw_bg(ax)
    a1 = ease_out(fade(t, 0, 0.3))
    a2 = ease_out(fade(t, 0.25, 0.65))
    a3 = ease_out(fade(t, 0.6, 1.0))

    # Convergence curve
    cx0, cx1 = 1.2, 10.5
    cy0, cy1 = 1.5, 8.8
    n_pts = 14
    xs = np.linspace(cx0, cx1, n_pts)
    # performance improves then plateaus
    raw = 1 - np.exp(-np.linspace(0, 3.5, n_pts)) + 0.05*np.random.RandomState(3).randn(n_pts)
    raw = np.clip(raw, 0, 1)
    ys = cy0 + raw*(cy1-cy0)

    # grid
    for yg in np.linspace(cy0, cy1, 5):
        ax.plot([cx0,cx1],[yg,yg], color=TEAL_L, lw=1, alpha=a1*0.4)
    ax.plot([cx0,cx1],[cy0,cy0], color=TEXT_M, lw=2, alpha=a1)
    ax.plot([cx0,cx0],[cy0,cy1], color=TEXT_M, lw=2, alpha=a1)
    ax.text((cx0+cx1)/2, cy0-0.35, "Experiment number", ha='center',
            fontsize=22, color=TEXT_M, alpha=a1)
    ax.text(cx0-0.5, (cy0+cy1)/2, "Performance", ha='center', va='center',
            fontsize=22, color=TEXT_M, alpha=a1, rotation=90)

    # plot the curve progressively
    show_n = max(2, int(n_pts * ease_out(t)))
    if show_n >= 2:
        ax.plot(xs[:show_n], ys[:show_n], color=TEAL_D, lw=3.5,
                solid_capstyle='round', alpha=a1, zorder=5)
        ax.scatter(xs[:show_n], ys[:show_n], s=120, color=TEAL_M,
                   edgecolors=TEAL_D, linewidths=1.5, zorder=6, alpha=a1)
        # highlight best point
        best_i = int(np.argmax(ys[:show_n]))
        ax.scatter(xs[best_i], ys[best_i], s=350, color=ACCENT,
                   edgecolors=WHITE, linewidths=2.5, zorder=7, alpha=a1)

    # target line
    target_y = cy0 + 0.88*(cy1-cy0)
    ax.plot([cx0,cx1],[target_y,target_y], color=ACCENT,
            lw=2.5, linestyle='--', alpha=a2, zorder=4)
    ax.text(cx1+0.15, target_y, "Target", ha='left', va='center',
            fontsize=22, color=ACCENT, fontweight='bold', alpha=a2)

    # right: key messages
    msgs = [
        (TEAL_D,  "Faster optimization"),
        (TEAL_M,  "Fewer trials needed"),
        (ACCENT,  "Each run has a purpose"),
    ]
    for i,(mc,mt) in enumerate(msgs):
        my = 7.8 - i*1.5
        ma = a2 * ease_out(clip01((t - 0.3 - i*0.1)))
        draw_rounded_box(ax, 15.0, my, 7.0, 1.1, mc, alpha=ma*0.18, radius=0.3)
        draw_rounded_box(ax, 11.9, my, 0.9, 1.1, mc, alpha=ma*0.9, radius=0.2)
        ax.text(11.9, my, "✓", ha='center', va='center',
                fontsize=28, color=WHITE, alpha=ma)
        ax.text(15.45, my, mt, ha='center', va='center',
                fontsize=26, color=TEXT, fontweight='bold', alpha=ma)

    # main headline
    ax.text(9.6, 10.3,
            "Faster optimization with fewer trials",
            ha='center', va='center', fontsize=40, color=TEAL_D,
            fontweight='bold', alpha=a1, zorder=10)

    # sub-tagline
    ax.text(9.6, 9.55,
            "Human expertise  +  data-driven iteration",
            ha='center', va='center', fontsize=28, color=TEXT_M, alpha=a2, zorder=10)

    # final motto
    if a3 > 0.05:
        draw_rounded_box(ax, 9.6, 0.9, 14.0, 0.85, TEAL_D, alpha=a3, radius=0.25)
        ax.text(9.6, 0.9,
                "Not replacing lab scientists — accelerating their decisions.",
                ha='center', va='center', fontsize=28, color=WHITE,
                fontweight='bold', alpha=a3, zorder=15)

# ─── Scene dispatch ───────────────────────────────────────────────────────────
# Scene time boundaries (seconds)
SCENE_BOUNDS = [(0,6),(6,13),(13,20),(20,30),(30,40),(40,50),(50,60)]
SCENE_FNS    = [scene1, scene2, scene3, scene4, scene5, scene6, scene7]

def render_frame(frame_idx):
    t_global = frame_idx / FPS      # seconds from start
    # find active scene
    for i,((t0,t1),fn) in enumerate(zip(SCENE_BOUNDS, SCENE_FNS)):
        if t0 <= t_global < t1:
            t_local = (t_global - t0) / (t1 - t0)  # 0..1 within scene
            break
    else:
        i, fn, t_local = 6, scene7, 1.0

    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor(BG)
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor(BG)

    fn(ax, t_local)

    path = os.path.join(OUT_DIR, f"frame_{frame_idx:05d}.png")
    fig.savefig(path, dpi=100, bbox_inches=None, pad_inches=0,
                facecolor=BG)
    plt.close(fig)

# ─── GIF scene range (scenes 4→5→6: frames 600–1500) ────────────────────────
GIF_START_F = 600
GIF_END_F   = 1500
GIF_STEP    = 3   # every 3rd frame → ~100 frames for an 8-10 s GIF @ 10fps

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GIF_DIR,  exist_ok=True)

    print(f"Rendering {TOTAL_F} frames …")
    for f in range(TOTAL_F):
        render_frame(f)
        if f % 90 == 0:
            pct = 100*f//TOTAL_F
            print(f"  {pct}% — frame {f}/{TOTAL_F}")

    print("Compiling MP4 …")
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(OUT_DIR, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        MP4_OUT
    ], check=True)

    print("Extracting GIF frames …")
    gif_frames = []
    for f in range(GIF_START_F, GIF_END_F, GIF_STEP):
        gif_frames.append(os.path.join(OUT_DIR, f"frame_{f:05d}.png"))

    # Use ffmpeg palette for high-quality GIF
    palette = os.path.join(GIF_DIR, 'palette.png')
    frame_pattern = os.path.join(OUT_DIR, 'frame_%05d.png')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-start_number', str(GIF_START_F),
        '-i', frame_pattern,
        '-frames:v', str(GIF_END_F - GIF_START_F),
        '-vf', 'fps=10,scale=960:540:flags=lanczos,palettegen',
        palette
    ], check=True)

    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-start_number', str(GIF_START_F),
        '-i', frame_pattern,
        '-i', palette,
        '-frames:v', str(GIF_END_F - GIF_START_F),
        '-lavfi', 'fps=10,scale=960:540:flags=lanczos[x];[x][1:v]paletteuse',
        '-loop', '0',
        GIF_OUT
    ], check=True)

    print("Done!")
    print(f"  MP4 → {MP4_OUT}")
    print(f"  GIF → {GIF_OUT}")
