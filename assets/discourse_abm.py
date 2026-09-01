"""
Agent-based model of competing discourses diffusing over a social network.

Complex contagion (Centola & Macy): an agent adopts a frame only when the
share of its neighbours holding that frame exceeds an individual threshold.
Two frames are seeded in different communities of a stochastic block model,
so the animation shows within-community consolidation followed by contested
boundaries between blocks.
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(20260101)

# ---------------------------------------------------------------- network ----
SIZES = [55, 55, 55, 55]
P_IN, P_OUT = 0.080, 0.005
probs = [[P_IN if i == j else P_OUT for j in range(4)] for i in range(4)]
G = nx.stochastic_block_model(SIZES, probs, seed=42)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
G = nx.convert_node_labels_to_integers(G)
N = G.number_of_nodes()

pos = nx.spring_layout(G, k=0.32, iterations=180, seed=7)
neighbors = [list(G.neighbors(n)) for n in G.nodes()]

# ----------------------------------------------------------------- agents ----
NEUTRAL, FRAME_A, FRAME_B = 0, 1, 2
state = np.zeros(N, dtype=int)
threshold = rng.beta(1.8, 7.5, N)          # heterogeneous adoption thresholds
stubborn = rng.random(N) < 0.05            # a few committed agents never switch

deg = np.array([len(nb) for nb in neighbors])
block = np.repeat(np.arange(4), SIZES)[: N]

seeds_a = rng.choice(np.where(block == 0)[0], 6, replace=False)
seeds_b = rng.choice(np.where(block == 2)[0], 6, replace=False)
state[seeds_a] = FRAME_A
state[seeds_b] = FRAME_B
stubborn[seeds_a] = stubborn[seeds_b] = True

FRAMES, ACTIVE_SHARE = 72, 0.10
history, shares = [state.copy()], []


def step(s):
    new = s.copy()
    order = rng.choice(N, size=int(ACTIVE_SHARE * N), replace=False)
    for i in order:
        if stubborn[i] or not neighbors[i]:
            continue
        nb = s[neighbors[i]]
        k = len(nb)
        pa, pb = np.count_nonzero(nb == FRAME_A) / k, np.count_nonzero(nb == FRAME_B) / k
        t = threshold[i]
        if pa > t or pb > t:
            if abs(pa - pb) < 1e-9:
                continue                                  # cross-pressured: no move
            cand = FRAME_A if pa > pb else FRAME_B
            if rng.random() < 0.88:                       # noisy adoption
                new[i] = cand
        elif s[i] != NEUTRAL and rng.random() < 0.012:    # occasional disengagement
            new[i] = NEUTRAL
    return new


for _ in range(FRAMES - 1):
    state = step(state)
    history.append(state.copy())

for s in history:
    shares.append((np.mean(s == FRAME_A), np.mean(s == FRAME_B), np.mean(s == NEUTRAL)))
shares = np.array(shares)

# ------------------------------------------------------------------- plot ----
BG = "#0d1117"
C_NEUTRAL, C_A, C_B = "#3b4453", "#f778ba", "#58a6ff"
EDGE, TXT, MUTED = "#1f2733", "#e6edf3", "#8b949e"
palette = np.array([C_NEUTRAL, C_A, C_B])

fig = plt.figure(figsize=(8.6, 5.1), dpi=100, facecolor=BG)
gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.06,
                      left=0.045, right=0.965, top=0.90, bottom=0.10)
ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG); ax.axis("off")
axl = fig.add_subplot(gs[1]); axl.set_facecolor(BG)

xy = np.array([pos[n] for n in G.nodes()])
segs = np.array([[pos[u], pos[v]] for u, v in G.edges()])
from matplotlib.collections import LineCollection
ax.add_collection(LineCollection(segs, colors=EDGE, linewidths=0.55, zorder=1))
ax.set_xlim(xy[:, 0].min() - 0.09, xy[:, 0].max() + 0.09)
ax.set_ylim(xy[:, 1].min() - 0.09, xy[:, 1].max() + 0.09)

sizes = 16 + 3.4 * deg
scat = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=palette[history[0]],
                  edgecolors=BG, linewidths=0.7, zorder=3)
halo = ax.scatter(xy[:, 0], xy[:, 1], s=sizes * 2.6, c=palette[history[0]],
                  alpha=0.0, edgecolors="none", zorder=2)

fig.text(0.045, 0.955, "Competing discourses on a social network",
         color=TXT, fontsize=13.5, fontweight="bold", family="DejaVu Sans")
fig.text(0.045, 0.917, "agent-based complex contagion  ·  heterogeneous thresholds  ·  4 communities",
         color=MUTED, fontsize=8.6, family="DejaVu Sans")

for spine in ("top", "right", "left"):
    axl.spines[spine].set_visible(False)
axl.spines["bottom"].set_color(EDGE)
axl.tick_params(colors=MUTED, labelsize=7.5, length=2)
axl.set_xlim(0, FRAMES - 1); axl.set_ylim(0, 1)
axl.set_yticks([0, 0.5, 1]); axl.set_yticklabels(["0", ".5", "1"])
axl.set_xlabel("time step", color=MUTED, fontsize=8)
axl.grid(axis="y", color=EDGE, linewidth=0.6)
axl.set_axisbelow(True)

(la,) = axl.plot([], [], color=C_A, lw=2.0)
(lb,) = axl.plot([], [], color=C_B, lw=2.0)
(ln,) = axl.plot([], [], color=C_NEUTRAL, lw=1.6, ls=(0, (3, 2)))
axl.legend([la, lb, ln], ["frame A", "frame B", "unadopted"],
           loc="upper right", ncol=3, frameon=False, fontsize=8,
           labelcolor=MUTED, handlelength=1.5, bbox_to_anchor=(1.0, 1.42))


def update(f):
    s = history[f]
    cols = palette[s]
    scat.set_color(cols)
    changed = (s != history[max(f - 1, 0)]).astype(float)
    halo.set_color(cols)
    halo.set_alpha(0.30 * changed if f else np.zeros(N))
    t = np.arange(f + 1)
    la.set_data(t, shares[: f + 1, 0])
    lb.set_data(t, shares[: f + 1, 1])
    ln.set_data(t, shares[: f + 1, 2])
    return scat, halo, la, lb, ln


anim = FuncAnimation(fig, update, frames=FRAMES, interval=110, blit=False)
anim.save("/home/claude/discourse_diffusion.gif",
          writer=PillowWriter(fps=9), savefig_kwargs={"facecolor": BG})
print("nodes", N, "edges", G.number_of_edges())
print("final shares  A=%.2f  B=%.2f  neutral=%.2f" % tuple(shares[-1]))
