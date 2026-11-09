"""Paper figures (static, light mode). Means with 95% CIs (mean +/- 1.96 SE over paired seeds)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6")
SYS = [("R0", "Peer rating"), ("R1", "Audit-only"), ("R2", "CTI"), ("R2_nobond", "CTI − bond"),
       ("R2_nodecl", "CTI − declaration")]
SYS_C = dict(R0=C["aqua"], R1=C["orange"], R2=C["blue"], R2_nobond=C["violet"], R2_nodecl=C["yellow"])
ATT = dict(A1="A1 selective defection", A2="A2 identity reset", A5="A5 bait-and-switch",
           A1R="A1-R payoff-responsive", A3="A3 purchased record", A3b="A3b rival downrating")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 9.5, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2})
os.makedirs("figures", exist_ok=True)


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def zero(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=INK2, lw=0.9, zorder=1)


def dotrow(ax, ys, means, errs, colors):
    for y, m, e, c in zip(ys, means, errs, colors):
        ax.plot([m - e, m + e], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=6.5, color=c, mec=SURF, mew=1.5, zorder=3)


att = pd.read_csv("results/e1_paired.csv")

# ---------------- Figure 1: coalition gain at defaults
d = att[(att.part == "main") & (att.K == 2) & (att["T"] == 40) & (att.p_audit == 0.05)]
fig, axes = plt.subplots(2, 3, figsize=(10, 4.6))
for ax, a in zip(axes.flat, ATT):
    rows = [ci(d[(d.system == s) & (d.attack == a)].gain) for s, _ in SYS]
    ys = np.arange(len(SYS))[::-1]
    dotrow(ax, ys, [r[0] for r in rows], [r[1] for r in rows], [SYS_C[s] for s, _ in SYS])
    zero(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in SYS] if a in ("A1", "A1R") else [])
    ax.set_title(ATT[a], loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("coalition gain vs honest play")
fig.suptitle("At 5% verifiable auditing, the CTI makes the profitable attacks unprofitable", x=0.01, ha="left",
             fontsize=11, color=INK, fontweight="bold")
fig.tight_layout()
save(fig, "fig1_gain_by_system")

# ---------------- Figure 2: gain vs audit rate on the fine grid, R1 vs R2
th = pd.read_csv("results/thresholds_fine.csv")
f = att[att.part == "fine"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
for ax, a in zip(axes, ("A1", "A2", "A5")):
    for s, name, col in (("R1", "Audit-only", C["orange"]), ("R2", "CTI", C["blue"])):
        g = f[(f.system == s) & (f.attack == a)].groupby("p_audit").gain
        m, e = g.mean(), 1.96 * g.sem()
        ax.fill_between(m.index, m - e, m + e, color=col, alpha=0.15, lw=0)
        ax.plot(m.index, m.values, color=col)
        ax.text(m.index[-1] + 0.004, m.values[-1], name, color=INK, va="center", fontsize=8.5)
    zero(ax, vertical=False)
    parts = []
    for s_, name, col in (("R2", "CTI", C["blue"]), ("R1", "audit-only", C["orange"])):
        r = th[(th.system == s_) & (th.attack == a)].iloc[0]
        if np.isfinite(r.point):
            ax.axvline(r.point, color=col, lw=1, ls=(0, (3, 3)))
            parts.append(f"{name} p* = {r.point:.2f} [{r.lo:.2f}, {r.hi:.2f}]")
        else:
            parts.append(f"{name}: threshold not reached by 0.20")
    ax.text(0, 1.01, "  ·  ".join(parts), transform=ax.transAxes, fontsize=7.2, color=INK2, va="bottom")
    ax.set_xlim(0, 0.235)
    ax.set_title(ATT[a] + "\n", loc="left")
    ax.set_xlabel("audit rate")
axes[0].set_ylabel("coalition gain vs honest play")
fig.suptitle("Verifiable audits: under the CTI the attacks become unprofitable near 4% auditing; under audit-only, resets and bait-and-switch stay profitable",
             x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
fig.tight_layout()
save(fig, "fig2_gain_vs_audit_rate")

# ---------------- Figure 3: components (bond sweep; A5 surcharges)
b = att[(att.part == "bond") & (att.p_audit == 0.05)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
for a, col in (("A1", C["blue"]), ("A2", C["orange"]), ("A5", C["aqua"])):
    g = b[(b.system == "R2") & (b.attack == a)].groupby("bond").gain
    m, e = g.mean(), 1.96 * g.sem()
    off = {"A1": -4, "A2": 0, "A5": 4}[a]
    ax1.errorbar(m.index + off, m.values, yerr=e.values, fmt="none", ecolor=col, elinewidth=1.4, capsize=0)
    ax1.plot(m.index + off, m.values, "-o", color=col, ms=6.5, mec=SURF, mew=1.5)
    ax1.text(m.index[-1] + 14, m.values[-1], a, color=INK, va="center", fontsize=8.5)
g = b[(b.system == "R2_nodecl") & (b.attack == "A5")].groupby("bond").gain.mean()
ax1.plot(g.index, g.values, color=C["aqua"], lw=1.2, ls=(0, (3, 3)))
ax1.text(g.index[-1] + 6, g.values[-1], "A5 without\nbinding terms", color=INK2, va="center", fontsize=7.5)
zero(ax1, vertical=False)
ax1.set_xlim(-10, 250)
ax1.set_xlabel("bond b (CTI, 5% auditing)")
ax1.set_ylabel("coalition gain vs honest play")
ax1.set_title("Larger bonds make A1 and A2 unprofitable; bait-and-switch also needs binding terms", loc="left")
d5 = att[(att.part == "main") & (att.K == 2) & (att["T"] == 40) & (att.p_audit == 0.05) & (att.attack == "A5")]
ys = np.arange(len(SYS))[::-1]
rows = [ci(d5[d5.system == s].unexpected_per_coal_tx) for s, _ in SYS]
dotrow(ax2, ys, [r[0] for r in rows], [r[1] for r in rows], [SYS_C[s] for s, _ in SYS])
for y, (s, _) in zip(ys, SYS):
    v = d5[d5.system == s].coal_tx_honest.mean()
    ax2.text(1.62, y, f"{v:.0f} tx", color=INK2, va="center", fontsize=7.5)
ax2.set_yticks(ys)
ax2.set_yticklabels([n for _, n in SYS])
ax2.set_xlim(0, 1.75)
ax2.grid(axis="y", visible=False)
ax2.set_xlabel("A5 surcharge per honest-buyer transaction with the coalition")
ax2.set_title("Binding terms cut surcharges per transaction and volume", loc="left")
fig.tight_layout()
save(fig, "fig3_components")

# ---------------- Figure 4: H1 value of purchased history
h = pd.read_csv("results/h1_value_of_purchases.csv")
panels = [("default", "Default market"), ("sensitive", "Reputation-sensitive market (exploratory)"),
          ("free_all", "Sensitive, free fakes everywhere (diagnostic)")]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7), sharey=True)
sys3 = [("R0", "Peer rating"), ("R1", "Audit-only"), ("R2", "CTI")]
for ax, (mk, title) in zip(axes, panels):
    for cmp, col, dy, lab in (("A3-A1", C["blue"], 0.12, "buy before cheating (A3)"),
                              ("A3d-A1", C["orange"], -0.12, "keep buying to dilute bad ratings (A3d)")):
        for i, (s_, _) in enumerate(sys3):
            r_ = h[(h.market == mk) & (h.system == s_) & (h.comparison == cmp)].iloc[0]
            y = len(sys3) - 1 - i + dy
            ax.plot([r_.lo, r_.hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r_["mean"], y, "o", ms=6.5, color=col, mec=SURF, mew=1.5, label=lab if i == 0 else None)
    zero(ax)
    ax.set_title(title, loc="left", fontsize=8.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("gain from purchased history")
axes[0].set_yticks(range(len(sys3))[::-1])
axes[0].set_yticklabels([n for _, n in sys3])
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7.8, frameon=False)
fig.suptitle("Purchased history did not pay by default; with scarce reputation, free fake ratings paid and audited systems reduced the gain",
             x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
fig.tight_layout()
save(fig, "fig4_purchased_history")

# ---------------- Figure 5: value of voluntary reset
r = pd.read_csv("results/reset_diag.csv")
r = r[r.eligible]
rows = [(False, "R0", "Peer rating"), (False, "R1", "Audit-only"), (False, "R2", "CTI"),
        (True, "R1", "Audit-only, noisy audits"), (True, "R2", "CTI, noisy audits")]
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for k, (nok, col, dy, lab) in enumerate(((True, C["blue"], 0.12, "replacement gets newcomer quota"),
                                         (False, C["orange"], -0.12, "no newcomer quota"))):
    for i, (noisy, s, name) in enumerate(rows):
        g = r[(r.noisy == noisy) & (r.system == s) & (r.newcomer_ok == nok)].value_profit
        m, e = ci(g)
        y = len(rows) - 1 - i + dy
        ax.plot([m - e, m + e], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", ms=6.5, color=col, mec=SURF, mew=1.5, label=lab if i == 0 else None)
zero(ax)
ax.set_yticks(range(len(rows))[::-1])
ax.set_yticklabels([n for _, _, n in rows])
ax.grid(axis="y", visible=False)
ax.set_xlabel("value of a voluntary reset (controller profit, final 3 rounds)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("Late resets paid only when the replacement kept newcomer eligibility", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
save(fig, "fig5_reset_value")

# ---------------- Figure 6: noisy audits
e2 = pd.read_csv("results/e2_paired.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), sharey=True)
for ax, a, title in ((axes[0], "A1", "A1 selective defection"), (axes[1], "A1S", "Sub-threshold attacker")):
    for s, name, col in (("R1", "Audit-only", C["orange"]), ("R2", "CTI", C["blue"])):
        g = e2[(e2.system == s) & (e2.attack == a)].groupby("p_audit").gain
        m, e = g.mean(), 1.96 * g.sem()
        ax.fill_between(m.index, m - e, m + e, color=col, alpha=0.15, lw=0)
        ax.plot(m.index, m.values, "-o", color=col, ms=6.5, mec=SURF, mew=1.5)
        ax.text(m.index[-1] + 0.006, m.values[-1], name, color=INK, va="center", fontsize=8.5)
    if a == "A1":
        ref = f[(f.system == "R2") & (f.attack == "A1") & (f.p_audit.isin([0.05, 0.1, 0.2]))].groupby("p_audit").gain.mean()
        ax.plot(ref.index, ref.values, color=C["blue"], lw=1.2, ls=(0, (3, 3)))
        ax.text(ref.index[-1] + 0.006, ref.values[-1], "CTI, verifiable", color=INK2, va="center", fontsize=7.5)
    zero(ax, vertical=False)
    ax.set_xlim(0.03, 0.25)
    ax.set_xticks([0.05, 0.10, 0.20])
    ax.set_title(title, loc="left")
    ax.set_xlabel("audit rate")
axes[0].set_ylabel("coalition gain vs honest play")
fig.suptitle("Noisy audits (pooled over noise settings): the CTI lowers attack profit but does not consistently remove it",
             x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
fig.tight_layout()
save(fig, "fig6_noisy_audits")
print("ok")
