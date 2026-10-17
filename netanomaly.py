"""
Ad-delivery network anomaly detection with NetworkX (synthetic data).

Pipeline: build graph -> validate records -> clean -> graph-based checks
-> compare against injected ground truth -> plot.

Run:  python netanomaly.py
"""
import json

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
LAYERS = ["channel", "platform", "exchange", "advertiser"]
RANK = {name: i for i, name in enumerate(LAYERS)}


# --------------------------------------------------------------------------
# 1. Synthetic data: channels -> platforms -> exchanges -> advertisers
#    Edge weight = impressions. Intermediaries pass on 93-100% of what they get.
# --------------------------------------------------------------------------
def build_graph(seed=SEED):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    ch = [f"CH{i:02d}" for i in range(12)]
    pf = [f"PF{i:02d}" for i in range(6)]
    ex = [f"EX{i:02d}" for i in range(4)]
    ad = [f"AD{i:02d}" for i in range(10)]
    for names, layer in zip([ch, pf, ex, ad], LAYERS):
        for n in names:
            G.add_node(n, layer=layer, rank=RANK[layer])

    def inflow(n):
        return sum(d["impressions"] for _, _, d in G.in_edges(n, data=True))

    # channel -> platform (each platform gets at least two channels)
    for i, c in enumerate(ch):
        first = pf[i % 6]
        second = rng.choice([p for p in pf if p != first])
        for p in (first, second):
            G.add_edge(c, p, impressions=float(rng.lognormal(12, 0.5)))

    # platform -> exchange
    for i, p in enumerate(pf):
        k = int(rng.integers(2, 4))
        others = [e for e in ex if e != ex[i % 4]]
        targets = [ex[i % 4]] + list(rng.choice(others, size=k - 1, replace=False))
        out_total = inflow(p) * rng.uniform(0.93, 1.0)
        for t, sh in zip(targets, rng.dirichlet(np.ones(k))):
            G.add_edge(p, t, impressions=float(out_total * sh))

    # ANOMALY 1: PF03 reports 1.8x more impressions than it received
    for _, _, d in G.out_edges("PF03", data=True):
        d["impressions"] *= 1.8

    # exchange -> advertiser (built after A1 so downstream flow stays consistent)
    for i, e in enumerate(ex):
        k = 3
        targets = [ad[(i * 3 + j) % 10] for j in range(k)]
        out_total = inflow(e) * rng.uniform(0.93, 1.0)
        for t, sh in zip(targets, rng.dirichlet(np.ones(k))):
            G.add_edge(e, t, impressions=float(out_total * sh))

    truth = {"A1_inflated_flow": {"PF03"}}

    # ANOMALY 2: reseller loop PF01 -> EX01 -> PF01
    if not G.has_edge("PF01", "EX01"):
        G.add_edge("PF01", "EX01", impressions=40_000.0)
    G.add_edge("EX01", "PF01", impressions=0.04 * inflow("EX01"))
    truth["A2_cycle"] = {"PF01", "EX01"}

    # ANOMALY 3: ghost exchange sending impressions that nobody sent to it
    G.add_node("EX99", layer="exchange", rank=RANK["exchange"])
    for a in ("AD02", "AD07", "AD08"):
        G.add_edge("EX99", a, impressions=float(rng.lognormal(11.5, 0.3)))
    truth["A3_orphan_source"] = {"EX99"}

    # ANOMALY 4: bad records (self-loop, negative value, NaN)
    G.add_edge("PF02", "PF02", impressions=5000.0)
    t_neg = next(p for p in pf if not G.has_edge("CH11", p))
    G.add_edge("CH11", t_neg, impressions=-12_000.0)
    t_nan = next(p for p in pf if not G.has_edge("CH10", p))
    G.add_edge("CH10", t_nan, impressions=float("nan"))
    truth["A4_bad_records"] = {"PF02", "CH11", t_neg, "CH10", t_nan}

    # ANOMALY 5: tiny disconnected island
    G.add_node("CH90", layer="channel", rank=RANK["channel"])
    G.add_node("PF90", layer="platform", rank=RANK["platform"])
    G.add_edge("CH90", "PF90", impressions=8_000.0)
    truth["A5_island"] = {"CH90", "PF90"}

    # ANOMALY 6: one channel spraying tiny traffic at every platform
    for p in pf:
        if not G.has_edge("CH07", p):
            G.add_edge("CH07", p, impressions=500.0)
    truth["A6_degree_outlier"] = {"CH07"}

    return G, truth


# --------------------------------------------------------------------------
# 2. Detection
# --------------------------------------------------------------------------
def finding(check, severity, nodes, detail):
    return {"check": check, "severity": severity, "nodes": sorted(nodes), "detail": detail}


def validate_records(G):
    """Rule-based record validation. Returns findings and the edges to drop."""
    findings, bad = [], []
    for u, v, d in G.edges(data=True):
        w = d.get("impressions")
        if u == v:
            findings.append(finding("invalid_record", "high", [u], f"self-loop on {u}"))
            bad.append((u, v))
        elif w is None or not np.isfinite(w):
            findings.append(finding("invalid_record", "high", [u, v], f"missing/NaN impressions on {u}->{v}"))
            bad.append((u, v))
        elif w < 0:
            findings.append(finding("invalid_record", "high", [u, v], f"negative impressions ({w:,.0f}) on {u}->{v}"))
            bad.append((u, v))
    return findings, bad


def check_structure(H):
    findings = []
    # Edges that go backwards (or sideways) in the supply chain
    for u, v in H.edges:
        if H.nodes[v]["rank"] <= H.nodes[u]["rank"]:
            findings.append(finding("backward_edge", "high", [u, v], f"{u}->{v} flows against the supply chain"))
    # Cycles: impressions should never loop
    for cyc in nx.simple_cycles(H):
        findings.append(finding("cycle", "high", cyc, "cycle: " + " -> ".join(cyc + [cyc[0]])))
    # Sources and sinks in the wrong place
    for n, d in H.nodes(data=True):
        if d["layer"] != "channel" and H.in_degree(n) == 0:
            findings.append(finding("orphan_source", "high", [n], f"{n} sends traffic but receives none"))
        if d["layer"] != "advertiser" and H.out_degree(n) == 0:
            findings.append(finding("dead_end", "medium", [n], f"{n} receives traffic but passes none on"))
    # Small disconnected components
    comps = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
    for comp in comps[1:]:
        findings.append(finding("island", "medium", comp, f"disconnected component of {len(comp)} nodes"))
    return findings


def check_flow(H, low=0.90, high=1.02):
    """Intermediaries should pass on 90-102% of the impressions they receive."""
    findings = []
    for n, d in H.nodes(data=True):
        if d["layer"] not in ("platform", "exchange"):
            continue
        inn = H.in_degree(n, weight="impressions")
        out = H.out_degree(n, weight="impressions")
        if inn <= 0:
            continue
        ratio = out / inn
        if ratio > high or ratio < low:
            sev = "high" if ratio > high else "medium"
            findings.append(finding("flow_imbalance", sev, [n], f"{n} out/in ratio = {ratio:.2f}"))
    return findings


def check_degree(H, k=3.0):
    """Robust per-layer degree outliers (median + k * scaled MAD, floor of 1)."""
    findings = []
    for layer in LAYERS:
        nodes = [n for n, d in H.nodes(data=True) if d["layer"] == layer]
        deg = np.array([H.degree(n) for n in nodes], dtype=float)
        med = np.median(deg)
        mad = np.median(np.abs(deg - med)) * 1.4826
        limit = med + k * max(mad, 1.0)
        for n, dg in zip(nodes, deg):
            if dg > limit:
                findings.append(finding("degree_outlier", "medium", [n],
                                        f"{n} degree {int(dg)} vs layer median {med:.0f}"))
    return findings


def run_detection(G):
    findings, bad = validate_records(G)
    H = G.copy()
    H.remove_edges_from(bad)  # analyse the cleaned graph
    findings += check_structure(H) + check_flow(H) + check_degree(H)
    return pd.DataFrame(findings), H


# --------------------------------------------------------------------------
# 3. Evaluation against the injected ground truth
# --------------------------------------------------------------------------
def evaluate(df, truth):
    rows = []
    for name, nodes in truth.items():
        hit = df[df["nodes"].apply(lambda ns: bool(set(ns) & nodes))]
        rows.append({"injected": name, "detected": len(hit) > 0,
                     "caught_by": ", ".join(sorted(hit["check"].unique()))})
    all_truth = set().union(*truth.values())
    fp = df[df["nodes"].apply(lambda ns: not (set(ns) & all_truth))]
    return pd.DataFrame(rows), fp


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------
def plot(H, df, path="network.png"):
    flagged = set().union(*df["nodes"]) if len(df) else set()
    pos = nx.multipartite_layout(H, subset_key="rank", align="vertical")
    colors = ["#d62728" if n in flagged else "#4c78a8" for n in H.nodes]
    widths = [0.4 + 0.25 * np.log10(max(d["impressions"], 1)) for _, _, d in H.edges(data=True)]
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, width=widths, alpha=0.35, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=420)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=7, font_color="white")
    ax.set_title("Synthetic ad-delivery network — red nodes flagged by at least one check\n"
                 "channels → platforms → exchanges → advertisers")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    G, truth = build_graph()
    df, H = run_detection(G)
    result, fp = evaluate(df, truth)

    df_out = df.copy()
    df_out["nodes"] = df_out["nodes"].apply(lambda ns: ";".join(ns))
    df_out.to_csv("findings.csv", index=False)
    with open("ground_truth.json", "w") as f:
        json.dump({k: sorted(v) for k, v in truth.items()}, f, indent=2)
    plot(H, df)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Findings: {len(df)}\n")
    print(df.groupby("check").size().to_string(), "\n")
    print(result.to_string(index=False), "\n")
    print(f"Detected {int(result['detected'].sum())}/{len(result)} injected anomalies")
    print(f"Findings not tied to any injected anomaly (false positives): {len(fp)}")
    if len(fp):
        print(fp[["check", "nodes", "detail"]].to_string(index=False))
