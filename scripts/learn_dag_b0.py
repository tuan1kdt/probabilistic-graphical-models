"""Học DAG B0 từ UCI Bank Marketing (bank-additional-full.csv).

Theo research-plan.md Phase 2–3:
- bỏ `duration` (post-treatment), gộp job/education, bin age, gom 5 biến vĩ mô thành `period`
- fit bin trên TRAIN (split 70/15/15 stratified theo y, seed 0)
- HillClimbSearch(bic-d, max_indegree=3) + ExpertKnowledge(temporal_order, forbidden_edges)
- xuất cạnh, hình DAG, CPD của y vào artifacts/dag/
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "dag"; OUT.mkdir(parents=True, exist_ok=True)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ---------- 1. tiền xử lý / rời rạc hoá ----------
JOB_GRP = {"admin.": "white_collar", "management": "white_collar", "technician": "white_collar",
           "entrepreneur": "white_collar", "self-employed": "white_collar",
           "blue-collar": "blue_collar", "services": "blue_collar", "housemaid": "blue_collar",
           "retired": "not_working", "student": "not_working", "unemployed": "not_working",
           "unknown": "unknown"}
EDU_GRP = {"basic.4y": "basic", "basic.6y": "basic", "basic.9y": "basic", "illiterate": "basic",
           "high.school": "high_school", "professional.course": "tertiary",
           "university.degree": "tertiary", "unknown": "unknown"}

def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    d["age_bin"] = pd.cut(raw["age"], [0, 29, 39, 49, 59, 200],
                          labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    d["job"] = raw["job"].map(JOB_GRP)
    d["education"] = raw["education"].map(EDU_GRP)
    d["marital"] = raw["marital"].replace({"unknown": "married"})      # 80 dòng unknown
    d["default"] = raw["default"].replace({"yes": "unknown"})          # chỉ 3 dòng yes
    d["housing"] = raw["housing"]
    d["loan"] = raw["loan"]
    d["poutcome"] = raw["poutcome"]                                     # gói pdays/previous
    # period: giai đoạn chiến dịch theo lãi suất euribor3m (proxy 5 biến vĩ mô)
    d["period"] = pd.cut(raw["euribor3m"], [-1, 1, 4, 10],
                         labels=["low_rate", "mid_rate", "high_rate"]).astype(str)
    d["contact"] = raw["contact"]                                       # T
    d["campaign"] = pd.cut(raw["campaign"], [0, 1, 3, 100],
                           labels=["1", "2-3", "4+"]).astype(str)       # mediator
    d["y"] = raw["y"]                                                   # Y
    return d

raw = pd.read_csv(ROOT / "data/raw/bank-additional-full.csv", sep=";").drop(columns=["duration"])
df = preprocess(raw)
train_idx, rest_idx = train_test_split(df.index, test_size=0.30, stratify=df["y"], random_state=SEED)
train = df.loc[train_idx]
print(f"rows total={len(df)} train={len(train)}  nodes={list(df.columns)}")
print({c: int(df[c].nunique()) for c in df.columns})

# ---------- 2. ràng buộc chuyên gia ----------
TIERS = [
    ["period", "age_bin", "job", "education", "marital", "default", "housing", "loan"],  # pre-treatment
    ["poutcome"],          # lịch sử chiến dịch trước
    ["contact"],           # treatment T
    ["campaign"],          # mediator (sau T, trước Y)
    ["y"],                 # outcome
]
nodes = [n for t in TIERS for n in t]
forbidden = [("y", n) for n in nodes if n != "y"]                       # không cạnh ra từ y
forbidden += [("contact", n) for n in TIERS[0] + TIERS[1]]              # contact không trỏ về pre-treatment
ek = ExpertKnowledge(forbidden_edges=forbidden, temporal_order=TIERS)

# ---------- 3. structure learning ----------
hc = HillClimbSearch(train[nodes])
dag = hc.estimate(scoring_method="bic-d", max_indegree=3, expert_knowledge=ek, show_progress=False)
edges = sorted(dag.edges())
print(f"\nDAG seed={SEED}: {len(edges)} cạnh, acyclic={nx.is_directed_acyclic_graph(nx.DiGraph(edges))}")
for u, v in edges: print(f"  {u} -> {v}")
print("\nparents(y)       =", sorted(dag.get_parents("y")))
print("parents(contact) =", sorted(dag.get_parents("contact")))
print("parents(campaign)=", sorted(dag.get_parents("campaign")))

# ---------- 4. fit CPD (BDeu ESS=5) để kiểm tra & xem CPD của y ----------
bn = DiscreteBayesianNetwork(edges); bn.add_nodes_from(nodes)
state_names = {c: sorted(df[c].unique()) for c in nodes}
est = BayesianEstimator(bn, train[nodes], state_names=state_names)
bn.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=5))
assert bn.check_model()
from pgmpy.estimators import BIC
print(f"\nBIC(train) = {BIC(train[nodes]).score(bn):,.0f}")

# ---------- 5. lưu ----------
(OUT / f"dag_b0_seed{SEED}_edges.json").write_text(json.dumps(
    {"seed": SEED, "nodes": nodes, "edges": edges, "tiers": TIERS, "forbidden_edges": forbidden,
     "scoring": "bic-d", "max_indegree": 3, "state_names": {k: list(map(str, v)) for k, v in state_names.items()}},
    ensure_ascii=False, indent=2))
pd.DataFrame(edges, columns=["from", "to"]).to_csv(OUT / f"dag_b0_seed{SEED}_edges.csv", index=False)
df.to_csv(ROOT / "artifacts" / "dag" / "bank_discretized.csv", index=False)

# hình: xếp lớp theo tier
pos, tier_y = {}, {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
for ti, tier in enumerate(TIERS):
    for k, n in enumerate(tier):
        stagger = 0.9 if (ti == 0 and k % 2) else 0.0      # so le hàng pre-treatment để cạnh không chồng
        pos[n] = ((k - (len(tier) - 1) / 2) * 1.6, tier_y[ti] * 1.5 + stagger)
color = {n: "#cfe2f3" for n in nodes}; color.update(contact="#f9cb9c", campaign="#fff2cc", y="#b6d7a8", poutcome="#d9d2e9", period="#d9d2e9")
G = nx.DiGraph(edges); G.add_nodes_from(nodes)
fig, ax = plt.subplots(figsize=(14, 8))
nx.draw_networkx_nodes(G, pos, node_color=[color[n] for n in G.nodes], node_size=2600, edgecolors="#444", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, node_size=2600, width=1.4, edge_color="#555",
                       connectionstyle="arc3,rad=0.15", ax=ax)
ax.set_title(f"DAG B0 — UCI Bank Marketing (HillClimb, BIC, max_indegree=3, TRAIN seed {SEED})\n"
             "lớp trên→dưới: pre-treatment · lịch sử · treatment (contact) · mediator (campaign) · outcome (y)", fontsize=11)
ax.axis("off"); fig.tight_layout()
fig.savefig(OUT / f"dag_b0_seed{SEED}.png", dpi=150)
print(f"\nSaved: {OUT}/dag_b0_seed{SEED}.png, .json, .csv, bank_discretized.csv")

# CPD của y (ở dạng P(y=yes | parents))
cpd = bn.get_cpds("y")
parents = cpd.variables[1:]
import itertools
rows = []
for combo in itertools.product(*[cpd.state_names[p] for p in parents]):
    p_yes = cpd.get_value(**dict(zip(parents, combo)), y="yes")
    rows.append(dict(zip(parents, combo), p_yes=round(float(p_yes), 3)))
cpd_y = pd.DataFrame(rows).sort_values("p_yes", ascending=False)
cpd_y.to_csv(OUT / f"cpd_y_seed{SEED}.csv", index=False)
print("\nP(y=yes | parents) — CPD của y, BDeu ESS=5:")
print(cpd_y.to_string(index=False))
