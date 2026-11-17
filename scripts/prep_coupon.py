"""Phase 1 — UCI 603 In-Vehicle Coupon: data audit, eligibility, resp_id, mã hoá, split theo người.

Theo research-plan-coupon.md mục 1–2. Chạy:  .venv/bin/python scripts/prep_coupon.py
Đầu ra:
  data/processed/coupon_discretized.csv      mọi dòng hợp lệ, đã mã hoá, kèm row_id và resp_id
  artifacts/coupon/audit.json                số liệu audit (eligibility flow, missing, block, overlap, T ⟂ người)
  artifacts/coupon/variable_dictionary.csv   từ điển biến: mức, timing, vai trò
  artifacts/coupon/splits.json               chỉ số row_id TRAIN/VAL/TEST cho 5 seed (split theo resp_id)
  artifacts/coupon/overlap_train_seed{k}.csv bảng T × C_core trên TRAIN
  artifacts/coupon/config_coupon.json        CONFIG dùng chung cho B0/H1/H2
Mọi hàm đều import được từ notebook 02.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/coupon/in-vehicle-coupon-recommendation.csv"
PROC = ROOT / "data/processed"; PROC.mkdir(parents=True, exist_ok=True)
ART = ROOT / "artifacts/coupon"; ART.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
RESP_COLS = ["gender", "age", "maritalStatus", "has_children", "education", "occupation", "income",
             "car", "Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]
SCEN_COLS = ["destination", "passenger", "weather", "temperature", "time", "coupon", "expiration",
             "toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min", "direction_same", "direction_opp"]
HABIT_OF_COUPON = {"Bar": "Bar", "Coffee House": "CoffeeHouse", "Carry out & Take away": "CarryAway",
                   "Restaurant(<20)": "RestaurantLessThan20", "Restaurant(20-50)": "Restaurant20To50"}

# ---------- Tầng thời gian của DAG (research-plan-coupon.md mục 3) ----------
TIERS = {
    "respondent": ["habit_match", "age_bin", "income_bin", "gender", "has_children"],   # tầng 1
    "scenario":   ["coupon", "destination", "time_bin", "passenger", "weather", "dist", "direction_same"],  # tầng 2
    "treatment":  ["T"],                                                                 # tầng 3
    "outcome":    ["Y"],                                                                 # tầng 4
}
NODES = [n for t in TIERS.values() for n in t]
# C_core = tập điều chỉnh tối thiểu có positivity đầy đủ trên TRAIN (min arm ≥ 100 ở mọi seed).
# time_bin và weather cũng →T (range P(T=1) 0,14 và 0,25) nhưng tạo ô trống (Bar×NoUrgent×afternoon, Bar×Snowy...),
# nên giữ trong DAG như C mở rộng và báo cáo độ nhạy, không đưa vào tập identify cốt lõi.
C_CORE = ["coupon", "destination"]
C_EXT = C_CORE + ["time_bin", "weather"]
C_FULL = C_EXT + ["passenger", "dist", "direction_same",
                  "habit_match", "age_bin", "income_bin", "gender", "has_children"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    raw = raw.rename(columns={"passanger": "passenger"})        # lỗi chính tả trong file gốc UCI
    raw.insert(0, "row_id", np.arange(len(raw)))
    return raw


def audit_raw(raw: pd.DataFrame) -> dict:
    cols = [c for c in raw.columns if c != "row_id"]
    missing = {c: int((raw[c] == "").sum()) for c in cols if (raw[c] == "").any()}
    levels = {c: int(raw[c].nunique()) for c in cols}
    constant = [c for c in cols if raw[c].nunique() == 1]
    dup = int(raw.duplicated(subset=cols).sum())
    return {"n_rows": int(len(raw)), "n_cols": len(cols), "missing_counts": missing,
            "levels": levels, "constant_cols": constant, "exact_duplicate_rows": dup,
            "y_rate": float((raw["Y"] == "1").mean()),
            "t_rate_2h": float((raw["expiration"] == "2h").mean())}


def make_resp_id(raw: pd.DataFrame) -> pd.Series:
    """Gom các dòng LIÊN TIẾP có cùng 13 cột người trả lời thành một block = một người (không có ID gốc)."""
    key = raw[RESP_COLS].astype(str).agg("|".join, axis=1)
    new_block = key.ne(key.shift(fill_value="__start__"))
    return new_block.cumsum().rename("resp_id") - 1


def recode(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"row_id": raw["row_id"].astype(int)})
    d["resp_id"] = make_resp_id(raw).astype(int)
    # tầng kịch bản
    d["coupon"] = raw["coupon"]
    d["destination"] = raw["destination"]
    d["time_bin"] = raw["time"].map({"7AM": "morning", "10AM": "morning", "2PM": "afternoon",
                                     "6PM": "evening", "10PM": "evening"})
    d["passenger"] = raw["passenger"]
    d["weather"] = raw["weather"]
    d["dist"] = np.select([raw["toCoupon_GEQ25min"] == "1", raw["toCoupon_GEQ15min"] == "1"],
                          ["25+", "15-25"], default="<15")
    d["direction_same"] = raw["direction_same"]
    # treatment / outcome
    d["T"] = (raw["expiration"] == "2h").astype(int)      # 1d = 0 (control), 2h = 1 (treatment)
    d["Y"] = raw["Y"].astype(int)
    # tầng người trả lời
    habit_col = raw["coupon"].map(HABIT_OF_COUPON)
    habit_raw = pd.Series([raw.at[i, c] for i, c in zip(raw.index, habit_col)], index=raw.index)
    d["habit_match"] = habit_raw.map({"never": "low", "less1": "low", "1~3": "mid", "4~8": "high", "gt8": "high"})
    d["age_bin"] = raw["age"].map({"below21": "<26", "21": "<26", "26": "26-35", "31": "26-35",
                                   "36": "36-50", "41": "36-50", "46": "36-50", "50plus": "50+"})
    d["income_bin"] = raw["income"].map({"Less than $12500": "<37.5k", "$12500 - $24999": "<37.5k",
                                         "$25000 - $37499": "<37.5k", "$37500 - $49999": "37.5-75k",
                                         "$50000 - $62499": "37.5-75k", "$62500 - $74999": "37.5-75k",
                                         "$75000 - $87499": "75k+", "$87500 - $99999": "75k+",
                                         "$100000 or More": "75k+"})
    d["gender"] = raw["gender"]
    d["has_children"] = raw["has_children"]
    # chỉ dùng cho ablation, không nằm trong DAG chính
    d["education_bin"] = raw["education"].map({"Some High School": "<=HS", "High School Graduate": "<=HS",
                                               "Some college - no degree": "some_college",
                                               "Associates degree": "some_college",
                                               "Bachelors degree": "bachelor+",
                                               "Graduate degree (Masters or Doctorate)": "bachelor+"})
    return d


def eligibility(raw: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Eligibility flow (slide 54): ghi số dòng trước/sau mỗi bước."""
    flow = [{"step": "raw UCI file", "n": int(len(d))}]
    keep = pd.Series(True, index=d.index)
    m = (d["coupon"] == "Bar") & (d["destination"] == "Work")
    keep &= ~m; flow.append({"step": "loại ô Bar × Work (0% T=2h, không có overlap)", "removed": int(m.sum()), "n": int(keep.sum())})
    m = keep & d["habit_match"].isna()
    keep &= ~m; flow.append({"step": "loại dòng thiếu tần suất đến đúng loại quán của coupon (habit_match)", "removed": int(m.sum()), "n": int(keep.sum())})
    # dòng trùng hoàn toàn: GIỮ (có thể là hai người cùng hồ sơ); chỉ ghi nhận
    dup = int(raw.loc[keep.values, [c for c in raw.columns if c != "row_id"]].duplicated().sum())
    flow.append({"step": "dòng trùng hoàn toàn (giữ, chỉ ghi nhận)", "removed": 0, "duplicates_present": dup, "n": int(keep.sum())})
    out = d.loc[keep].reset_index(drop=True)
    assert out[NODES].isna().sum().sum() == 0, out[NODES].isna().sum()
    return out, flow


def variable_dictionary(d: pd.DataFrame) -> pd.DataFrame:
    role = {}
    for n in TIERS["respondent"]: role[n] = ("người trả lời, trước T", "baseline C (→Y); cấm →T")
    for n in TIERS["scenario"]:
        role[n] = ("thiết kế kịch bản, trước T",
                   "confounder C_core (→T, →Y; positivity đầy đủ)" if n in C_CORE
                   else "confounder C_ext (→T, →Y; có ô trống, chỉ độ nhạy)" if n in C_EXT else "C phụ (→Y)")
    role["T"] = ("lúc hiển thị coupon", "treatment: expiration 1d=0, 2h=1")
    role["Y"] = ("sau T", "outcome: chấp nhận coupon")
    role["education_bin"] = ("người trả lời", "chỉ ablation")
    rows = []
    for c in NODES + ["education_bin"]:
        vc = d[c].value_counts()
        rows.append({"variable": c, "n_levels": int(d[c].nunique()),
                     "levels": ", ".join(f"{k}={v}" for k, v in vc.items()),
                     "timing": role[c][0], "role": role[c][1]})
    return pd.DataFrame(rows)


def make_splits(d: pd.DataFrame, seeds=SEEDS, frac=(0.7, 0.15, 0.15)) -> dict:
    """Split theo resp_id (group split), stratify theo tertile tỉ lệ Y trung bình của người."""
    resp = d.groupby("resp_id")["Y"].mean().rename("y_rate").to_frame()
    resp["strata"] = pd.qcut(resp["y_rate"].rank(method="first"), 3, labels=["lo", "mid", "hi"])
    splits = {}
    for s in seeds:
        tr, rest = train_test_split(resp.index, test_size=1 - frac[0], stratify=resp["strata"], random_state=s)
        va, te = train_test_split(rest, test_size=frac[2] / (frac[1] + frac[2]),
                                  stratify=resp.loc[rest, "strata"], random_state=s)
        part = {"train": tr, "val": va, "test": te}
        splits[str(s)] = {k: d.loc[d["resp_id"].isin(v), "row_id"].astype(int).tolist() for k, v in part.items()}
        splits[str(s)]["n_resp"] = {k: int(len(v)) for k, v in part.items()}
    return splits


def overlap_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = train.groupby(cols + ["T"]).agg(n=("Y", "size"), y_rate=("Y", "mean")).unstack("T")
    g.columns = [f"{a}_T{b}" for a, b in g.columns]
    g = g.fillna({"n_T0": 0, "n_T1": 0})
    g["min_arm"] = g[["n_T0", "n_T1"]].min(axis=1).astype(int)
    g["p_T1"] = g["n_T1"] / (g["n_T0"] + g["n_T1"])
    return g.reset_index()


def t_independence(d: pd.DataFrame) -> dict:
    """Giả định 1: T không phụ thuộc người trả lời, chỉ phụ thuộc kịch bản."""
    out = {}
    for c in TIERS["respondent"] + TIERS["scenario"]:
        r = d.groupby(c)["T"].mean()
        out[c] = {"min_pT1": round(float(r.min()), 3), "max_pT1": round(float(r.max()), 3), "range": round(float(r.max() - r.min()), 3)}
    return out


def naive_and_standardized(d: pd.DataFrame, cset: list[str]) -> dict:
    naive = float(d.loc[d["T"] == 1, "Y"].mean() - d.loc[d["T"] == 0, "Y"].mean())
    g = d.groupby(cset + ["T"])["Y"].agg(["mean", "size"]).unstack("T")
    ok = g["size"].notna().all(axis=1)
    w = g.loc[ok, "size"].sum(axis=1)
    ate = float(((g.loc[ok, "mean"][1] - g.loc[ok, "mean"][0]) * w).sum() / w.sum())
    return {"naive_diff": round(naive, 4), "standardized_ATE": round(ate, 4),
            "cells": int(len(g)), "cells_with_both_T": int(ok.sum()), "rows_covered_frac": round(float(w.sum() / len(d)), 3)}


def main():
    raw = load_raw()
    audit = {"source": {"file": str(RAW.relative_to(ROOT)), "sha256": sha256(RAW), "uci_id": 603,
                        "doi": "10.24432/C5GS4P", "license": "CC BY 4.0", "downloaded": "2026-11-06"},
             "raw": audit_raw(raw)}
    d = recode(raw)
    blocks = d.groupby("resp_id").size()
    audit["respondent_blocks"] = {"n_blocks": int(blocks.shape[0]),
                                  "size_distribution": {str(k): int(v) for k, v in blocks.value_counts().sort_index().items()},
                                  "n_distinct_respondent_keys": int(raw[RESP_COLS].drop_duplicates().shape[0])}
    d, flow = eligibility(raw, d)
    audit["eligibility_flow"] = flow
    audit["after_eligibility"] = {"n_rows": int(len(d)), "n_resp": int(d["resp_id"].nunique()),
                                  "y_rate": round(float(d["Y"].mean()), 4), "t_rate": round(float(d["T"].mean()), 4)}
    audit["T_independence_check"] = t_independence(d)
    audit["effect_preview_all_rows"] = {"C_core": naive_and_standardized(d, C_CORE),
                                        "C_ext": naive_and_standardized(d, C_EXT)}
    audit["cate_by_coupon_preview"] = {c: round(float(x["Y"][x["T"] == 1].mean() - x["Y"][x["T"] == 0].mean()), 4)
                                       for c, x in d.groupby("coupon")}

    splits = make_splits(d)
    audit["splits"] = {s: {k: len(v) for k, v in sp.items() if k != "n_resp"} | {"n_resp": sp["n_resp"]} for s, sp in splits.items()}
    audit["overlap_train"] = {}
    for s in SEEDS:
        tr = d[d["row_id"].isin(splits[str(s)]["train"])]
        rec = {}
        for name, cset in [("C_core", C_CORE), ("C_ext", C_EXT)]:
            ov = overlap_table(tr, cset)
            ov.to_csv(ART / f"overlap_train_seed{s}_{name}.csv", index=False)
            rec[name] = {"cells": int(len(ov)), "min_arm": int(ov["min_arm"].min()),
                         "cells_min_arm_lt30": int((ov["min_arm"] < 30).sum()),
                         "cells_empty_arm": int((ov["min_arm"] == 0).sum())}
        audit["overlap_train"][str(s)] = rec

    d.to_csv(PROC / "coupon_discretized.csv", index=False)
    variable_dictionary(d).to_csv(ART / "variable_dictionary.csv", index=False)
    json.dump(splits, open(ART / "splits.json", "w"))
    json.dump(audit, open(ART / "audit.json", "w"), indent=1, ensure_ascii=False)

    config = {
        "data": {"uci_id": 603, "name": "In-Vehicle Coupon Recommendation", "raw_file": str(RAW.relative_to(ROOT)),
                 "processed_file": "data/processed/coupon_discretized.csv", "group_col": "resp_id",
                 "treatment": "T", "treatment_coding": {"1d": 0, "2h": 1}, "outcome": "Y",
                 "excluded": ["Bar × Work", "rows with missing habit_match"]},
        "seeds": SEEDS, "split": {"train": 0.7, "val": 0.15, "test": 0.15, "group": "resp_id", "stratify_on": "respondent y_rate tertile"},
        "tiers": TIERS, "nodes": NODES, "C_core": C_CORE, "C_ext": C_EXT, "C_full": C_FULL,
        "forbidden_edges_rules": ["no edge out of Y", "no edge from respondent tier into T", "no edge from T into tier 1–2"],
        "required_edges": [["T", "Y"], ["coupon", "Y"], ["habit_match", "Y"]],
        "cpd": {"estimator": "BayesianEstimator", "prior_type": "BDeu", "equivalent_sample_size": 5},
        "b0": {"algorithm": "HillClimbSearch", "scoring": "bic-d", "max_indegree": 3},
        "h1": {"replace_node": "Y", "hidden": [32, 16], "dropout": 0.2, "lr": 0.001, "batch_size": 256, "max_epochs": 200, "patience": 10, "loss": "bce"},
        "h2": {"method": "notears-nonlinear", "lambda1": 0.01, "lambda2": 0.01, "edge_threshold": 0.3, "cycle_repair": "drop-weakest-edge"},
        "refute": {"placebo": {"placebo_type": "permute", "num_simulations": 100}, "subset": {"subset_fraction": 0.8, "num_simulations": 100}},
        "uncertainty": {"method": "bootstrap-train-by-resp_id-fixed-test", "n_boot": 200, "level": 0.95},
    }
    json.dump(config, open(ART / "config_coupon.json", "w"), indent=1, ensure_ascii=False)
    return raw, d, audit, splits


if __name__ == "__main__":
    raw, d, audit, splits = main()
    print(json.dumps({k: audit[k] for k in ["raw", "respondent_blocks", "eligibility_flow", "after_eligibility",
                                            "effect_preview_all_rows", "cate_by_coupon_preview", "splits", "overlap_train"]},
                     indent=1, ensure_ascii=False))
    print("\nT independence (range of P(T=1) across levels):")
    for k, v in audit["T_independence_check"].items():
        print(f"  {k:16s} {v['min_pT1']:.3f}–{v['max_pT1']:.3f}  range={v['range']:.3f}")
