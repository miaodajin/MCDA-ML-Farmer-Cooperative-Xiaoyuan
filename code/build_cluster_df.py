import numpy as np
import pandas as pd
from pathlib import Path

MERGE_KEY = "varID"
RAW_PATH = Path(__file__).resolve().parent / "raw_df.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "cluster_df.csv"
READ_ENCODING = "utf-8-sig"
WRITE_ENCODING = "utf-8-sig"

# 特殊缺失编码统一替换为真正缺失
SPECIAL_MISSING = {".", "", "8888", "8888.0", "9999", "9999.0", "不知道", 8888, 9999}

# 连续变量
CONTINUOUS_COLS = [
    "var3_1",
    "var3_6",
    "var6_7",
    "var6_47",
    "var6_49",
    "var6_52",
    "var6_54",
    "var8_5",
]

# 候选特征列
CANDIDATE_COLS = [
    "var3_1",
    "var3_6",
    "var6_7",
    "var6_47",
    "var6_49",
    "var6_52",
    "var6_54",
    "var8_5",
    "var8_4",
    "var8_18",
    "var8_19",
    "var8_12",
    "var5_16",
    "var9_4",
    "var9_16",
    "var9_12",
    "var10_2",
    "var10_15",
    "var10_16",
    "var10_18",
    "var10_26",
    "var10_30",
    "var20_12",
    "var20_11",
    "var20_13",
    "var20_14",
    "var20_15",
]

# 重命名映射
RENAME_MAP = {
    "var3_1": "land_area",
    "var3_6": "plot_num",
    "var6_7": "tillage_mech_ratio",
    "var6_47": "chem_fert_kg_per_mu",
    "var6_49": "organic_fert_self",
    "var6_52": "organic_fert_buy",
    "var6_54": "irrigated_area_ratio",
    "var8_5": "mobile_hours_per_day",
    "var8_4": "mobile_difficulty",
    "var8_18": "internet_training",
    "var8_19": "e_commerce",
    "var8_12": "can_get_info_online",
    "var5_16": "ag_insurance",
    "var9_4": "bank_credit_user",
    "var9_16": "applied_bank_loan",
    "trade_credit": "trade_credit",
    "safe_drinking_water": "safe_drinking_water",
    "var10_15": "garbage_sorting",
    "var10_16": "garbage_central_treatment",
    "var10_18": "sanitary_toilet",
    "var20_12": "health_knowledge_learning",
    "has_checkup": "has_checkup",
    "straw_utilized": "straw_utilized",
    "pesticide_pack_safe": "pesticide_pack_safe",
    "organic_fert_total": "organic_fert_total",
    "diet_control_score": "diet_control_score",
}


def replace_special_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(SPECIAL_MISSING, np.nan)
    return df


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarize_continuous(df: pd.DataFrame, col: str):
    ser = df[col].dropna()
    if ser.empty:
        print(f"{col}: 全部缺失，无法计算分布。")
        return
    quantiles = ser.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print(
        f"{col} min={ser.min():.3f}, max={ser.max():.3f}, "
        f"p1={quantiles.loc[0.01]:.3f}, p5={quantiles.loc[0.05]:.3f}, "
        f"p50={quantiles.loc[0.5]:.3f}, p95={quantiles.loc[0.95]:.3f}, "
        f"p99={quantiles.loc[0.99]:.3f}"
    )


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    # 截尾以避免极端值主导聚类
    return series.clip(lower=lo, upper=hi)


def encode_yes_no(series: pd.Series, col_name: str) -> pd.Series:
    print(f"{col_name} 唯一取值: {series.dropna().unique()[:20]}")
    s = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    yes_values = {"1", "1.0", "是"}
    no_values = {"0", "0.0", "否", "2", "2.0"}
    return s.map(lambda x: 1 if x in yes_values else (0 if x in no_values else np.nan))


def encode_ordered_can_get_info(series: pd.Series) -> pd.Series:
    print(f"var8_12 唯一取值: {series.dropna().unique()[:20]}")
    mapping = {"完全可以": 2, "有时可以": 1, "比较困难": 0}
    s = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    return s.map(mapping)


def encode_trade_credit(series: pd.Series) -> pd.Series:
    print(f"var9_12 唯一取值: {series.dropna().unique()[:20]}")
    s = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    mapping = {"从未": 0, "偶尔": 1, "经常": 1}
    return s.map(lambda x: mapping.get(x, np.nan))


def encode_checkup(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype(str).str.strip()
    print(f"var20_11 唯一取值样例: {raw.dropna().unique()[:20]}")
    def mapper(val: str):
        if val in {"nan", ""}:
            return np.nan
        parts = [p.strip() for p in val.replace("，", ",").split(",") if p.strip()]
        if not parts:
            return np.nan
        if any(p == "1" or p == "2" for p in parts):
            return 1
        if parts == ["3"]:
            return 0
        if any(p in {"4", "9999"} for p in parts):
            return np.nan
        return np.nan
    has_checkup = raw.map(mapper)
    return raw, has_checkup


def extract_codes_from_string(series: pd.Series) -> list[set[str]]:
    codes_list = []
    for val in series:
        if pd.isna(val):
            codes_list.append(set())
            continue
        text = str(val).strip().replace("，", ",")
        if not text:
            codes_list.append(set())
            continue
        codes = {item.strip() for item in text.split(",") if item.strip()}
        codes_list.append(codes)
    return codes_list


def extract_codes_from_onehot(df: pd.DataFrame, prefix: str) -> list[set[str]]:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    codes_list = []
    for _, row in df[cols].iterrows():
        codes = set()
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            text = str(val).strip()
            if text in {"0", "0.0", "", "nan"}:
                continue
            code = col.replace(f"{prefix}_", "")
            codes.add(code)
        codes_list.append(codes)
    return codes_list


def derive_straw_utilized(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("var10_26_")]
    if cols:
        print(f"检测到 var10_26 拆分列: {cols}")
        codes_list = extract_codes_from_onehot(df, "var10_26")
    elif "var10_26" in df.columns:
        print("使用 var10_26 单列字符串编码。")
        codes_list = extract_codes_from_string(df["var10_26"])
    else:
        print("未找到 var10_26 数据，straw_utilized 无法计算。")
        return pd.Series([np.nan] * len(df))
    safe_set = {"2", "3", "4", "5"}
    result = []
    for codes in codes_list:
        if not codes:
            result.append(np.nan)
            continue
        if codes & safe_set:
            result.append(1)  # 混合选择含 2/3/4/5 时按利用记 1
            continue
        if codes <= {"1"}:
            result.append(0)
            continue
        result.append(np.nan)
    return pd.Series(result, index=df.index)


def derive_pesticide_pack_safe(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("var10_30_")]
    if cols:
        print(f"检测到 var10_30 拆分列: {cols}")
        codes_list = extract_codes_from_onehot(df, "var10_30")
    elif "var10_30" in df.columns:
        print("使用 var10_30 单列字符串编码。")
        codes_list = extract_codes_from_string(df["var10_30"])
    else:
        print("未找到 var10_30 数据，pesticide_pack_safe 无法计算。")
        return pd.Series([np.nan] * len(df))
    safe_set = {"2", "4", "5"}
    unsafe_set = {"1", "3", "6"}
    result = []
    for codes in codes_list:
        if not codes:
            result.append(np.nan)
            continue
        if codes & safe_set:
            result.append(1)  # 安全与不安全混合时也按 1 处理
            continue
        if codes and codes <= unsafe_set:
            result.append(0)
            continue
        if codes == {"7"}:
            result.append(np.nan)
            continue
        result.append(np.nan)
    return pd.Series(result, index=df.index)


def main():
    print(f"读取 {RAW_PATH}")
    raw_df = pd.read_csv(RAW_PATH, encoding=READ_ENCODING)
    dup_count = raw_df.duplicated(subset=MERGE_KEY).sum()
    if dup_count:
        print(f"检测到 {dup_count} 个重复 {MERGE_KEY}，保留第一条记录。")
        raw_df = raw_df.drop_duplicates(subset=MERGE_KEY, keep="first")
    else:
        print("varID 唯一，无重复。")

    raw_df = replace_special_missing(raw_df)
    print("替换特殊缺失后的缺失示例：")
    print(raw_df.isna().sum().head(10))

    existing_cols = [c for c in CANDIDATE_COLS if c in raw_df.columns]
    missing_cols = sorted(set(CANDIDATE_COLS) - set(existing_cols))
    if missing_cols:
        print(f"以下列不存在，将跳过：{missing_cols}")
    df = raw_df[[MERGE_KEY] + existing_cols].copy()

    # 连续变量类型转换与截尾
    for col in CONTINUOUS_COLS:
        if col not in df.columns:
            print(f"缺少连续变量 {col}，跳过。")
            continue
        df[col] = coerce_numeric(df[col])
        if col == "var8_5":
            df.loc[df[col] > 24, col] = np.nan  # 手机时长超过 24 小时视为缺失
        summarize_continuous(df, col)
        df[col] = winsorize_series(df[col])  # 1%-99% 截尾，防止极端值主导聚类

    # 是/否类变量统一编码
    yes_no_cols = [
        "var5_16",
        "var8_18",
        "var8_19",
        "var9_4",
        "var9_16",
        "var10_15",
        "var10_16",
        "var10_18",
        "var20_12",
        "var10_2",
        "var8_4",
    ]
    for col in yes_no_cols:
        if col not in df.columns:
            print(f"缺少是/否变量 {col}，跳过。")
            continue
        df[col] = encode_yes_no(df[col], col)

    # var8_12 有序编码
    if "var8_12" in df.columns:
        df["var8_12"] = encode_ordered_can_get_info(df["var8_12"])
    else:
        print("缺少 var8_12，无法进行有序编码。")

    # trade_credit 派生
    if "var9_12" in df.columns:
        df["trade_credit"] = encode_trade_credit(df["var9_12"])
    else:
        print("缺少 var9_12，trade_credit 无法计算。")

    # 体检情况处理
    var20_11_raw = None
    if "var20_11" in df.columns:
        var20_11_raw, df["has_checkup"] = encode_checkup(df["var20_11"])
    else:
        print("缺少 var20_11，has_checkup 无法计算。")

    # 秸秆处理
    df["straw_utilized"] = derive_straw_utilized(df)

    # 农药包装处置
    df["pesticide_pack_safe"] = derive_pesticide_pack_safe(df)

    # 安全饮水编码
    if "var10_2" in df.columns:
        df["safe_drinking_water"] = df["var10_2"]
    else:
        df["safe_drinking_water"] = np.nan
    df["safe_drinking_water"] = df["safe_drinking_water"].map(
        lambda x: 1
        if (not pd.isna(x)) and str(x).strip() in {"1", "1.0"}
        else (0 if (not pd.isna(x)) and str(x).strip() in {"2", "2.0"} else np.nan)
    )

    # 有机肥合计
    if {"var6_49", "var6_52"} <= set(df.columns):
        df["organic_fert_total"] = df["var6_49"].fillna(0) + df["var6_52"].fillna(0)
    else:
        df["organic_fert_total"] = np.nan

    # 饮食控制得分（可选）
    if {"var20_13", "var20_14", "var20_15"} <= set(df.columns):
        diet_cols = ["var20_13", "var20_14", "var20_15"]
        df["diet_control_score"] = df[diet_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        df["diet_control_score"] = np.nan

    # 重命名并收集特征
    feature_df = pd.DataFrame()
    feature_df[MERGE_KEY] = df[MERGE_KEY]
    for orig, new in RENAME_MAP.items():
        if orig in df.columns:
            feature_df[new] = df[orig]
        else:
            print(f"{orig} 缺失，{new} 未生成。")
    if var20_11_raw is not None:
        feature_df["var20_11_raw"] = var20_11_raw

    # 缺失比例检查
    feature_cols = [c for c in feature_df.columns if c not in {MERGE_KEY, "var20_11_raw"}]
    missing_ratio = feature_df[feature_cols].isna().mean().sort_values(ascending=False)
    print("特征缺失比例（降序）：")
    print(missing_ratio)
    high_missing = missing_ratio[missing_ratio > 0.7].index.tolist()
    if high_missing:
        print(f"缺失比例>70%的特征将剔除：{high_missing}")
        feature_cols = [c for c in feature_cols if c not in high_missing]
    else:
        print("无缺失比例>70%的特征。")

    # 行层面缺失比例
    missing_frac = feature_df[feature_cols].isna().mean(axis=1)
    print("每行缺失比例描述：")
    print(missing_frac.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
    keep_mask = missing_frac <= 0.5
    dropped_rows = (~keep_mask).sum()
    if dropped_rows:
        print(f"删除缺失比例>50%的样本 {dropped_rows} 行。")
    feature_df = feature_df.loc[keep_mask].reset_index(drop=True)

    # 缺失值填补
    continuous_features = [
        "land_area",
        "plot_num",
        "tillage_mech_ratio",
        "chem_fert_kg_per_mu",
        "organic_fert_self",
        "organic_fert_buy",
        "organic_fert_total",
        "irrigated_area_ratio",
        "mobile_hours_per_day",
    ]
    median_fill_cols = [c for c in continuous_features if c in feature_cols]
    mode_fill_cols = [c for c in feature_cols if c not in median_fill_cols]

    print("填补前缺失计数（选用特征）：")
    print(feature_df[feature_cols].isna().sum())

    for col in median_fill_cols:
        median_val = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(median_val)
    for col in mode_fill_cols:
        mode_series = feature_df[col].mode(dropna=True)
        if mode_series.empty:
            fill_val = 0
        else:
            fill_val = mode_series.iloc[0]
        feature_df[col] = feature_df[col].fillna(fill_val)

    print("填补后缺失计数（应为 0）：")
    print(feature_df[feature_cols].isna().sum())

    cluster_df = feature_df[[MERGE_KEY] + feature_cols]
    print(f"cluster_df 形状: {cluster_df.shape}")
    print("特征描述统计：")
    print(cluster_df[feature_cols].describe().T)

    cluster_df.to_csv(OUTPUT_PATH, index=False, encoding=WRITE_ENCODING)
    print(f"已保存 cluster_df 至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
