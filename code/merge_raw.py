import pandas as pd
from pathlib import Path

MERGE_KEY = "varID"
DATA_SUBDIR = Path("\u519c\u6237")
FILE_NUMBERS = [3, 5, 6, 7, 8, 9, 10, 11, 17, 20]
OUTPUT_FILENAME = "raw_df.csv"
READ_ENCODING = "utf-8-sig"
WRITE_ENCODING = "utf-8-sig"
raw_df = None


def first_valid(series: pd.Series):
    non_null = series.dropna()
    if not non_null.empty:
        return non_null.iloc[0]
    return pd.NA


def deduplicate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"\u68c0\u6d4b\u5230\u91cd\u590d\u5217 {duplicate_cols}\uff0c\u4ec5\u4fdd\u7559\u7b2c\u4e00\u6b21\u51fa\u73b0\u3002")
        df = df.loc[:, ~df.columns.duplicated()]
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows:
        print(f"\u68c0\u6d4b\u5230 {duplicate_rows} \u884c\u5b8c\u5168\u76f8\u540c\uff0c\u5c06\u88ab\u5220\u9664\u3002")
        df = df.drop_duplicates()
    duplicate_key_mask = df.duplicated(subset=MERGE_KEY, keep=False)
    if duplicate_key_mask.any():
        duplicate_key_rows = duplicate_key_mask.sum()
        household_count = df.loc[duplicate_key_mask, MERGE_KEY].nunique()
        print(
            f"{duplicate_key_rows} \u884c {MERGE_KEY} \u503c\u91cd\u590d\uff0c\u6d89\u53ca {household_count} \u6237\uff0c"
            "\u6bcf\u5217\u8bb0\u5f55\u53d6\u7b2c\u4e00\u4e2a\u975e\u7a7a\u503c\u540e\u8054\u5408\u3002"
        )
        df = (
            df.sort_values(MERGE_KEY)
            .groupby(MERGE_KEY, as_index=False)
            .agg(first_valid)
        )
    else:
        df = df.sort_values(MERGE_KEY).reset_index(drop=True)
    return df


def resolve_data_dir(script_dir: Path) -> Path:
    candidate = script_dir.parent / DATA_SUBDIR
    if candidate.exists():
        return candidate
    alternates = []
    for folder in script_dir.parent.iterdir():
        if folder.is_dir() and folder.name != script_dir.name:
            if any((folder / f"hh_{number}.csv").exists() for number in FILE_NUMBERS):
                alternates.append(folder)
    if alternates:
        fallback = alternates[0]
        print(f"\u672a\u627e\u5230 {candidate} \u76ee\u5f55\uff0c\u8bd5\u7528 {fallback}")
        return fallback
    raise FileNotFoundError(f"\u65e0\u6cd5\u627e\u5230\u4efb\u4f55\u542b hh_XX.csv \u7684\u6587\u4ef6\u5939\uff0c\u8bf7\u68c0\u67e5\u8def\u5f84\u3002")


def read_single_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding=READ_ENCODING)
    print(f"{file_path.name} \u884c\u5217: {df.shape}")
    if MERGE_KEY not in df.columns:
        print(f"{file_path.name} \u5217\u540d: {list(df.columns)}")
        raise KeyError(f"{file_path.name} \u7f3a\u5c11 {MERGE_KEY}")
    return df


def load_dataframes(data_dir: Path):
    dataframes = []
    file_names = []
    for number in FILE_NUMBERS:
        file_name = f"hh_{number}.csv"
        file_path = data_dir / file_name
        if not file_path.exists():
            chunk_paths = sorted(data_dir.glob(f"hh_{number}_*.csv"))
            if not chunk_paths:
                raise FileNotFoundError(f"{file_path} \u4e0d\u5b58\u5728\uff0c\u8bf7\u68c0\u67e5\u8def\u5f84\u3002")
            merged_chunk = None
            for chunk_path in chunk_paths:
                chunk_df = read_single_csv(chunk_path)
                if merged_chunk is None:
                    merged_chunk = chunk_df
                else:
                    merged_chunk = merged_chunk.merge(chunk_df, on=MERGE_KEY, how="outer")
            dataframes.append(merged_chunk)
            file_names.append(f"hh_{number}_*.csv")
            print(f"hh_{number}_*.csv \u5408\u5e76\u6563\u5e03\u5b8f\u540e\u884c\u5217: {merged_chunk.shape}")
            continue
        df = read_single_csv(file_path)
        dataframes.append(df)
        file_names.append(file_name)
    return file_names, dataframes


def main():
    global raw_df
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(script_dir)
    file_names, dataframes = load_dataframes(data_dir)
    raw_df = dataframes[0]
    print(f"\u521d\u59cb\u5316 {file_names[0]} \u540e raw_df \u884c\u5217: {raw_df.shape}")
    for file_name, df in zip(file_names[1:], dataframes[1:]):
        print(f"\u5408\u5e76 {file_name} \u524d raw_df \u884c\u5217: {raw_df.shape}")
        raw_df = raw_df.merge(df, on=MERGE_KEY, how="outer")
        print(f"\u5408\u5e76 {file_name} \u540e raw_df \u884c\u5217: {raw_df.shape}")
    raw_df = deduplicate_dataframe(raw_df)
    print(f"\u53bb\u91cd\u540e raw_df \u5f62\u72b6: {raw_df.shape}")
    output_path = script_dir / OUTPUT_FILENAME
    raw_df.to_csv(output_path, index=False, encoding=WRITE_ENCODING)
    print("raw_df head:")
    print(raw_df.head())
    print(f"raw_df \u5f62\u72b6: {raw_df.shape}")
    print(f"raw_df \u5217\u540d: {list(raw_df.columns)}")
    print(f"\u5df2\u5c06 raw_df \u4fdd\u5b58\u81f3 {output_path}")


if __name__ == "__main__":
    main()
