from sentence_transformers import SentenceTransformer
import re
import pandas as pd

def _clean_text(x: str) -> str:
    s = str(x)

    # 이스케이프된 줄바꿈/탭 처리
    s = s.replace("\\n", " ").replace("\\t", " ")

    # 제어문자 제거/치환
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 영어 이외 값 제거
    s = re.sub(r'[^\x00-\x7F]+', '', s)

    # 줄바꿈은 공백으로 합치기(임베딩 입력 안정화)
    s = re.sub(r"\s*\n\s*", " ", s)

    # 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _merge_unique_text(series: pd.Series, sep=" | "):
    vals = [v.strip() for v in series.fillna("").astype(str).tolist() if v and v.strip()]
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sep.join(out)

def get_embeddings(embed_model: str) -> SentenceTransformer:
    embedder = SentenceTransformer(embed_model)
    return embedder

def parse_tags(x: str) -> list[str]:
    s = _clean_text(x)
    if not s:
        return []
    # 콤마 기반 split
    tags = [t.strip() for t in s.split(",") if t.strip()]
    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for t in tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

def build_embed_text(row: pd.Series) -> str:
    name = _clean_text(row.get("Whisky Name", ""))
    tags = parse_tags(row.get("Tags", ""))
    # ns = row.get("Nose Score", "")
    # ts = row.get("Taste Score", "")
    # fs = row.get("Finish Score", "")

    nose = row.get("Nose Comment", "")
    taste = row.get("Taste Comment", "")
    finish = row.get("Finish Comment", "")

    parts = []
    parts.append(f"Whisky: {name}")
    if tags:
        parts.append("Tags: " + ", ".join(tags))

    # 점수의 결측치가 많으므로 해당 데이터는 임베딩 가치가 없다고 판단. 나중에 원할 시 해당 항목 추가
    # parts.append(f"Scores: nose={ns}, taste={ts}, finish={fs}")

    if nose:
        parts.append(f"Nose: {nose}")
    if taste:
        parts.append(f"Taste: {taste}")
    if finish:
        parts.append(f"Finish: {finish}")

    return "\n".join(parts).strip()

def process_embedding_data(df: pd.DataFrame) -> pd.DataFrame:
    LEN_SUM = 200
    comment_cols = ["Nose Comment", "Taste Comment", "Finish Comment"]

    # 1) 결측치 -> "" / 모든 값을 문자열로 통일
    for c in comment_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()    

    df["comment_count"] = sum((df[c].str.len() > 0).astype(int) for c in comment_cols)
    df["comment_len_sum"] = df[comment_cols].apply(lambda r: sum(len(x) for x in r if x), axis=1)
    keep = (df["comment_count"] >= 3) & (df["comment_len_sum"] >= LEN_SUM)

    df = df.loc[keep].copy().drop('comment_len_sum', axis=1)

    cleaned_text_df = df.copy()
    for col in comment_cols:
        cleaned_text_df[col] = cleaned_text_df[col].apply(lambda r: _clean_text(r))

    agg = {}
    for col in cleaned_text_df.columns:
        agg[col] = _merge_unique_text

    merged = cleaned_text_df.groupby("Whisky Name", as_index=False).agg(agg)
    merged["embed_text"] = merged.apply(build_embed_text, axis=1)

    return merged