from typing import Final
import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

# 保存したモデルを読み込む
model: Final[Doc2Vec] = Doc2Vec.load("kdb_2023.model")

# シラバスデータを読み込む
csv_path: Final[str] = "./source/kdb.csv"
df: pd.DataFrame = pd.read_csv(csv_path)

st.title("KdB-Doc2Vec")
st.markdown(
    "筑波大学のシラバスデータ（KdB）を用いて、科目概要を [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) でベクトル化し、類似した科目を表示します。[kdb-parse](https://github.com/Mimori256/kdb-parse) の科目データを使用しています。科目データは2024年1月7日時点のものです。"
)

st.header("科目番号を入力")
st.markdown(
    "入力された科目番号の科目概要と類似した科目を表示します。科目番号を [KdB](https://kdb.tsukuba.ac.jp/) で検索する。"
)
document_id: Final[str] = st.text_input("科目番号", "GC22201")

course: pd.Series = df[df["科目番号"] == document_id].iloc[0]
with st.container():
    st.subheader(course["科目名"])
    st.markdown(
        "[KdB シラバス](https://kdb.tsukuba.ac.jp/syllabi/2023/{})。".format(document_id)
    )
    st.table(course[["科目番号", "授業方法", "単位数", "標準履修年次", "実施学期", "曜時限", "教室", "担当教員"]])


st.header("似た科目")
st.write("類似した科目を降順に表示しています。")

similar_documents: Final[list[tuple[str, float]]] = model.dv.most_similar(document_id)

for similar_doc_id, similarity in similar_documents:
    course: pd.Series = df[df["科目番号"] == similar_doc_id].iloc[0]
    with st.container():
        st.subheader(course["科目名"])
        st.markdown(
            "類似度は {:.3f} です。また、[KdB シラバス](https://kdb.tsukuba.ac.jp/syllabi/2023/{})。".format(
                similarity, similar_doc_id
            )
        )
        st.table(course[["科目番号", "授業方法", "単位数", "標準履修年次", "実施学期", "曜時限", "教室", "担当教員"]])
