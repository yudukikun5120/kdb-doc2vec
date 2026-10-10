# kdb-doc2vec

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kdb-doc2vec.streamlit.app)

筑波大学シラバスデータを用いて科目概要を Doc2Vec でベクトル化し、類似した科目を表示する。

## シラバスデータ
[kdb-parse](https://github.com/Mimori256/kdb-parse) の科目データを使用しています。

## 実行
依存関係は [uv](https://docs.astral.sh/uv/) で管理しています。

```sh
uv sync
uv run streamlit run app.py
```

## 貢献
貢献は歓迎します。
