from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import MeCab

# MeCabのインスタンスを作成
mecab = MeCab.Tagger("-Owakati")

# CSVファイルからデータを読み込む
csv_path = "kdb.csv"
df = pd.read_csv(csv_path)

# 日本語テキストを分かち書きしてタグ付け
tagged_data = [
    TaggedDocument(words=mecab.parse(str(text)).split(), tags=[str(document_id)])
    for document_id, text in zip(df["科目番号"], df["授業概要"])
]

# Doc2Vec モデルの作成と学習
model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("kdb_2023.model")
