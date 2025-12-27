<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>Rust製の 圧倒的柔軟/高速 な文章解析エンジン</p>
<h2>> <a href="https://docs.rs/tf-idf-vectorizer/latest/tf_idf_vectorizer/index.html">docs.rs</a> <</h2>
</div>


 
コーパス構築 → TF 計算 → IDF 計算 → TF-IDF ベクトル化 / 類似度検索 までを一通りサポート。

## 特徴
- ジェネリックパラメータ (f32 / f64 / 符号なし整数など) 対応エンジン 量子化
- Trait ジェネリクス によるTF-IDFエンジンの差し替え可能
- 様々な語彙頻度解析
- Incremental Indexing(インデックスへの追加はO(1)で構築コストなく即時反映されます)
- コーパス情報の分離 indexに対して差し替え可能
- 類似度計算ユーティリティ (`SimilarityAlgorithm`, `Hits`, `Query`) 検索用途
- スレッド間安全
- 復元性 ドキュメントの統計情報を保持
- serde対応

## データモデル
以下のモデルに準じます
### 類似度モデル
- 内積
- コサイン角度
- BM25
### 語彙重みモデル
- tf
- idf
- 上記複合(eg. tf-idf)
### クエリモデル
- 重み付語彙集合
- 論理式
- 上記複合
### 検索モデル
- ブーリアンモデル
- ベクトル空間モデル
- 上記複合

## セットアップ
Cargo.toml
```toml
[dependencies]
tf-idf-vectorizer = "0.9"  # 本READMEは `v0.9.x` 向け
```

## 基本的な使い方

```rust
use std::sync::Arc;

use tf_idf_vectorizer::{Corpus, SimilarityAlgorithm, TFIDFVectorizer, TokenFrequency, vectorizer::evaluate::query::Query};

fn main() {
    // build corpus
    let corpus = Arc::new(Corpus::new());

    // make token frequencies
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "高速", "並列", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "柔軟", "安全", "rust"]);

    // add documents to vectorizer
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(corpus);    
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);
    vectorizer.del_doc(&"doc1".to_string());
    vectorizer.add_doc("doc3".to_string(), &freq1);

    // build query
    let query = Query::and(Query::token("rust"), Query::token("安全"));
    let algorithm = SimilarityAlgorithm::CosineSimilarity;
    let mut result = vectorizer.search(&algorithm, query);
    result.sort_by_score_desc();

    // print result
    println!("Search Results: \n{}", result);
    // debug
    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer);
}
```

## パフォーマンス指針
- トークン辞書 (token_dim_sample / token_dim_set) は再構築を避けキャッシュ
- TF スパース化でゼロ省略
- 整数スケール型 (u16/u32) を使うとメモリ圧縮 (正規化時は 1/max 乗算のみ 演算はfloatのほうが少し高速です)
- 逆Indexを即時生成

実測値でWikipediaJP全記事の全文検索(2.3M docs)の記事の単一クエリ検索において  

`avg: 20ms, min: 0mx, max: 2547ms`
avgはランダムな語1000
minはcorpusの出現最小語
maxはcorpusの出現最大語

[他テスト結果はこちら](doc-search-test.md)

精度はまちまちです。 tokenizeや類似度アルゴリズムに大きく影響を受ける
```
> Rust
Found 465 results in 4 ms.
results:
score: 0.019235 doc_len: 987    key: "4322200_Rust Foundation.json"
score: 0.019130 doc_len: 42644  key: "2609267_Rust (プログラミング言語).json"
score: 0.012565 doc_len: 1508   key: "1679440_ルスト (ブルゲンラント州).json"
score: 0.011875 doc_len: 1037   key: "213405_トキオ.json"
score: 0.009898 doc_len: 39721  key: "4963581_Rust (コンピュータゲーム).json"
score: 0.008443 doc_len: 2807   key: "201762_RS.json"
score: 0.007891 doc_len: 4487   key: "1777419_ベルンハルト・ルスト.json"
score: 0.007572 doc_len: 2792   key: "3579959_シュガー・マウンテン.json"
score: 0.007285 doc_len: 2355   key: "4774208_エジプトへの逃避途上の休息 (ダヴィト、アントワープ).json"
score: 0.007225 doc_len: 2231   key: "3859440_Exa (ソフトウェア).json"
score: 0.006375 doc_len: 2489   key: "4091867_ブライアン・ラスト.json"
score: 0.006247 doc_len: 3047   key: "3892364_The Rust.json"
score: 0.006019 doc_len: 6155   key: "1156673_ラストベルト.json"
score: 0.005941 doc_len: 4352   key: "4475787_ラスト (映画).json"
score: 0.005868 doc_len: 1095   key: "2549609_ラスト.json"
score: 0.005851 doc_len: 2499   key: "2313868_ラスト・マクファーソン・デミング.json"
score: 0.005835 doc_len: 440    key: "702193_カーゴ.json"
score: 0.005705 doc_len: 1426   key: "3127428_ベイビー、ザ・スターズ・シャイン・ブライト.json"
score: 0.005451 doc_len: 556    key: "1683422_ルスト.json"
score: 0.005043 doc_len: 2825   key: "3211357_Servo.json"
score: 0.004967 doc_len: 794    key: "227855_Exa.json"
score: 0.004953 doc_len: 1825   key: "1542667_ラスト・イン・ピース.json"
score: 0.004822 doc_len: 29010  key: "1022_C言語.json"
score: 0.004671 doc_len: 2100   key: "3545028_Redox (オペレーティングシステム).json"
score: 0.004507 doc_len: 1636   key: "1204378_サビキン目.json"
score: 0.003971 doc_len: 3458   key: "4251947_Ruffle.json"
score: 0.003884 doc_len: 4763   key: "993157_Rust Blaster.json"
score: 0.003707 doc_len: 2482   key: "1960205_背信の門.json"
score: 0.003672 doc_len: 2740   key: "3578726_ボトム型.json"
score: 0.003666 doc_len: 4533   key: "4981263_ウェルド_ライブ・イン・ザ・フリー・ワールド.json"
```

環境:
- CPU: 11900K
- RAM: DDR4 dual-ch 2800 48GB
- allocator: mimalloc on windows

## 型概要
| 型 | 役割 |
|----|------|
| Corpus | 文書集合メタ/頻度取得 |
| TokenFrequency | 単一文書内のトークン頻度 |
| TFVector | 1 文書の TF スパースベクトル |
| IDFVector | 全体 IDF とメタ |
| TFIDFVectorizer | TF/IDF 管理と検索入口 |
| TFIDFData | シリアライズ用中間体 |
| DefaultTFIDFEngine | TF/IDF 計算バックエンド |
| SimilarityAlgorithm / Hits / Query | 検索クエリと結果 |

[docs.rs](https://docs.rs/tf-idf-vectorizer/latest/tf_idf_vectorizer/index.html)

## 例 (examples/)
`cargo run --example basic` で最小例を実行。  

# 貢献はプルリクで(。-`ω-)
