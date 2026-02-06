<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>Rust製の 圧倒的柔軟/高速 な文章解析エンジン</p>
</div>

<h2>> <a href="https://docs.rs/tf-idf-vectorizer/latest/tf_idf_vectorizer/index.html">docs.rs</a> <</h2>
 
コーパス構築 → TF 計算 → IDF 計算 → TF-IDF ベクトル化 / 類似度検索 までを一通りサポート。

## 特徴
- ジェネリックパラメータ (f16 / f32 / u16 / u32) 対応エンジン 量子化
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
tf-idf-vectorizer = "0.10"  # 本READMEは `v0.10.x` 向け
```

## 基本的な使い方
以下に使用できます
- 単語頻度解析
- 文書検索
- 類似度比較

上記を文書ごと、もしくは文書集合に対して様々なオプションで扱えます

eg. cos 類似度 論理クエリ検索
```rust
use std::sync::Arc;

use half::f16;
use tf_idf_vectorizer::{Corpus, SimilarityAlgorithm, TFIDFVectorizer, TermFrequency, vectorizer::evaluate::query::Query};

fn main() {
    // build corpus
    let corpus = Arc::new(Corpus::new());

    // make term frequencies
    let mut freq1 = TermFrequency::new();
    freq1.add_terms(&["rust", "高速", "並列", "rust"]);
    let mut freq2 = TermFrequency::new();
    freq2.add_terms(&["rust", "柔軟", "安全", "rust"]);

    // add documents to vectorizer
    let mut vectorizer: TFIDFVectorizer<f16> = TFIDFVectorizer::new(corpus);    
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);
    vectorizer.del_doc(&"doc1".to_string());
    vectorizer.add_doc("doc3".to_string(), &freq1);

    let query = Query::and(Query::term("rust"), Query::term("安全"));
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

```
 cargo run --example basic                                                                                                                                                                                                                                                                                                                                                                                                                               
   Compiling tf-idf-vectorizer v0.10.0 (I:\RustBuilds\tf-idf-vectorizer)                                                                                                                                                                                                                                                                                                                                                                                  
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.60s                                                                                                                                                                                                                                                                                                                                                                                   
     Running `target\debug\examples\basic.exe`
Search Results: 
score: 0.768706 doc_len: 4      key: "doc2"

result count: 1
... debug略
```

## パフォーマンス指針
- トークン辞書 (term_dim_sample / term_dim_set) は再構築を避けキャッシュ
- TF スパース化でゼロ省略
- 整数スケール型 (f16/u16) を使うとメモリ圧縮 (f16はsqrtで圧縮 u16は大規模文書でダメになる場合あり)
- 逆Indexを即時生成

実測値でWikipediaJP全記事の全文検索(2.3M docs)の記事の単一クエリ検索において  

`avg: 20ms, min: 0mx, max: 2547ms`
avgはランダムな語1000
minはcorpusの出現最小語
maxはcorpusの出現最大語

[他テスト結果はこちら](doc-search-test.md)

精度はまちまちです。 termizeや類似度アルゴリズムに大きく影響を受ける  
f16 での検索精度 cosine類似度 単クエリ
```
> Rust
Found 465 results in 11 ms.
results:
score: 0.019971 doc_len: 42644  key: "2609267_Rust (プログラミング言語).json"
score: 0.017489 doc_len: 987    key: "4322200_Rust Foundation.json"
score: 0.010983 doc_len: 39721  key: "4963581_Rust (コンピュータゲーム).json"
score: 0.010525 doc_len: 1508   key: "1679440_ルスト (ブルゲンラント州).json"
score: 0.008609 doc_len: 2231   key: "3859440_Exa (ソフトウェア).json"
score: 0.008346 doc_len: 2355   key: "4774208_エジプトへの逃避途上の休息 (ダヴィト、アントワープ).json"
score: 0.008109 doc_len: 4487   key: "1777419_ベルンハルト・ルスト.json"
score: 0.007618 doc_len: 2792   key: "3579959_シュガー・マウンテン.json"
score: 0.007241 doc_len: 6155   key: "1156673_ラストベルト.json"
score: 0.007205 doc_len: 3047   key: "3892364_The Rust.json"
score: 0.005957 doc_len: 2499   key: "2313868_ラスト・マクファーソン・デミング.json"
score: 0.005943 doc_len: 1037   key: "213405_トキオ.json"
score: 0.005915 doc_len: 2489   key: "4091867_ブライアン・ラスト.json"
score: 0.005660 doc_len: 29010  key: "1022_C言語.json"
score: 0.005601 doc_len: 1636   key: "1204378_サビキン目.json"
score: 0.005541 doc_len: 2825   key: "3211357_Servo.json"
score: 0.005285 doc_len: 4352   key: "4475787_ラスト (映画).json"
score: 0.005134 doc_len: 2740   key: "3578726_ボトム型.json"
score: 0.005003 doc_len: 1825   key: "1542667_ラスト・イン・ピース.json"
score: 0.004746 doc_len: 4763   key: "993157_Rust Blaster.json"
score: 0.004674 doc_len: 2100   key: "3545028_Redox (オペレーティングシステム).json"
score: 0.004288 doc_len: 2807   key: "201762_RS.json"
score: 0.004100 doc_len: 4374   key: "3820827_システムプログラミング言語.json"
score: 0.004089 doc_len: 3458   key: "4251947_Ruffle.json"
score: 0.003802 doc_len: 16508  key: "4758112_エイント・シー・スウィート.json"
score: 0.003690 doc_len: 4533   key: "4981263_ウェルド_ライブ・イン・ザ・フリー・ワールド.json"
score: 0.003642 doc_len: 6945   key: "3822619_ルスト (バーデン).json"
score: 0.003608 doc_len: 7826   key: "782218_プログラミング言語の比較.json"
score: 0.003602 doc_len: 12674  key: "736037_クロージャ.json"
score: 0.003570 doc_len: 3801   key: "3088_Linuxカーネル.json"
```

環境:
- CPU: 11900K
- RAM: DDR4 dual-ch 2800 48GB
- allocator: mimalloc on windows

## 型概要
| 型 | 役割 |
|----|------|
| Corpus | 文書集合メタ/頻度取得 |
| TermFrequency | 単一文書内のトークン頻度 |
| TFVector | 1 文書の TF スパースベクトル |
| IDFVector | 全体 IDF とメタ |
| TFIDFVectorizer | TF/IDF 管理と検索入口 |
| TFIDFData | シリアライズ用中間体 |
| DefaultTFIDFEngine | TF/IDF 計算バックエンド |
| SimilarityAlgorithm / Hits / Query | 検索クエリと結果 |

[docs.rs](https://docs.rs/tf-idf-vectorizer/latest/tf_idf_vectorizer/index.html)

## 例 (examples/)
`cargo run --example basic` で最小例を実行。  

# Todo
- [x] 十分なパフォーマンス最適化
- [x] Rcの削除
- [ ] Reverse Indexのメモリ展開でページアウトできるように

# 貢献はプルリクで(。-`ω-)
