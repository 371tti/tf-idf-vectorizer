# indexについて

文章ベクトルを保持します。
SparceVecの配列でつまり行列になるということで。

## 構造
```rust
struct DocElement {
    vector: ZeroSpVec,
    token_count: u64,
    name: String,
}

struct Index {
    elements: Vec<DocElement>
    corpus_dim: usize,
    corpus_hash: u64,
}
```

AoS構造とする(同時アクセスする場合が大半なので)

## 制約
- indexをcorpusと比較処理する場合はcorpusの次元数が`corpus_dim`以上
- 比較処理に使用するcorpusの`corpus_dim`までの内容のhashが`corpus_hash`と一致する