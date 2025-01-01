use std::collections::HashMap;
use std::str;

use fst::{Map, MapBuilder, Streamer};
use rayon::vec;
use serde::Serialize;
use sprs::CsVec;

use crate::{csvec_trait::FromVec, token::TokenFrequency};

pub struct Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    // doc_id -> (圧縮ベクトル, 文書の総トークン数)
    // 圧縮ベクトル: インデックス順にトークンの TF を保持
    pub index: HashMap<IdType, (CsVec<u16>, u64 /* token num */)>,
    pub avg_tokens_len: u64,  // 全文書の平均トークン長
    pub max_tokens_len: u64,  // 全文書の最大トークン長
    pub idf: Map<Vec<u8>>,    // fst::Map 形式の IDF
    pub total_doc_count: u64, // 文書総数
}

impl<IdType> Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    // ---------------------------------------------------------------------------------------------
    // コンストラクタ
    // ---------------------------------------------------------------------------------------------
    pub fn new_with_set(
        index: HashMap<IdType, (CsVec<u16>, u64)>,
        idf: Map<Vec<u8>>,
        avg_tokens_len: u64,
        max_tokens_len: u64,
        total_doc_count: u64,
    ) -> Self {
        Self {
            index,
            idf,
            avg_tokens_len,
            max_tokens_len,
            total_doc_count,
        }
    }

    pub fn get_index(&self) -> &HashMap<IdType, (CsVec<u16>, u64)> {
        &self.index
    }

    // ---------------------------------------------------------------------------------------------
    // 公開メソッド: 検索 (Cosine Similarity)
    // ---------------------------------------------------------------------------------------------

    /// 単純なコサイン類似度検索
    pub fn search_cos_similarity(&self, query: &[&str], n: usize) -> Vec<(&IdType, f64)> {
        // クエリの CsVec を作成
        let query_csvec = self.build_query_csvec(query);

        // 類似度スコアを計算
        let mut similarities = self
            .index
            .iter()
            .filter_map(|(id, (doc_vec, _doc_len))| {
                let similarity = Self::cos_similarity(doc_vec, &query_csvec);
                (similarity > 0.0).then(|| (id, similarity))
            })
            .collect::<Vec<_>>();

        // スコア降順でソートして上位 n 件を返す
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities
    }

    /// 文書長を正規化するパラメータを入れたコサイン類似度検索
    pub fn search_cos_similarity_tuned(&self, query: &[&str], n: usize, b: f64) -> Vec<(&IdType, f64)> {
        let query_csvec = self.build_query_csvec(query);

        let max_for_len_norm = self.max_tokens_len as f64 / self.avg_tokens_len as f64;

        let mut similarities = self
            .index
            .iter()
            .filter_map(|(id, (doc_vec, doc_len))| {
                // 0.5 + ( (doc_len / avg_tokens_len) / max_for_len_norm - 0.5 ) * b
                let len_norm = 0.5
                    + (((*doc_len as f64 / self.avg_tokens_len as f64) / max_for_len_norm) - 0.5) * b;

                let similarity = Self::cos_similarity(doc_vec, &query_csvec) * len_norm;
                (similarity > 0.0).then(|| (id, similarity))
            })
            .collect::<Vec<_>>();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities
    }

    // ---------------------------------------------------------------------------------------------
    // 公開メソッド: BM25 x TF-IDF 検索
    // ---------------------------------------------------------------------------------------------

    pub fn search_bm25_tfidf(&self, query: &[&str], n: usize, k1: f64, b: f64) -> Vec<(&IdType, f64)> {
        println!("{:?}", query); // 開発デバッグ用の出力 (必要に応じて削除/ロギングに切り替え推奨)

        let query_csvec = self.build_query_csvec(query);

        let mut similarities = self
            .index
            .iter()
            .filter_map(|(id, (doc_vec, doc_len))| {
                let score = Self::bm25_with_csvec_optimized(
                    &query_csvec,
                    doc_vec,
                    *doc_len,
                    self.avg_tokens_len as f64,
                    k1,
                    b,
                );
                (score > 0.0).then(|| (id, score))
            })
            .collect::<Vec<_>>();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities
    }

    // ---------------------------------------------------------------------------------------------
    // 公開メソッド: Index の合成
    // ---------------------------------------------------------------------------------------------

    pub fn synthesize_index(&self, other: &Self) -> Self {
        let new_max_token_len = self.max_tokens_len.max(other.max_tokens_len);
        let new_avg_token_len = (self.avg_tokens_len + other.avg_tokens_len) / 2;
        let new_total_doc_count = self.total_doc_count + other.total_doc_count;

        //  2つのidfを合成
        let mut builder = MapBuilder::memory();
        let mut this_stream = self.idf.stream();
        let mut other_stream = other.idf.stream();
        let mut do_pop_this = true;
        let mut do_pop_other = true;
        let mut this_skip_index_csvec: Vec<usize> = vec![0];
        let mut other_skip_index_csvec: Vec<usize> = vec![0];
        loop {
            let next = if do_pop_this { this_stream.next() } else { None };
            let next_other = if do_pop_other { other_stream.next() } else { None };
            match (next, next_other) {
                (None, None) => break,
                (None, Some(other)) => {
                    let (other_token, other_idf) = other;
                    builder.insert(other_token, other_idf).unwrap();
                    do_pop_this = false;
                    do_pop_other = true;
                    other_skip_index_csvec.push(other_skip_index_csvec.last().copied().unwrap() + 1);
                },
                (Some(this), None) => {
                    let (this_token, this_idf) = this;
                    builder.insert(this_token, this_idf).unwrap();
                    do_pop_this = true;
                    do_pop_other = false;
                    this_skip_index_csvec.push(this_skip_index_csvec.last().copied().unwrap() + 1);
                },
                (Some(this), Some(other)) => {
                    let (this_token, this_idf) = this;
                    let (other_token, other_idf) = other;
                    if this_token < other_token {
                        builder.insert(this_token, this_idf).unwrap();
                        do_pop_this = true;
                        do_pop_other = false;
                        this_skip_index_csvec.push(this_skip_index_csvec.last().copied().unwrap() + 1);
                        other_skip_index_csvec.first_mut().map(|x| *x + 1);
                    } else if this_token == other_token {
                        builder.insert(this_token, this_idf.max(other_idf)).unwrap();
                        do_pop_this = true;
                        do_pop_other = true;
                        this_skip_index_csvec.push(this_skip_index_csvec.last().copied().unwrap() + 1);
                        other_skip_index_csvec.push(other_skip_index_csvec.last().copied().unwrap() + 1);
                    } else {
                        builder.insert(other_token, other_idf).unwrap();
                        do_pop_this = false;
                        do_pop_other = true;
                        this_skip_index_csvec.first_mut().map(|x| *x + 1);
                        other_skip_index_csvec.push(other_skip_index_csvec.last().copied().unwrap() + 1);
                    }
                },
            }
        }
        let new_idf = builder.into_map();


        let mut new_index = 
    }

    // ---------------------------------------------------------------------------------------------
    // BM25 実装 (公開: ほかで呼び出したい場合のみ pub)
    // ---------------------------------------------------------------------------------------------
    pub fn bm25_with_csvec_optimized(
        query_vec: &CsVec<u16>, // クエリのTF-IDFベクトル（u16）
        doc_vec: &CsVec<u16>,   // 文書のTF-IDFベクトル（u16）
        doc_len: u64,           // 文書のトークン数
        avg_doc_len: f64,       // 平均文書長
        k1: f64,                // BM25のパラメータ
        b: f64,                 // 文書長補正のパラメータ
    ) -> f64 {
        // 文書長補正を計算
        let len_norm = 1.0 - b + b * (doc_len as f64 / avg_doc_len);

        // 定数の事前計算
        const MAX_U16_AS_F64: f64 = 1.0 / (u16::MAX as f64); // 1 / 65535.0
        let k1_len_norm = k1 * len_norm;

        // クエリと文書のインデックスおよびデータ配列を直接取得
        let (query_indices, query_data) = (query_vec.indices(), query_vec.data());
        let (doc_indices, doc_data) = (doc_vec.indices(), doc_vec.data());

        let (mut q, mut d) = (0, 0);
        let (q_len, d_len) = (query_vec.nnz(), doc_vec.nnz());

        let mut score = 0.0;
        while q < q_len && d < d_len {
            let q_idx = query_indices[q];
            let d_idx = doc_indices[d];

            if q_idx == d_idx {
                let tf_f = (doc_data[d] as f64) * MAX_U16_AS_F64;
                let idf_f = (query_data[q] as f64) * MAX_U16_AS_F64;

                let numerator = tf_f * (k1 + 1.0);
                let denominator = tf_f + k1_len_norm;
                score += idf_f * (numerator / denominator);

                q += 1;
                d += 1;
            } else if q_idx < d_idx {
                q += 1;
            } else {
                d += 1;
            }
        }
        score
    }

    // ---------------------------------------------------------------------------------------------
    // プライベート: クエリ（&str のスライス）を CsVec<u16> に変換 (IDF を用いた TF-IDF)
    // ---------------------------------------------------------------------------------------------
    fn build_query_csvec(&self, query: &[&str]) -> CsVec<u16> {
        // 1) クエリトークン頻度を作成
        let mut freq = TokenFrequency::new();
        freq.add_tokens(query);

        // 2) IDF からクエリの TF-IDF (u16) を生成
        let query_tfidf_map: HashMap<String, u16> = freq.get_tfidf_hashmap_fst_parallel(&self.idf);

        // 3) IDF の順序でソートされた Vec<u16> を作る
        let mut sorted_tfidf = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token_bytes, _)) = stream.next() {
            // トークンは bytes -> &str へ変換
            let token_str = str::from_utf8(token_bytes).unwrap_or("");
            let tfidf = query_tfidf_map.get(token_str).copied().unwrap_or(0);
            sorted_tfidf.push(tfidf);
        }

        // 4) CsVec に変換して返す
        CsVec::from_vec(sorted_tfidf)
    }

    // ---------------------------------------------------------------------------------------------
    // プライベート: コサイン類似度
    // ---------------------------------------------------------------------------------------------
    fn cos_similarity(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> f64 {
        // 内積
        let dot_product = Self::dot_product_u16(vec_a, vec_b) as f64;

        // ノルム（ベクトルの長さ）
        let norm_a = (Self::dot_product_u16(vec_a, vec_a) as f64).sqrt();
        let norm_b = (Self::dot_product_u16(vec_b, vec_b) as f64).sqrt();

        // コサイン類似度を返す
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    // ---------------------------------------------------------------------------------------------
    // プライベート: ドット積
    // ---------------------------------------------------------------------------------------------
    fn dot_product_u16(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> u64 {
        let mut result = 0u64;

        let mut iter_a = vec_a.iter();
        let mut iter_b = vec_b.iter();

        let mut a = iter_a.next();
        let mut b = iter_b.next();

        while let (Some((index_a, &val_a)), Some((index_b, &val_b))) = (a, b) {
            match index_a.cmp(&index_b) {
                std::cmp::Ordering::Equal => {
                    result = result.saturating_add((val_a as u64) * (val_b as u64));
                    a = iter_a.next();
                    b = iter_b.next();
                }
                std::cmp::Ordering::Less => {
                    a = iter_a.next();
                }
                std::cmp::Ordering::Greater => {
                    b = iter_b.next();
                }
            }
        }

        result
    }
}
