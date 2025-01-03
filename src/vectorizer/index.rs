use std::collections::HashMap;
use std::str;

use fst::{Map, MapBuilder, Streamer};
use serde::Serialize;
use vec_plus::vec::{sparse_vec::ZeroSparseVec, vec_trait::Math};

use super::token::TokenFrequency;


#[derive(Clone, Debug)]
pub struct Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize + std::fmt::Debug,
{
    // doc_id -> (圧縮ベクトル, 文書の総トークン数)
    // 圧縮ベクトル: インデックス順にトークンの TF を保持
    pub index: HashMap<IdType, (ZeroSparseVec<u16>, u64 /* token num */)>,
    pub avg_tokens_len: u64,  // 全文書の平均トークン長
    pub max_tokens_len: u64,  // 全文書の最大トークン長
    pub idf: Map<Vec<u8>>,    // fst::Map 形式の IDF
    pub total_doc_count: u64, // 文書総数
}

impl<IdType> Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize + std::fmt::Debug,
{
    // ---------------------------------------------------------------------------------------------
    // コンストラクタ
    // ---------------------------------------------------------------------------------------------
    pub fn new_with_set(
        index: HashMap<IdType, (ZeroSparseVec<u16>, u64)>,
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

    pub fn get_index(&self) -> &HashMap<IdType, (ZeroSparseVec<u16>, u64)> {
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

    pub fn synthesize_index(&mut self, mut other: Self /* otherを優先 */) {
        let new_max_token_len = self.max_tokens_len.max(other.max_tokens_len);
        let new_total_doc_count = self.total_doc_count + other.total_doc_count;
        // 加重平均で平均トークン長を再計算
        let sum_self = self.avg_tokens_len as u128 * self.total_doc_count as u128;
        let sum_other = other.avg_tokens_len as u128 * other.total_doc_count as u128;
        let new_avg_token_len =
            ((sum_self + sum_other) / new_total_doc_count as u128) as u64;

        // その他の初期計算
        let this_max_idf = (1.0 + self.total_doc_count as f64 / (2.0)).ln() as f32;
        let other_max_idf = (1.0 + other.total_doc_count as f64 / (2.0)).ln() as f32;
        let combined_max_idf = (1.0 + new_total_doc_count as f64 / (2.0)).ln() as f32;

        //  値の準備
        let mut builder = MapBuilder::memory();
        let mut this_stream = self.idf.stream();
        let mut other_stream = other.idf.stream();
        let mut new_index_index: usize = 0;
        let mut this_new_csvec_index_vec: Vec<usize> = Vec::new();
        let mut other_new_csvec_index_vec: Vec<usize> = Vec::new();

        let mut next_this = this_stream.next();
        let mut next_other = other_stream.next();
        //  両方のidfを合成, csvecのindexを再計算
        while next_this != None && next_other != None {
            let (this_token, this_idf ) = next_this.unwrap();
            let (other_token, other_idf) = next_other.unwrap();
            if this_token < other_token {
                builder.insert(this_token, this_idf).unwrap();
                next_this = this_stream.next();
                    this_new_csvec_index_vec.push(new_index_index);
                // otherのindexはそのまま

            } else if this_token == other_token {
                builder.insert(this_token, Self::synthesize_idf(this_idf, other_idf, 
                    self.total_doc_count, 
                    other.total_doc_count, 
                    new_total_doc_count, 
                    this_max_idf, 
                    other_max_idf, 
                    combined_max_idf
                )).unwrap();
                next_this = this_stream.next();
                    this_new_csvec_index_vec.push(new_index_index);
                next_other = other_stream.next();
                    other_new_csvec_index_vec.push(new_index_index);
            } else {
                builder.insert(other_token, other_idf).unwrap();
                // thisのindexはそのまま

                next_other = other_stream.next();
                    other_new_csvec_index_vec.push(new_index_index);
            }
            new_index_index += 1;
        }
        if next_this != None {
            loop {
                let (this_token, this_idf) = next_this.unwrap();
                builder.insert(this_token, this_idf).unwrap();
                next_this = this_stream.next();
                    this_new_csvec_index_vec.push(new_index_index);
                new_index_index += 1;
                if next_this == None {
                    break;
                }
            }
        } else if next_other != None {
            loop {
                let (other_token, other_idf) = next_other.unwrap();
                builder.insert(other_token, other_idf).unwrap();
                next_other = other_stream.next();
                    other_new_csvec_index_vec.push(new_index_index);
                new_index_index += 1;
                if next_other == None {
                    break;
                }
            }
        }
        let new_idf = builder.into_map();
        println!("{:?}", new_idf);

        //  csvecのindexを合成
        self.index.iter_mut().for_each(|(_id, (csvec, _))| {
            let indices = csvec.sparse_indices_mut();
            for indice in indices {
                *indice = this_new_csvec_index_vec[*indice];
            }
        });

        other.index.iter_mut().for_each(|(_id, (csvec, _))| {
            let indices = csvec.sparse_indices_mut();
            for indice in indices {
                *indice = other_new_csvec_index_vec[*indice];
            }
        });

        //  インデックスの合成
        self.index.extend(other.index);
        self.avg_tokens_len = new_avg_token_len;
        self.max_tokens_len = new_max_token_len;
        self.idf = new_idf;
        self.total_doc_count = new_total_doc_count;
    }

    // ---------------------------------------------------------------------------------------------
    //  プライベート:IDF の合成
    // ---------------------------------------------------------------------------------------------
    #[inline(always)]
    fn synthesize_idf(
        this_idf: u64,
        other_idf: u64,
        this_doc_count: u64,
        other_doc_count: u64,
        total_doc_count: u64,
        this_max_idf: f32,
        other_max_idf: f32,
        combined_max_idf: f32,
    ) -> u64 {
        const MAX_U16: f32 = 65535.0;

        let a = (this_idf as f32 * this_max_idf / MAX_U16).exp();
        let b = (other_idf as f32 * other_max_idf / MAX_U16).exp();

        let denominator = (this_doc_count as f32) / a + (other_doc_count as f32) / b - 2.0;
        let inner = 1.0 + (total_doc_count as f32) / denominator;
        ((inner.ln() / combined_max_idf) * MAX_U16).round() as u64
    }

    // ---------------------------------------------------------------------------------------------
    // BM25 実装 (公開: ほかで呼び出したい場合のみ pub)
    // ---------------------------------------------------------------------------------------------
    pub fn bm25_with_csvec_optimized(
        query_vec: &ZeroSparseVec<u16>, // クエリのTF-IDFベクトル（u16）
        doc_vec: &ZeroSparseVec<u16>,   // 文書のTF-IDFベクトル（u16）
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
        let (query_indices, query_data) = (query_vec.sparse_indices(), query_vec.sparse_values());
        let (doc_indices, doc_data) = (doc_vec.sparse_indices(), doc_vec.sparse_values());

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
    fn build_query_csvec(&self, query: &[&str]) -> ZeroSparseVec<u16> {
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
        ZeroSparseVec::from(sorted_tfidf)
    }

    // ---------------------------------------------------------------------------------------------
    // プライベート: コサイン類似度
    // ---------------------------------------------------------------------------------------------
    fn cos_similarity(vec_a: &ZeroSparseVec<u16>, vec_b: &ZeroSparseVec<u16>) -> f64 {
        // 内積
        let dot_product = vec_a.u64_dot(vec_b) as f64;

        // ノルム（ベクトルの長さ）
        let norm_a = (vec_a.u64_dot(vec_a) as f64).sqrt();
        let norm_b = (vec_b.u64_dot(vec_b) as f64).sqrt();

        // コサイン類似度を返す
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}
