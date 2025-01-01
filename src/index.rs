use std::collections::HashMap;
use std::str;

use fst::{Map, Streamer};
use serde::Serialize;
use sprs::CsVec;

use crate::{csvec_trait::FromVec, token::TokenFrequency};



pub struct Index<IdType> 
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    // doc_id -> (圧縮ベクトル, 文書の総トークン数)
    // 圧縮ベクトル: インデックス順にトークンの TF を保持
    pub index: HashMap<IdType, (CsVec<u16>, u64/*token num */)>,
    pub avg_tokens_len: u64,     // 全文書の平均トークン長
    pub max_tokens_len: u64,     // 全文書の最大トークン長
    pub idf: Map<Vec<u8>>,       // fst::Map 形式の IDF
    pub total_doc_count: u64,    // 文書総数
}

impl<IdType> Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    
    pub fn new_with_set(index: HashMap<IdType, (CsVec<u16>, u64/*token num */)>, idf: Map<Vec<u8>>, avg_tokens_len: u64, max_tokens_len: u64, total_doc_count: u64) -> Self {
        Self {
            index,
            idf,
            avg_tokens_len,
            max_tokens_len,
            total_doc_count,
        }
    }

    pub fn get_index(&self) -> &HashMap<IdType, (CsVec<u16>, u64/*token num */)> {
        &self.index
    }

    pub fn search_cos_similarity(&self, query: &[&str], n: usize) -> Vec<(&IdType, f64)> {
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, _idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cos similarityで検索
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, document)| {
            let similarity = Self::cos_similarity(&document.0, &query_csvec);
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    pub fn search_cos_similarity_tuned(&self, query: &[&str], n: usize, b:f64) -> Vec<(&IdType, f64)> {
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, _idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cos similarityで検索
        let max_for_len_norm = self.max_tokens_len as f64 / self.avg_tokens_len as f64;
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, (document, doc_len))| {
            let len_norm = 0.5 + ((((*doc_len as f64 / self.avg_tokens_len as f64) / max_for_len_norm) - 0.5) * b);
            let similarity = Self::cos_similarity(document, &query_csvec) * len_norm;
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    fn cos_similarity(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> f64 {
        // 内積を計算
        let dot_product = Self::dot_product_u16(vec_a, vec_b) as f64;
        
        // ノルム（ベクトルの長さ）を計算
        let norm_a = (Self::dot_product_u16(vec_a, vec_a) as f64).sqrt();
        let norm_b = (Self::dot_product_u16(vec_b, vec_b) as f64).sqrt();
    
        // コサイン類似度を返す
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0 // 少なくとも一方のベクトルがゼロの場合
        }
    }

    fn dot_product_u16(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> u64 {
        let mut result: u64 = 0;
    
        // `CsVec` のインデックスと値をイテレート
        let mut iter_a = vec_a.iter();
        let mut iter_b = vec_b.iter();
    
        let mut a = iter_a.next();
        let mut b = iter_b.next();
    
        while let (Some((index_a, &val_a)), Some((index_b, &val_b))) = (a, b) {
            match index_a.cmp(&index_b) {
                std::cmp::Ordering::Equal => {
                    // 両方のインデックスが一致
                    result = result.saturating_add((val_a as u64) * (val_b as u64));
                    a = iter_a.next();
                    b = iter_b.next();
                }
                std::cmp::Ordering::Less => {
                    // vec_a の次へ進む
                    a = iter_a.next();
                }
                std::cmp::Ordering::Greater => {
                    // vec_b の次へ進む
                    b = iter_b.next();
                }
            }
        }
    
        result
    }
 
    pub fn search_bm25_tfidf(&self, query: &[&str], n: usize, k1: f64, b: f64) -> Vec<(&IdType, f64)> {
        println!("{:?}", query);
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, _idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cos similarityで検索
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, document)| {
            let similarity = Self::bm25_with_csvec_optimized(
                &query_csvec,
                &document.0,
                 document.1, 
                 self.avg_tokens_len as f64, 
                 k1, 
                 b);
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    pub fn bm25_with_csvec_optimized(
        query_vec: &CsVec<u16>, // クエリのTF-IDFベクトル（u16）
        doc_vec: &CsVec<u16>,   // 文書のTF-IDFベクトル（u16）
        doc_len: u64,           // 文書のトークン数
        avg_doc_len: f64,       // 平均文書長
        k1: f64,                // BM25のパラメータ
        b: f64,                 // 文書長補正のパラメータ
    ) -> f64 {
        let mut score = 0.0;
    
        // 文書長補正を計算
        let len_norm = 1.0 - b + b * (doc_len as f64 / avg_doc_len);
    
        // 定数の事前計算
        const MAX_U16_AS_F64: f64 = 1.0 / (u16::MAX as f64); // 1 / 65535.0
        let k1_len_norm = k1 * len_norm;
    
        // クエリと文書のインデックスおよびデータ配列に直接アクセス
        let query_indices = query_vec.indices();
        let query_data = query_vec.data();
        let doc_indices = doc_vec.indices();
        let doc_data = doc_vec.data();
    
        let mut q = 0; // クエリベクトルのインデックス
        let mut d = 0; // 文書ベクトルのインデックス
        let q_len = query_vec.nnz();
        let d_len = doc_vec.nnz();
    
        // クエリと文書のインデックスを走査
        while q < q_len && d < d_len {
            let q_idx = query_indices[q];
            let d_idx = doc_indices[d];
    
            if q_idx == d_idx {
                // クエリと文書の両方に存在するトークン
                let tf_f = (doc_data[d] as f64) * MAX_U16_AS_F64; // 文書内TF-IDF
                let idf_f = (query_data[q] as f64) * MAX_U16_AS_F64; // クエリのTF-IDF（IDF含む）
    
                // BM25のスコア計算
                let numerator = tf_f * (k1 + 1.0);
                let denominator = tf_f + k1_len_norm;
                score += idf_f * (numerator / denominator);
    
                q += 1;
                d += 1;
            } else if q_idx < d_idx {
                // クエリにしか存在しないトークン
                q += 1;
            } else {
                // 文書にしか存在しないトークン
                d += 1;
            }
        }
    
            score
        }
    }