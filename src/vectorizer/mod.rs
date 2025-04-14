
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use token::TokenFrequency;

pub mod index;
pub mod token;
pub mod analyzer;

pub struct TFIDFVectorizer {
    pub corpus: TokenFrequency,
    doc_num: u64,
}

impl TFIDFVectorizer {
    /// 新しいTF-IDFベクトライザを作成するメソッド
    pub fn new() -> Self {
        Self {
            corpus: TokenFrequency::new(),
            doc_num: 0,
        }
    }

    /// コーパスになったドキュメントの数を取得するメソッド
    pub fn doc_num(&self) -> u64 {
        self.doc_num
    }

    /// コーパスのトークン数を取得するメソッド
    /// トークン数はユニークなトークンの数を返す
    pub fn add_corpus(&mut self, tokens: &[&str]) {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        // corpus_token_freqに追加
        self.corpus.add_tokens(tokens);
 
        self.doc_num += 1;
    }

    /// TF-IDFベクトルを生成するメソッド
    /// 
    /// # Arguments
    /// * `tokens` - ドキュメントのトークン
    /// 
    /// # Returns
    /// * `Vec<(&str, f64)>` - TF-IDFベクトル
    pub fn tf_idf_vector(&self, tokens: &[&str]) -> Vec<(&str, f64)> {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        let mut result: Vec<(&str, f64)> = Vec::new();
        // corpus_token_freqに追加
        let idf_vec: Vec<(&str, f64)> = self.corpus.idf_vector_ref_str(self.doc_num as u64);
        for (added_token, idf) in idf_vec.iter() {
            let tf: f64 = doc_tf.tf_token(added_token);
            if tf != 0.0 {
                let tf_idf = tf * idf;
                result.push((*added_token, tf_idf));
            }
        }
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        result
    }

    /// TF-IDFベクトルを生成するメソッド
    /// /// 並列処理を使用して、検索を高速化します。
    /// 
    /// # Arguments
    /// * `tokens` - ドキュメントのトークン
    /// * `thread_count` - スレッド数
    /// 
    /// # Returns
    /// * `Vec<(&str, f64)>` - TF-IDFベクトル
    pub fn tf_idf_vector_parallel(&self, tokens: &[&str], thread_count: usize) -> Vec<(&str, f64)> {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        let idf_vec: Vec<(&str, f64)> = self.corpus.idf_vector_ref_str(self.doc_num as u64);

        // カスタムスレッドプールを作成し、スレッド数を指定
        let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();
        let mut result: Vec<(&str, f64)> = pool.install(|| {
            idf_vec
                .par_iter()
                .filter_map(|(added_token, idf)| {
                    let tf: f64 = doc_tf.tf_token(added_token);
                    if tf != 0.0 {
                        Some((*added_token, tf * idf))
                    } else {
                        None
                    }
                })
                .collect()
        });
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        result
    }
}