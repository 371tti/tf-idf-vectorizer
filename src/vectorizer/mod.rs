
use token::TokenFrequency;

pub mod index;
pub mod token;
pub mod analyzer;

pub struct TFIDFVectorizer {
    pub corpus: TokenFrequency,
    doc_num: u64,
}

impl TFIDFVectorizer {
    pub fn new() -> Self {
        Self {
            corpus: TokenFrequency::new(),
            doc_num: 0,
        }
    }

    pub fn doc_num(&self) -> u64 {
        self.doc_num
    }

    pub fn add_corpus(&mut self, tokens: &[&str]) {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        // corpus_token_freqに追加
        self.corpus.add_tokens(tokens);
 
        self.doc_num += 1;
    }

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
}