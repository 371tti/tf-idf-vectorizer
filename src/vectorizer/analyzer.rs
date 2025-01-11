use std::{collections::{HashMap, HashSet}, sync::{atomic::{AtomicU64, Ordering}, Arc, Mutex}};
use std::str;

use fst::{MapBuilder, Streamer};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use vec_plus::vec::{default_sparse_vec::DefaultSparseVec, hi_layer_sparse_vec::ZeroSparseVec};

use super::{index::Index, token::TokenFrequency};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Document {
    pub text: Option<String>,
    pub tokens: TokenFrequency,
}

impl Document {
    pub fn new() -> Self {
        Document {
            text: None,
            tokens: TokenFrequency::new(),
        }
    }

    pub fn new_with_set(text: Option<&str>, tokens: TokenFrequency) -> Self {
        Document {
            text: text.map(|s| s.to_string()),
            tokens,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentAnalyzer<IdType>
where
    IdType: Eq + std::hash::Hash + Clone + Serialize + Send + Sync + std::fmt::Debug,
{
    pub documents: HashMap<IdType, Document>,
    pub idf: TokenFrequency,
    pub total_doc_count: u64,
}

impl<IdType> DocumentAnalyzer<IdType>
where
    IdType: Eq + std::hash::Hash + Clone + Serialize + Send + Sync + std::fmt::Debug,
{

    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            idf: TokenFrequency::new(),
            total_doc_count: 0,
        }
    }

    pub fn add_document(&mut self, id: IdType, content: &[&str], text: Option<&str>) -> Option<&Document>{
        if let Some(document) = self.documents.get_mut(&id) {
            self.idf.sub_tokens_string(&document.tokens.get_token_set());
            document.text = text.map(|s| s.to_string());
            document.tokens.reset();
            document. tokens.add_tokens(content);
            self.idf.add_tokens_string(&document.tokens.get_token_set());
            return self.documents.get(&id);
        } else {
            let mut tokens = TokenFrequency::new();
            tokens.add_tokens(content);
            self.idf.add_tokens_string(&tokens.get_token_set());
            self.documents.insert(id.clone(), Document::new_with_set(text, tokens));
            self.total_doc_count += 1;
            return self.documents.get(&id);
        }
    }

    pub fn get_document(&self, id: &IdType) -> Option<&Document> {
        self.documents.get(id)
    }

    pub fn del_document(&mut self, id: &IdType) -> Option<Document> {
        if let Some(document) = self.documents.remove(id) {
            self.total_doc_count -= 1;
            self.idf
                .sub_tokens_string(&document.tokens.get_token_set());
            Some(document)
        } else {
            None
        }
    }

    pub fn get_document_count(&self) -> u64 {
        self.total_doc_count
    }

    pub fn get_token_set_vec(&self) -> Vec<String> {
        self.idf.get_token_set()
    }

    pub fn get_token_set_vec_ref(&self) -> Vec<&str> {
        self.idf.get_token_set_ref()
    }

    pub fn get_token_set(&self) -> HashSet<String> {
        self.idf.get_token_hashset()
    }

    pub fn get_token_set_ref(&self) -> HashSet<&str> {
        self.idf.get_token_hashset_ref()
    }

    pub fn get_token_set_len(&self) -> usize {
        self.idf.get_token_set_len()
    }

    pub fn generate_index(&self) -> Index<IdType> {
        // 統計の初期化
        let total_doc_tokens_len = Arc::new(AtomicU64::new(0));
        let max_doc_tokens_len = Arc::new(AtomicU64::new(0));
        let now_prosessing = Arc::new(AtomicU64::new(0));
    
        // idf のfst生成
        let mut builder = MapBuilder::memory();
        let mut idf_vec = self.idf.get_idf_vector_ref_parallel(self.total_doc_count);
        idf_vec.sort_by(|a, b| a.0.cmp(b.0));
        for (token, idf) in idf_vec {
            builder.insert(token.as_bytes(), idf as u64).unwrap();
        }
        let idf = Arc::new(builder.into_map());
    
        // 並列処理用のスレッドセーフなIndex
        let index = Arc::new(Mutex::new(HashMap::new()));
    
        // ドキュメントごとの処理を並列化
        self.documents.par_iter().for_each(|(id, document)| {
            now_prosessing.fetch_add(1, Ordering::SeqCst);
            let mut tf_idf_sort_vec: Vec<u16> = Vec::new();
    
            let tf_idf_vec: HashMap<String, u16> =
                document.tokens.get_tfidf_hashmap_fst_parallel(&idf);
    
            let mut stream = idf.stream();
            while let Some((token, _)) = stream.next() {
                let tf_idf = *tf_idf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
                tf_idf_sort_vec.push(tf_idf);
            }
    
            let tf_idf_csvec: DefaultSparseVec<u16> = DefaultSparseVec::from(tf_idf_sort_vec);
            let doc_tokens_len = document.tokens.get_total_token_count();
    
            total_doc_tokens_len.fetch_add(doc_tokens_len, Ordering::SeqCst);
    
            max_doc_tokens_len.fetch_max(doc_tokens_len, Ordering::SeqCst);
    
            let mut index_guard = index.lock().unwrap();
            index_guard.insert(id.clone(), (tf_idf_csvec, doc_tokens_len));
        });
    
        // 統計計算
        let avg_total_doc_tokens_len = (total_doc_tokens_len.load(Ordering::SeqCst)
            / self.total_doc_count as u64) as u64;
        let max_doc_tokens_len = max_doc_tokens_len.load(Ordering::SeqCst);
    
        // indexの返却
        Index::new_with_set(
            Arc::try_unwrap(index).unwrap_or(HashMap::new().into()).into_inner().unwrap(),
            Arc::try_unwrap(idf).unwrap(),
            avg_total_doc_tokens_len,
            max_doc_tokens_len,
            self.total_doc_count,
        )
    }
}