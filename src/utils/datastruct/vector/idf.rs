use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IDFVector
{
    /// IDF Vector it is not sparse because it is mostly filled
    pub idf_vec: Vec<f32>,
    /// latest entropy
    pub latest_entropy: u64,
    /// document count
    pub doc_num: u64,
}

impl IDFVector
{
    pub fn new() -> Self {
        Self {
            idf_vec: Vec::new(),
            latest_entropy: 0,
            doc_num: 0,
        }
    }
}