use crate::{TokenFrequency, utils::datastruct::map::{IndexMap, IndexSet}, vectorizer::KeyRc};

#[derive(Clone, Debug)]
pub enum QueryInner {
    None,
    All,
    Nop(Box<str>),
    Not(Box<QueryInner>),
    And(Box<QueryInner>, Box<QueryInner>),
    Or(Box<QueryInner>, Box<QueryInner>),
}

#[derive(Clone, Debug)]
pub struct Query {
    pub(crate) inner: QueryInner,
}

impl Query {
    pub fn none() -> Self {
        Query { inner: QueryInner::None }
    }

    pub fn all() -> Self {
        Query { inner: QueryInner::All }
    }

    pub fn token<S>(token: &S) -> Self 
    where
        S: AsRef<str> + ?Sized,
    {
        Query { inner: QueryInner::Nop(Box::from(token.as_ref())) }
    }

    pub fn not(order: Query) -> Self {
        Query { inner: QueryInner::Not(Box::new(order.inner)) }
    }

    pub fn and(left: Query, right: Query) -> Self {
        Query { inner: QueryInner::And(Box::new(left.inner), Box::new(right.inner)) }
    }

    pub fn or(left: Query, right: Query) -> Self {
        Query { inner: QueryInner::Or(Box::new(left.inner), Box::new(right.inner)) }
    }

    pub fn from_freq_or(freq: &TokenFrequency) -> Self {
        let mut iter = freq.token_set_iter();
        if let Some(first_token) = iter.next() {
            let mut query = Query::token(first_token);
            for token in iter {
                let token_query = Query::token(token);
                query = Query::or(query, token_query);
            }
            query
        } else {
            Query::none()
        }
    }

    pub fn from_freq_and(freq: &TokenFrequency) -> Self {
        let mut iter = freq.token_set_iter();
        if let Some(first_token) = iter.next() {
            let mut query = Query::token(first_token);
            for token in iter {
                let token_query = Query::token(token);
                query = Query::and(query, token_query);
            }
            query
        } else {
            Query::none()
        }
    }

    pub fn get_all_tokens(&self) -> Vec<&str> {
        let mut tokens = Vec::new();
        Self::collect_tokens_ref(&self.inner, &mut tokens);
        tokens
    }

    pub(crate) fn collect_tokens_ref<'a>(query: &'a QueryInner, tokens: &mut Vec<&'a str>) {
        match query {
            QueryInner::All => {
                // do nothing
            }
            QueryInner::None => {}
            QueryInner::Nop(token) => {
                tokens.push(token);
            }
            QueryInner::Not(inner) => {
                Self::collect_tokens_ref(inner, tokens);
            }
            QueryInner::And(left, right) => {
                Self::collect_tokens_ref(left, tokens);
                Self::collect_tokens_ref(right, tokens);
            }
            QueryInner::Or(left, right) => {
                Self::collect_tokens_ref(left, tokens);
                Self::collect_tokens_ref(right, tokens);
            }
        }
    }

    pub(crate) fn build_ref<K>(query: &QueryInner, token_dim_rev_index: &IndexMap<Box<str>, Vec<KeyRc<K>>>, documents: &IndexSet<KeyRc<K>>) -> Vec<usize> 
    where 
        K: Eq + std::hash::Hash,
    {
        match query {
            QueryInner::All => {
                let mut result = Vec::with_capacity(documents.len());
                for (idx, _) in documents.iter().enumerate() {
                    result.push(idx);
                }
                result
            }
            QueryInner::None => Vec::new(),
            QueryInner::Nop(token) => {
                if let Some(doc_keys) = token_dim_rev_index.get(token) {
                    let mut result = Vec::with_capacity(doc_keys.len());
                    for doc_key in doc_keys {
                        if let Some(idx) = documents.get_index(doc_key) {
                            result.push(idx);
                        }
                    }
                    result.sort_unstable();
                    result
                } else {
                    Vec::new()
                }
            }
            QueryInner::Not(inner) => {
                let inner_indices = Self::build_ref(inner, token_dim_rev_index, documents);
                let mut result = Vec::with_capacity(documents.len() - inner_indices.len());
                let mut inner_iter = inner_indices.iter().peekable();
                for (idx, _) in documents.iter().enumerate() {
                    match inner_iter.peek() {
                        Some(&&inner_idx) if inner_idx == idx => {
                            inner_iter.next();
                        }
                        _ => {
                            result.push(idx);
                        }
                    }
                }
                result
            }
            QueryInner::And(left, right) => {
                let left_indices = Self::build_ref(left, token_dim_rev_index, documents);
                let right_indices = Self::build_ref(right, token_dim_rev_index, documents);
                let mut result = Vec::with_capacity(std::cmp::min(left_indices.len(), right_indices.len()));
                let mut l = 0;
                let mut r = 0;
                while l < left_indices.len() && r < right_indices.len() {
                    match left_indices[l].cmp(&right_indices[r]) {
                        std::cmp::Ordering::Less => {
                            l += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            r += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            result.push(left_indices[l]);
                            l += 1;
                            r += 1;
                        }
                    }
                }
                result
            }
            QueryInner::Or(left, right) => {
                let left_indices = Self::build_ref(left, token_dim_rev_index, documents);
                let right_indices = Self::build_ref(right, token_dim_rev_index, documents);
                let mut result = Vec::with_capacity(left_indices.len() + right_indices.len());
                let mut l = 0;
                let mut r = 0;
                while l < left_indices.len() || r < right_indices.len() {
                    if l >= left_indices.len() {
                        result.push(right_indices[r]);
                        r += 1;
                    } else if r >= right_indices.len() {
                        result.push(left_indices[l]);
                        l += 1;
                    } else {
                        match left_indices[l].cmp(&right_indices[r]) {
                            std::cmp::Ordering::Less => {
                                result.push(left_indices[l]);
                                l += 1;
                            }
                            std::cmp::Ordering::Greater => {
                                result.push(right_indices[r]);
                                r += 1;
                            }
                            std::cmp::Ordering::Equal => {
                                result.push(left_indices[l]);
                                l += 1;
                                r += 1;
                            }
                        }
                    }
                }
                result
            }
        }
    }

    pub fn build<K>(&self, token_dim_rev_index: &IndexMap<Box<str>, Vec<KeyRc<K>>>, documents: &IndexSet<KeyRc<K>>) -> Vec<usize> 
    where 
        K: Eq + std::hash::Hash,
    {
        let mut res = Self::build_ref(&self.inner, token_dim_rev_index, documents);
        res.sort_unstable();
        res.dedup();
        res
    }
}