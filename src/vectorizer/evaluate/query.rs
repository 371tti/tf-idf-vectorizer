use crate::{TermFrequency, utils::datastruct::map::{IndexMap, IndexSet}};

#[derive(Clone, Debug)]
pub enum QueryInner {
    None,
    All,
    Nop(Box<str>),
    Not(Box<QueryInner>),
    And(Box<QueryInner>, Box<QueryInner>),
    Or(Box<QueryInner>, Box<QueryInner>),
}

/// Query Structure
///
/// Represents a search query with logical filtering conditions.
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

    pub fn term<S>(term: &S) -> Self 
    where
        S: AsRef<str> + ?Sized,
    {
        Query { inner: QueryInner::Nop(Box::from(term.as_ref())) }
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

    pub fn from_freq_or(freq: &TermFrequency) -> Self {
        let mut iter = freq.term_set_iter();
        if let Some(first_term) = iter.next() {
            let mut query = Query::term(first_term);
            for term in iter {
                let term_query = Query::term(term);
                query = Query::or(query, term_query);
            }
            query
        } else {
            Query::none()
        }
    }

    pub fn from_freq_and(freq: &TermFrequency) -> Self {
        let mut iter = freq.term_set_iter();
        if let Some(first_term) = iter.next() {
            let mut query = Query::term(first_term);
            for term in iter {
                let term_query = Query::term(term);
                query = Query::and(query, term_query);
            }
            query
        } else {
            Query::none()
        }
    }

    pub fn get_all_terms(&self) -> Vec<&str> {
        let mut terms = Vec::new();
        Self::collect_terms_ref(&self.inner, &mut terms);
        terms
    }

    pub(crate) fn collect_terms_ref<'a>(query: &'a QueryInner, terms: &mut Vec<&'a str>) {
        match query {
            QueryInner::All => {
                // do nothing
            }
            QueryInner::None => {}
            QueryInner::Nop(term) => {
                terms.push(term);
            }
            QueryInner::Not(inner) => {
                Self::collect_terms_ref(inner, terms);
            }
            QueryInner::And(left, right) => {
                Self::collect_terms_ref(left, terms);
                Self::collect_terms_ref(right, terms);
            }
            QueryInner::Or(left, right) => {
                Self::collect_terms_ref(left, terms);
                Self::collect_terms_ref(right, terms);
            }
        }
    }

    pub(crate) fn build_ref<K>(query: &QueryInner, term_dim_rev_index: &IndexMap<Box<str>, Vec<u32>>, documents: &IndexSet<K>) -> Vec<usize> 
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
            QueryInner::Nop(term) => {
                if let Some(doc_keys) = term_dim_rev_index.get(term) {
                    let mut result = doc_keys.iter().map(|&id| id as usize).collect::<Vec<usize>>();
                    result.sort_unstable();
                    result
                } else {
                    Vec::new()
                }
            }
            QueryInner::Not(inner) => {
                let inner_indices = Self::build_ref(inner, term_dim_rev_index, documents);
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
                let left_indices = Self::build_ref(left, term_dim_rev_index, documents);
                let right_indices = Self::build_ref(right, term_dim_rev_index, documents);
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
                let left_indices = Self::build_ref(left, term_dim_rev_index, documents);
                let right_indices = Self::build_ref(right, term_dim_rev_index, documents);
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

    pub fn build<K>(&self, term_dim_rev_index: &IndexMap<Box<str>, Vec<u32>>, documents: &IndexSet<K>) -> Vec<usize> 
    where 
        K: Eq + std::hash::Hash,
    {
        let mut res = Self::build_ref(&self.inner, term_dim_rev_index, documents);
        res.sort_unstable();
        res.dedup();
        res
    }
}