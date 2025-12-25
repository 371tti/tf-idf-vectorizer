use crate::{TokenFrequency, utils::datastruct::map::{IndexMap, IndexSet}, vectorizer::KeyRc};

pub struct QueryBuilder<'a, K> {
    pub token_dim_rev_index: &'a IndexMap<Box<str>, Vec<KeyRc<K>>>,
    pub documents: &'a IndexSet<KeyRc<K>>,
    pub scan_doc_indices: Vec<usize>,
}

pub enum OrderDocIdx {
    Nop(Box<str>),
    Not(Box<OrderDocIdx>),
    And(Box<OrderDocIdx>, Box<OrderDocIdx>),
    Or(Box<OrderDocIdx>, Box<OrderDocIdx>),
}

pub struct Query {
    pub(crate) doc_indices: Vec<usize>,
    pub(crate) token_freq: TokenFrequency,
}

impl<'a, K> QueryBuilder<'a, K>
where
    K: Clone + Eq + std::hash::Hash,
{
    pub fn new(
        token_dim_rev_index: &'a IndexMap<Box<str>, Vec<KeyRc<K>>>,
        documents: &'a IndexSet<KeyRc<K>>,
    ) -> Self {
        Self {
            token_dim_rev_index,
            documents,
            scan_doc_indices: Vec::new(),
        }
    }

    fn list_of_contains_docs(&self, token: &str) -> Vec<usize> {
        self.token_dim_rev_index.get(token).map(|keys| {
            keys.iter().filter_map(|key| {
                self.documents.get_index(key)
            }).collect::<Vec<usize>>()
        }).unwrap_or_else(Vec::new)
    }

    pub fn token(&self, token: &str) -> OrderDocIdx {
        OrderDocIdx::Nop(Box::from(token))
    }

    pub fn not(&self, order: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::Not(Box::new(order))
    }

    pub fn and(&self, left: OrderDocIdx, right: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::And(Box::new(left), Box::new(right))
    }

    pub fn or(&self, left: OrderDocIdx, right: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::Or(Box::new(left), Box::new(right))
    }
    fn build_ref(&self, order: OrderDocIdx, freq: &mut TokenFrequency) -> Vec<usize> {
        match order {
            OrderDocIdx::Nop(token) => {
                freq.add_token(token.as_ref());
                // token is Box<str>, so we need to look up the doc indices
                let mut indices = self.list_of_contains_docs(&token);
                indices.sort_unstable();
                indices.dedup();
                indices
            }
            OrderDocIdx::Not(inner) => {
                let inner_indices = self.build_ref(*inner, freq);
                let mut result = Vec::new();
                let mut inner_iter = inner_indices.into_iter().peekable();
                let mut next_inner = inner_iter.peek().copied();
                for idx in 0..self.documents.len() {
                    match next_inner {
                        Some(inner_idx) if inner_idx == idx => {
                            inner_iter.next();
                            next_inner = inner_iter.peek().copied();
                        }
                        _ => result.push(idx),
                    }
                }
                result
            }
            OrderDocIdx::And(left, right) => {
                let left_indices = self.build_ref(*left, freq);
                let right_indices = self.build_ref(*right, freq);
                let mut result = Vec::new();
                let mut l = 0;
                let mut r = 0;
                while l < left_indices.len() && r < right_indices.len() {
                    match left_indices[l].cmp(&right_indices[r]) {
                        std::cmp::Ordering::Less => l += 1,
                        std::cmp::Ordering::Greater => r += 1,
                        std::cmp::Ordering::Equal => {
                            result.push(left_indices[l]);
                            l += 1;
                            r += 1;
                        }
                    }
                }
                result
            }
            OrderDocIdx::Or(left, right) => {
                let left_indices = self.build_ref(*left, freq);
                let right_indices = self.build_ref(*right, freq);
                let mut result = Vec::with_capacity(left_indices.len() + right_indices.len());
                let mut l = 0;
                let mut r = 0;
                while l < left_indices.len() && r < right_indices.len() {
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
                while l < left_indices.len() {
                    result.push(left_indices[l]);
                    l += 1;
                }
                while r < right_indices.len() {
                    result.push(right_indices[r]);
                    r += 1;
                }
                result
            }
        }
    }

    pub fn build<F>(self, order: F) -> Query 
    where 
        F: FnOnce(&QueryBuilder<'a, K>) -> OrderDocIdx
    {
        let mut freq = TokenFrequency::new();
        let order = order(&self);
        let doc_indices = self.build_ref(order, &mut freq);
        Query {
            doc_indices,
            token_freq: freq,
        }
    }

    pub fn build_with_freq<F>(self, order: F, freq: TokenFrequency) -> Query 
    where 
        F: FnOnce(&QueryBuilder<'a, K>) -> OrderDocIdx
    {
        let doc_indices = self.build_ref(order(&self), &mut freq.clone()); // this freq is not used
        Query {
            doc_indices,
            token_freq: freq,
        }
    }
    
    pub fn build_with_order(self, order: OrderDocIdx) -> Query 
    where 
        K: Clone + Eq + std::hash::Hash,
    {
        let mut freq = TokenFrequency::new();
        let doc_indices = self.build_ref(order, &mut freq);
        Query {
            doc_indices,
            token_freq: freq,
        }
    }

    pub fn build_with_order_and_freq(self, order: OrderDocIdx, freq: TokenFrequency) -> Query 
    where 
        K: Clone + Eq + std::hash::Hash,
    {
        let doc_indices = self.build_ref(order, &mut freq.clone()); // this freq is not used
        Query {
            doc_indices,
            token_freq: freq,
        }
    }
}

pub mod q {
    use crate::vectorizer::evaluate::query::OrderDocIdx;

    pub fn token(token: &str) -> OrderDocIdx {
        OrderDocIdx::Nop(Box::from(token))
    }

    pub fn not(order: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::Not(Box::new(order))
    }

    pub fn and(left: OrderDocIdx, right: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: OrderDocIdx, right: OrderDocIdx) -> OrderDocIdx {
        OrderDocIdx::Or(Box::new(left), Box::new(right))
    }
}