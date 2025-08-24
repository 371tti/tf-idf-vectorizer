
use num::Num;

pub mod compare;

pub trait ComputeOp<N> 
where N: Num {
    fn batch_dot(&mut self);
    fn batch_hadamard(&mut self);
    fn batch_scalar(&mut self);
    fn batch_div(&mut self);
}

pub struct ComputePool<N>
where N: Num {
    pub pool: Vec<ComputeTile<N>>,
}

pub struct ComputeBlock<N> 
where N: Num {
    pub size: Vec<ComputeTile<N>>,
}

pub struct ComputeTile<N> 
where N: Num {
    pub tile_a: [N; 128],
    pub tile_b: [N; 128],
} 
pub trait ComputeOp<N> 
where N: Num {
    fn batch_dot(&mut self);
    fn batch_hadamard(&mut self);
    fn batch_scalar(&mut self);
    fn batch_div(&mut self);
}

pub struct ComputePool<N>
where N: Num {
    pub pool: Vec<ComputeTile<N>>,
}

pub struct ComputeBlock<N> 
where N: Num {
    pub size: Vec<ComputeTile<N>>,
}

pub struct ComputeTile<N> 
where N: Num {
    pub tile_a: [N; 128],
    pub tile_b: [N; 128],
} 