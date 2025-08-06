use num::Num;

pub mod compare;

pub trait Hadamard<N>
where
    N: Num + Copy,
{
    fn hadamard(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>);
}