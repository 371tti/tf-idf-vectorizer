use num::Zero;

use crate::ZeroSpVec;

enum Query<N: num::Num> {
    Vec(ZeroSpVec<N>),
    And(Box<Query<N>>, Box<Query<N>>),
    Or(Box<Query<N>>, Box<Query<N>>),
    Not(Box<Query<N>>),
}