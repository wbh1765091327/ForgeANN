//! Vertex

use std::array::TryFromSliceError;
use std::ops::{Mul, Sub};

use crate::common::Metric;
use crate::utils::calc_distance;

/// Vertex with data type T and dimension N
#[derive(Debug)]
pub struct Vertex<'a, T> {
    /// Vertex value
    val: &'a [T],

    /// Vertex Id
    id: u32,
}

impl<'a, T> Vertex<'a, T>
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Into<f32>,
{
    /// Compare the vertex with another.
    #[inline(always)]
    pub fn compare(&self, other: &Vertex<'a, T>, _metric: Metric) -> f32 {
        // <[T; N]>::distance_compare(self.val, other.val, metric)
        calc_distance(self.val, other.val, self.val.len())
    }
}

impl<'a, T> Vertex<'a, T>
where
    T: Copy,
{
    /// Create the vertex with data
    pub fn new(val: &'a [T], id: u32) -> Self {
        Self { val, id }
    }

    /// Get the vector associated with the vertex.
    #[inline]
    pub fn vector(&self) -> &[T] {
        self.val
    }

    /// Get the vertex id.
    #[inline]
    pub fn vertex_id(&self) -> u32 {
        self.id
    }
}

impl<'a, T> TryFrom<(&'a [T], u32)> for Vertex<'a, T>
where
    T: Copy,
{
    type Error = TryFromSliceError;

    fn try_from((mem_slice, id): (&'a [T], u32)) -> Result<Self, Self::Error> {
        let array: &[T] = mem_slice;
        Ok(Vertex::new(array, id))
    }
}
