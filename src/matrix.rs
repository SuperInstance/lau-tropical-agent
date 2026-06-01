//! Tropical matrix multiplication (max-plus linear algebra).

use crate::semiring::Tropical;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A matrix over the tropical semiring (max-plus algebra).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TropicalMatrix {
    /// Row-major storage. NEG_INF represents the tropical zero.
    data: Vec<Vec<Tropical>>,
    nrows: usize,
    ncols: usize,
}

impl TropicalMatrix {
    /// Create a new tropical matrix from raw data.
    pub fn new(data: Vec<Vec<Tropical>>) -> Self {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };
        TropicalMatrix { data, nrows, ncols }
    }

    /// Create a matrix filled with -∞ (tropical zero).
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let data = vec![vec![Tropical::NEG_INF; ncols]; nrows];
        TropicalMatrix { data, nrows, ncols }
    }

    /// Create an identity matrix (0 on diagonal, -∞ elsewhere).
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i][i] = Tropical::ONE;
        }
        m
    }

    /// Create from a regular f64 matrix (finite values only).
    pub fn from_f64(data: &[Vec<f64>]) -> Self {
        Self::new(data.iter().map(|row| row.iter().map(|&v| Tropical(v)).collect()).collect())
    }

    pub fn nrows(&self) -> usize { self.nrows }
    pub fn ncols(&self) -> usize { self.ncols }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> Tropical {
        self.data[i][j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: Tropical) {
        self.data[i][j] = val;
    }

    /// Tropical matrix multiplication: C[i][j] = max_k (A[i][k] + B[k][j]).
    pub fn mul(&self, other: &TropicalMatrix) -> TropicalMatrix {
        assert_eq!(self.ncols, other.nrows, "Matrix dimensions mismatch for tropical multiply");
        let mut result = Self::zeros(self.nrows, other.ncols);
        for i in 0..self.nrows {
            for j in 0..other.ncols {
                let mut best = Tropical::NEG_INF;
                for k in 0..self.ncols {
                    best = best + self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = best;
            }
        }
        result
    }

    /// Tropical matrix power: A^n (repeated tropical multiplication).
    pub fn pow(&self, n: u32) -> TropicalMatrix {
        assert_eq!(self.nrows, self.ncols, "Matrix must be square for pow");
        if n == 0 {
            return Self::identity(self.nrows);
        }
        if n == 1 {
            return self.clone();
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = self.mul(&result);
        }
        result
    }

    /// Tropical matrix addition (element-wise max).
    pub fn add(&self, other: &TropicalMatrix) -> TropicalMatrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);
        let mut result = Self::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    /// Kleene star: A* = I ⊕ A ⊕ A² ⊕ ... ⊕ A^(n-1)
    /// Computes the closure for shortest paths.
    pub fn kleene_star(&self) -> TropicalMatrix {
        assert_eq!(self.nrows, self.ncols, "Matrix must be square for Kleene star");
        let n = self.nrows;
        let mut result = Self::identity(n);
        let mut power = self.clone();
        for _ in 0..n {
            result = result.add(&power);
            power = self.mul(&power);
        }
        result
    }

    /// Convert to nalgebra DMatrix for interop.
    pub fn to_nalgebra(&self) -> DMatrix<f64> {
        let mut vals = Vec::with_capacity(self.nrows * self.ncols);
        for col in 0..self.ncols {
            for row in 0..self.nrows {
                vals.push(self.data[row][col].0);
            }
        }
        DMatrix::from_vec(self.nrows, self.ncols, vals)
    }

    /// Trace (tropical): max of diagonal elements.
    pub fn trace(&self) -> Tropical {
        let mut best = Tropical::NEG_INF;
        for i in 0..self.nrows.min(self.ncols) {
            best = best + self.data[i][i]; // max
        }
        best
    }

    /// Apply to a vector (tropical matrix-vector product).
    pub fn apply_vec(&self, v: &[Tropical]) -> Vec<Tropical> {
        assert_eq!(self.ncols, v.len());
        let mut result = vec![Tropical::NEG_INF; self.nrows];
        for (i, row) in self.data.iter().enumerate() {
            for (j, item) in row.iter().enumerate() {
                result[i] = result[i] + *item * v[j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mul() {
        let a = TropicalMatrix::from_f64(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let eye = TropicalMatrix::identity(2);
        let result = a.mul(&eye);
        assert_eq!(result.get(0, 0), Tropical(1.0));
        assert_eq!(result.get(0, 1), Tropical(2.0));
        assert_eq!(result.get(1, 0), Tropical(3.0));
        assert_eq!(result.get(1, 1), Tropical(4.0));
    }

    #[test]
    fn test_matrix_multiply() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C[0][0] = max(1+5, 2+7) = max(6, 9) = 9
        // C[0][1] = max(1+6, 2+8) = max(7, 10) = 10
        // C[1][0] = max(3+5, 4+7) = max(8, 11) = 11
        // C[1][1] = max(3+6, 4+8) = max(9, 12) = 12
        let a = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = TropicalMatrix::from_f64(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), Tropical(9.0));
        assert_eq!(c.get(0, 1), Tropical(10.0));
        assert_eq!(c.get(1, 0), Tropical(11.0));
        assert_eq!(c.get(1, 1), Tropical(12.0));
    }

    #[test]
    fn test_matrix_pow() {
        let a = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let a2 = a.pow(2);
        let manual = a.mul(&a);
        assert_eq!(a2.get(0, 0), manual.get(0, 0));
        assert_eq!(a2.get(1, 1), manual.get(1, 1));
    }

    #[test]
    fn test_matrix_add() {
        let a = TropicalMatrix::from_f64(&[vec![1.0, 5.0]]);
        let b = TropicalMatrix::from_f64(&[vec![3.0, 2.0]]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), Tropical(3.0)); // max(1,3)
        assert_eq!(c.get(0, 1), Tropical(5.0)); // max(5,2)
    }

    #[test]
    fn test_apply_vec() {
        let a = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v = vec![Tropical(1.0), Tropical(1.0)];
        let r = a.apply_vec(&v);
        // [max(1+1, 2+1), max(3+1, 4+1)] = [3, 5]
        assert_eq!(r[0], Tropical(3.0));
        assert_eq!(r[1], Tropical(5.0));
    }

    #[test]
    fn test_trace() {
        let a = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(a.trace(), Tropical(4.0)); // max(1,4) = 4
    }

    #[test]
    fn test_zeros() {
        let z = TropicalMatrix::zeros(3, 2);
        assert_eq!(z.get(0, 0), Tropical::NEG_INF);
        assert_eq!(z.nrows(), 3);
        assert_eq!(z.ncols(), 2);
    }

    #[test]
    fn test_to_nalgebra() {
        let a = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m = a.to_nalgebra();
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_kleene_star() {
        let a = TropicalMatrix::from_f64(&[vec![0.0, 1.0], vec![2.0, 0.0]]);
        let star = a.kleene_star();
        // star should contain shortest path distances
        assert!(star.get(0, 0).is_finite());
        assert!(star.get(1, 1).is_finite());
    }
}
