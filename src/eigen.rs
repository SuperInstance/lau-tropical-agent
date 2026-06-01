//! Tropical eigenvalues: the max-plus spectral theory.

use crate::semiring::Tropical;
use crate::matrix::TropicalMatrix;

/// Compute the tropical eigenvalue (max-plus spectral radius) of a square tropical matrix.
/// The tropical eigenvalue λ is the maximum cycle mean: max over all cycles of (total weight / length).
pub fn tropical_eigen(matrix: &TropicalMatrix) -> Tropical {
    assert_eq!(matrix.nrows(), matrix.ncols(), "Matrix must be square");
    let n = matrix.nrows();
    if n == 0 {
        return Tropical::NEG_INF;
    }

    let mut best_mean = Tropical::NEG_INF;

    // Check cycles of length 1..=n
    // For efficiency, use the power method: the k-th power gives best paths of length k
    let mut power = matrix.clone();
    for k in 1..=n {
        // Check diagonal elements (cycles of length k)
        for i in 0..n {
            let val = power.get(i, i);
            if val.is_finite() {
                let mean = val.0 / k as f64;
                if mean > best_mean.0 {
                    best_mean = Tropical(mean);
                }
            }
        }
        if k < n {
            power = matrix.mul(&power);
        }
    }

    best_mean
}

/// Compute tropical eigenvector associated with the tropical eigenvalue.
/// Uses the Kleene star method.
pub fn tropical_eigenvector(matrix: &TropicalMatrix) -> Vec<Tropical> {
    let eigenvalue = tropical_eigen(matrix);
    if eigenvalue.is_zero() {
        return vec![Tropical::NEG_INF; matrix.nrows()];
    }

    // Normalize: A' = A - λ (tropical division)
    let n = matrix.nrows();
    let mut normalized = TropicalMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let val = matrix.get(i, j);
            if val.is_finite() {
                normalized.set(i, j, Tropical(val.0 - eigenvalue.0));
            }
        }
    }

    // Eigenvector is any column of the Kleene star with a finite diagonal
    let star = normalized.kleene_star();
    (0..n).map(|i| star.get(i, 0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigen_identity() {
        let eye = TropicalMatrix::identity(3);
        let eigen = tropical_eigen(&eye);
        // Cycles on diagonal all have weight 0, so mean = 0/1 = 0
        assert_eq!(eigen, Tropical(0.0));
    }

    #[test]
    fn test_eigen_single_element() {
        let m = TropicalMatrix::from_f64(&[vec![5.0]]);
        let eigen = tropical_eigen(&m);
        assert_eq!(eigen, Tropical(5.0));
    }

    #[test]
    fn test_eigen_2x2() {
        let m = TropicalMatrix::from_f64(&[
            vec![1.0, 3.0],
            vec![2.0, 1.0],
        ]);
        let eigen = tropical_eigen(&m);
        // Cycle (0->1->0): weight 3+2=5, length 2, mean 2.5
        // Diagonal cycles: mean 1
        assert!((eigen.0 - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_eigen_empty() {
        let m = TropicalMatrix::zeros(0, 0);
        let eigen = tropical_eigen(&m);
        assert_eq!(eigen, Tropical::NEG_INF);
    }

    #[test]
    fn test_eigenvector_identity() {
        let eye = TropicalMatrix::identity(2);
        let v = tropical_eigenvector(&eye);
        assert_eq!(v.len(), 2);
        // All entries should be finite for identity
        assert!(v[0].is_finite());
    }

    #[test]
    fn test_eigen_negative_cycle() {
        let m = TropicalMatrix::from_f64(&[
            vec![f64::NEG_INFINITY, (-1.0)],
            vec![(-1.0), f64::NEG_INFINITY],
        ]);
        // This isn't valid tropical since NEG_INF is tropical zero, so cycle 0->1->0 = -1 + -1 = -2, mean -1
        let eigen = tropical_eigen(&m);
        assert!(eigen.is_finite());
    }

    #[test]
    fn test_eigen_3x3_chain() {
        let m = TropicalMatrix::from_f64(&[
            vec![f64::NEG_INFINITY, 5.0, f64::NEG_INFINITY],
            vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 3.0],
            vec![2.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        ]);
        // Cycle 0->1->2->0: weight 5+3+2=10, length 3, mean 10/3
        let eigen = tropical_eigen(&m);
        assert!((eigen.0 - 10.0 / 3.0).abs() < 1e-10);
    }
}
