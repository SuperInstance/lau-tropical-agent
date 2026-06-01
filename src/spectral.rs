//! Tropical spectral theory: eigenvalues, singular values, and spectral radius.

use crate::semiring::Tropical;
use crate::matrix::TropicalMatrix;

/// Compute the tropical spectral radius (same as tropical eigenvalue).
pub fn spectral_radius(matrix: &TropicalMatrix) -> Tropical {
    crate::eigen::tropical_eigen(matrix)
}

/// Compute tropical singular values.
/// In max-plus algebra, singular values are related to the cycle means of A⊗A^T.
pub fn tropical_singular_values(matrix: &TropicalMatrix) -> Vec<Tropical> {
    let n = matrix.nrows();
    let m = matrix.ncols();

    // Compute A ⊗ A^T (tropical grammian)
    let mut grammian = TropicalMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut best = Tropical::NEG_INF;
            for k in 0..m {
                best = best + matrix.get(i, k) * matrix.get(j, k);
            }
            grammian.set(i, j, best);
        }
    }

    // Singular values are related to eigenvalues of the grammian
    let eigen = crate::eigen::tropical_eigen(&grammian);

    // Return singular value as sqrt-like tropical operation (half the eigenvalue)
    let mut sv = vec![Tropical::NEG_INF; n.min(m)];
    if eigen.is_finite() {
        sv[0] = Tropical(eigen.0 / 2.0);
    }
    sv
}

/// Compute the tropical condition number (ratio of largest to smallest "singular values").
pub fn tropical_condition_number(matrix: &TropicalMatrix) -> f64 {
    let sv = tropical_singular_values(matrix);
    let max_sv = sv.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max);
    let min_sv = sv.iter().filter(|t| t.is_finite()).map(|t| t.0).fold(f64::INFINITY, f64::min);
    if min_sv.is_infinite() {
        return f64::INFINITY;
    }
    max_sv - min_sv // tropical "ratio"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_radius_identity() {
        let eye = TropicalMatrix::identity(3);
        assert_eq!(spectral_radius(&eye), Tropical(0.0));
    }

    #[test]
    fn test_spectral_radius_single() {
        let m = TropicalMatrix::from_f64(&[vec![7.0]]);
        assert_eq!(spectral_radius(&m), Tropical(7.0));
    }

    #[test]
    fn test_singular_values_identity() {
        let eye = TropicalMatrix::identity(2);
        let sv = tropical_singular_values(&eye);
        assert_eq!(sv.len(), 2);
        assert!(sv[0].is_finite());
    }

    #[test]
    fn test_singular_values_rectangular() {
        let m = TropicalMatrix::from_f64(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let sv = tropical_singular_values(&m);
        assert_eq!(sv.len(), 2); // min(2,3) = 2
        assert!(sv[0].is_finite());
    }

    #[test]
    fn test_condition_number_identity() {
        let eye = TropicalMatrix::identity(3);
        let cn = tropical_condition_number(&eye);
        assert!(cn.is_finite());
    }

    #[test]
    fn test_spectral_radius_2x2() {
        let m = TropicalMatrix::from_f64(&[
            vec![1.0, 4.0],
            vec![3.0, 2.0],
        ]);
        let sr = spectral_radius(&m);
        // Cycle 0->1->0: max(4+3, 1+2) = 7, length 2, mean 3.5
        assert!((sr.0 - 3.5).abs() < 1e-10);
    }
}
