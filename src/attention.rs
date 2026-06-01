//! Tropical attention: max-plus softmax approximation.

use crate::semiring::Tropical;
use crate::matrix::TropicalMatrix;
use serde::{Deserialize, Serialize};

/// Tropical attention mechanism — a piecewise-linear approximation of softmax attention.
/// In the tropical limit, softmax becomes hardmax.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TropicalAttention {
    /// Temperature parameter (approaches 0 for hard tropical limit).
    temperature: f64,
}

impl TropicalAttention {
    /// Create a new tropical attention with given temperature.
    pub fn new(temperature: f64) -> Self {
        TropicalAttention { temperature }
    }

    /// Create hard tropical attention (temperature = 0).
    pub fn hard() -> Self {
        TropicalAttention { temperature: 0.0 }
    }

    /// Compute attention weights in the tropical (max-plus) sense.
    /// Returns a weight vector where the maximum element gets weight 1.
    pub fn weights(&self, scores: &[Tropical]) -> Vec<f64> {
        if scores.is_empty() {
            return vec![];
        }
        let max_val = scores.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max);
        if self.temperature <= 0.0 {
            // Hard tropical: all weight on the maximum
            scores.iter().map(|t| if t.0 == max_val { 1.0 } else { 0.0 }).collect()
        } else {
            // Soft tropical approximation
            let exps: Vec<f64> = scores.iter().map(|t| ((t.0 - max_val) / self.temperature).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        }
    }

    /// Tropical attention: output = tropical matrix-vector product with attention weights.
    pub fn attend(&self, keys: &TropicalMatrix, queries: &TropicalMatrix) -> TropicalMatrix {
        // Tropical dot product: K^T ⊗ Q in max-plus sense
        let n = keys.nrows();
        let m = queries.ncols();
        let d = keys.ncols();
        let mut result = TropicalMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut best = Tropical::NEG_INF;
                for k in 0..d {
                    let val = keys.get(i, k) * queries.get(k, j);
                    best = best + val;
                }
                result.set(i, j, best);
            }
        }
        result
    }

    /// Get the temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_attention_weights() {
        let scores = vec![Tropical(1.0), Tropical(5.0), Tropical(3.0)];
        let attn = TropicalAttention::hard();
        let w = attn.weights(&scores);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 1.0);
        assert_eq!(w[2], 0.0);
    }

    #[test]
    fn test_soft_attention_weights() {
        let scores = vec![Tropical(1.0), Tropical(5.0), Tropical(3.0)];
        let attn = TropicalAttention::new(1.0);
        let w = attn.weights(&scores);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(w[1] > w[2]);
        assert!(w[2] > w[0]);
    }

    #[test]
    fn test_empty_weights() {
        let attn = TropicalAttention::hard();
        let w = attn.weights(&[]);
        assert!(w.is_empty());
    }

    #[test]
    fn test_tied_max_weights() {
        let scores = vec![Tropical(3.0), Tropical(3.0), Tropical(1.0)];
        let attn = TropicalAttention::hard();
        let w = attn.weights(&scores);
        assert_eq!(w[0], 1.0);
        assert_eq!(w[1], 1.0);
        assert_eq!(w[2], 0.0);
    }

    #[test]
    fn test_attend() {
        let keys = TropicalMatrix::from_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let queries = TropicalMatrix::from_f64(&[vec![0.0], vec![1.0]]);
        let attn = TropicalAttention::hard();
        let result = attn.attend(&keys, &queries);
        assert!(result.get(0, 0).is_finite());
        assert!(result.get(1, 0).is_finite());
    }

    #[test]
    fn test_temperature_accessor() {
        let attn = TropicalAttention::new(0.5);
        assert_eq!(attn.temperature(), 0.5);
    }
}
