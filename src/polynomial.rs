//! Tropical polynomials: piecewise-linear functions and Newton polytopes.

use crate::semiring::Tropical;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A tropical polynomial: max of (a_i + dot(x, m_i)) where m_i are exponent vectors.
/// For univariate: max_i (a_i + i * x).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TropicalPoly {
    /// Coefficients and their degrees. (coefficient, degree).
    /// Univariate: degree is just the power.
    terms: Vec<(Tropical, usize)>,
}

impl TropicalPoly {
    /// Create from a list of (coefficient, degree) pairs.
    pub fn new(terms: Vec<(Tropical, usize)>) -> Self {
        let mut p = TropicalPoly { terms };
        p.canonicalize();
        p
    }

    /// Create a tropical monomial: a ⊗ x^n = a + n*x.
    pub fn monomial(coeff: Tropical, degree: usize) -> Self {
        TropicalPoly::new(vec![(coeff, degree)])
    }

    /// Evaluate the tropical polynomial at x: max_i (a_i + degree_i * x).
    pub fn eval(&self, x: f64) -> Tropical {
        let mut best = Tropical::NEG_INF;
        for (coeff, deg) in &self.terms {
            if coeff.is_finite() {
                let val = coeff.0 + (*deg as f64) * x;
                best = best + Tropical(val); // max
            }
        }
        best
    }

    /// The corner locus (roots) — points where the maximum is attained by at least two terms.
    /// These are the "tropical zeros" of the polynomial.
    pub fn corner_locus(&self) -> Vec<f64> {
        let mut corners = Vec::new();
        for i in 0..self.terms.len() {
            for j in (i + 1)..self.terms.len() {
                let (ci, di) = self.terms[i];
                let (cj, dj) = self.terms[j];
                if di != dj && ci.is_finite() && cj.is_finite() {
                    // ci + di*x = cj + dj*x => x = (cj - ci) / (di - dj)
                    let x = (cj.0 - ci.0) / (di as f64 - dj as f64);
                    corners.push(x);
                }
            }
        }
        corners.sort_by(|a, b| a.partial_cmp(b).unwrap());
        corners
    }

    /// The Newton polytope: convex hull of exponent vectors (here just degrees as 1D points).
    /// Returns (min_degree, max_degree).
    pub fn newton_polytope(&self) -> Option<(usize, usize)> {
        let finite: Vec<_> = self.terms.iter().filter(|(c, _)| c.is_finite()).collect();
        if finite.is_empty() {
            return None;
        }
        let min_deg = finite.iter().map(|(_, d)| *d).min().unwrap();
        let max_deg = finite.iter().map(|(_, d)| *d).max().unwrap();
        Some((min_deg, max_deg))
    }

    /// Tropical polynomial addition (max of the two).
    pub fn add(&self, other: &TropicalPoly) -> TropicalPoly {
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        TropicalPoly::new(terms)
    }

    /// Tropical polynomial multiplication (convolution of terms).
    pub fn mul(&self, other: &TropicalPoly) -> TropicalPoly {
        let mut terms = Vec::new();
        for (c1, d1) in &self.terms {
            for (c2, d2) in &other.terms {
                terms.push((*c1 * *c2, d1 + d2)); // coeff: c1+c2, degree: d1+d2
            }
        }
        TropicalPoly::new(terms)
    }

    /// Degree of the polynomial (highest degree with finite coefficient).
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .filter(|(c, _)| c.is_finite())
            .map(|(_, d)| *d)
            .max()
            .unwrap_or(0)
    }

    fn canonicalize(&mut self) {
        // Remove zero-coefficient terms
        self.terms.retain(|(c, _)| c.is_finite());
        // Sort by degree
        self.terms.sort_by_key(|(_, d)| *d);
    }
}

impl fmt::Display for TropicalPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "-∞");
        }
        let parts: Vec<String> = self
            .terms
            .iter()
            .map(|(c, d)| {
                if *d == 0 {
                    format!("{}", c.0)
                } else if *d == 1 {
                    format!("{}⊕{}x", c.0, 1.0)
                } else {
                    format!("({})⊗x^{}", c.0, d)
                }
            })
            .collect();
        write!(f, "max({})", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_simple() {
        // p(x) = max(0 + 0*x, 1 + 1*x) = max(0, 1+x)
        let p = TropicalPoly::new(vec![(Tropical(0.0), 0), (Tropical(1.0), 1)]);
        assert_eq!(p.eval(0.0), Tropical(1.0)); // max(0, 1) = 1
        assert_eq!(p.eval(-1.0), Tropical(0.0)); // max(0, 0) = 0
        assert_eq!(p.eval(5.0), Tropical(6.0)); // max(0, 6) = 6
    }

    #[test]
    fn test_eval_quadratic() {
        // p(x) = max(0, x, 2x) — tropical quadratic
        let p = TropicalPoly::new(vec![
            (Tropical(0.0), 0),
            (Tropical(0.0), 1),
            (Tropical(0.0), 2),
        ]);
        assert_eq!(p.eval(1.0), Tropical(2.0)); // max(0, 1, 2) = 2
        assert_eq!(p.eval(-1.0), Tropical(0.0)); // max(0, -1, -2) = 0
    }

    #[test]
    fn test_corner_locus() {
        // p(x) = max(0, x) has corner at x=0
        let p = TropicalPoly::new(vec![(Tropical(0.0), 0), (Tropical(0.0), 1)]);
        let corners = p.corner_locus();
        assert_eq!(corners, vec![0.0]);
    }

    #[test]
    fn test_corner_locus_three_terms() {
        // p(x) = max(0, x, 2x-2) has corners at x=0 and x=2
        let p = TropicalPoly::new(vec![
            (Tropical(0.0), 0),
            (Tropical(0.0), 1),
            (Tropical(-2.0), 2),
        ]);
        let corners = p.corner_locus();
        // corner_locus returns all pairwise crossings
        assert!(corners.contains(&2.0));
    }

    #[test]
    fn test_newton_polytope() {
        let p = TropicalPoly::new(vec![
            (Tropical(1.0), 0),
            (Tropical(2.0), 3),
            (Tropical(1.0), 5),
        ]);
        assert_eq!(p.newton_polytope(), Some((0, 5)));
    }

    #[test]
    fn test_poly_multiply() {
        // (0⊕1x) * (0⊕1x) = max(0, x, x, 2x) = max(0, x, 2x)
        let p = TropicalPoly::new(vec![(Tropical(0.0), 0), (Tropical(0.0), 1)]);
        let q = p.mul(&p);
        assert_eq!(q.eval(1.0), Tropical(2.0)); // max(0, 1, 2) = 2
    }

    #[test]
    fn test_poly_add() {
        let p = TropicalPoly::monomial(Tropical(1.0), 2);
        let q = TropicalPoly::monomial(Tropical(3.0), 0);
        let r = p.add(&q);
        // r(x) = max(3, 1+2x)
        assert_eq!(r.eval(0.0), Tropical(3.0)); // max(3, 1) = 3
        assert_eq!(r.eval(2.0), Tropical(5.0)); // max(3, 5) = 5
    }

    #[test]
    fn test_degree() {
        let p = TropicalPoly::new(vec![
            (Tropical(1.0), 0),
            (Tropical(2.0), 3),
        ]);
        assert_eq!(p.degree(), 3);
    }

    #[test]
    fn test_monomial_eval() {
        let m = TropicalPoly::monomial(Tropical(2.0), 3);
        assert_eq!(m.eval(4.0), Tropical(14.0)); // 2 + 3*4 = 14
    }
}
