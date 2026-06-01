//! Tropical rational maps: piecewise-linear functions from max-plus expressions.

use crate::polynomial::TropicalPoly;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A tropical rational function: f/g where f and g are tropical polynomials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TropicalRationalMap {
    numerator: TropicalPoly,
    denominator: TropicalPoly,
}

impl TropicalRationalMap {
    /// Create a new tropical rational map from numerator and denominator polynomials.
    pub fn new(numerator: TropicalPoly, denominator: TropicalPoly) -> Self {
        TropicalRationalMap { numerator, denominator }
    }

    /// Evaluate the tropical rational map at x: f(x) - g(x).
    /// In tropical terms: f(x) ⊘ g(x) = f(x).0 - g(x).0
    pub fn eval(&self, x: f64) -> f64 {
        let num = self.numerator.eval(x);
        let den = self.denominator.eval(x);
        if num.is_finite() && den.is_finite() {
            num.0 - den.0
        } else if num.is_finite() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get the numerator polynomial.
    pub fn numerator(&self) -> &TropicalPoly {
        &self.numerator
    }

    /// Get the denominator polynomial.
    pub fn denominator(&self) -> &TropicalPoly {
        &self.denominator
    }
}

impl fmt::Display for TropicalRationalMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ({})", self.numerator, self.denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_eval() {
        // f = max(0, x), g = max(0, x-1) = max(0, -1+x)
        let f = TropicalPoly::new(vec![(Tropical(0.0), 0), (Tropical(0.0), 1)]);
        let g = TropicalPoly::new(vec![(Tropical(0.0), 0), (Tropical(-1.0), 1)]);
        let r = TropicalRationalMap::new(f, g);
        // At x=2: f=max(0,2)=2, g=max(0,1)=1, r=2-1=1
        assert_eq!(r.eval(2.0), 1.0);
    }

    #[test]
    fn test_rational_display() {
        let f = TropicalPoly::monomial(Tropical(1.0), 1);
        let g = TropicalPoly::monomial(Tropical(0.0), 0);
        let r = TropicalRationalMap::new(f, g);
        let s = format!("{}", r);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_rational_accessors() {
        let f = TropicalPoly::monomial(Tropical(1.0), 1);
        let g = TropicalPoly::monomial(Tropical(0.0), 0);
        let r = TropicalRationalMap::new(f.clone(), g.clone());
        assert_eq!(r.numerator().degree(), 1);
        assert_eq!(r.denominator().degree(), 0);
    }

    #[test]
    fn test_rational_constant() {
        let f = TropicalPoly::monomial(Tropical(3.0), 0);
        let g = TropicalPoly::monomial(Tropical(1.0), 0);
        let r = TropicalRationalMap::new(f, g);
        assert_eq!(r.eval(42.0), 2.0); // 3 - 1
    }
}
