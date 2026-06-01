//! Tropical semiring (ℝ ∪ {-∞}, max, +) with verified semiring laws.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul};

/// The tropical semiring element: ℝ ∪ {-∞} with ⊕ = max, ⊗ = +.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Tropical(pub f64);

impl Tropical {
    /// Tropical zero element: -∞ (additive identity for max).
    pub const NEG_INF: Tropical = Tropical(f64::NEG_INFINITY);

    /// Tropical one element: 0 (multiplicative identity for +).
    pub const ONE: Tropical = Tropical(0.0);

    /// Create a tropical number from an f64.
    pub fn new(val: f64) -> Self {
        Tropical(val)
    }

    /// Is this the zero element (-∞)?
    pub fn is_zero(&self) -> bool {
        self.0 == f64::NEG_INFINITY
    }

    /// Is this finite (not -∞)?
    pub fn is_finite(&self) -> bool {
        self.0 != f64::NEG_INFINITY
    }

    /// Tropical exponentiation: a^n = n * a (repeated tropical multiplication).
    pub fn pow(self, n: u32) -> Self {
        if self.is_zero() {
            return Self::NEG_INF;
        }
        Tropical(self.0 * n as f64)
    }

    /// Tropical division (inverse of ⊗): a/b = a - b.
    /// Only meaningful when b is finite and non-zero-tropical.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        Tropical(self.0 - other.0)
    }
}

impl Default for Tropical {
    fn default() -> Self {
        Self::NEG_INF
    }
}

impl fmt::Display for Tropical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == f64::NEG_INFINITY {
            write!(f, "-∞")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

/// Tropical addition: max(a, b).
impl Add for Tropical {
    type Output = Tropical;
    fn add(self, rhs: Self) -> Self::Output {
        Tropical(self.0.max(rhs.0))
    }
}

/// Tropical multiplication: a + b.
impl Mul for Tropical {
    type Output = Tropical;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Tropical::NEG_INF;
        }
        Tropical(self.0 + rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_add_commutative() {
        let a = Tropical(3.0);
        let b = Tropical(5.0);
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn test_tropical_add_associative() {
        let a = Tropical(1.0);
        let b = Tropical(2.0);
        let c = Tropical(3.0);
        assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn test_tropical_add_identity() {
        // -∞ is the additive identity: max(a, -∞) = a
        let a = Tropical(42.0);
        assert_eq!(a + Tropical::NEG_INF, a);
        assert_eq!(Tropical::NEG_INF + a, a);
    }

    #[test]
    fn test_tropical_add_idempotent() {
        let a = Tropical(7.0);
        assert_eq!(a + a, a);
    }

    #[test]
    fn test_tropical_mul_commutative() {
        let a = Tropical(3.0);
        let b = Tropical(5.0);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_tropical_mul_associative() {
        let a = Tropical(1.0);
        let b = Tropical(2.0);
        let c = Tropical(3.0);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_tropical_mul_identity() {
        // 0 is the multiplicative identity: a + 0 = a
        let a = Tropical(42.0);
        assert_eq!(a * Tropical::ONE, a);
        assert_eq!(Tropical::ONE * a, a);
    }

    #[test]
    fn test_tropical_mul_absorbing() {
        // -∞ is absorbing: a + (-∞) = -∞
        let a = Tropical(42.0);
        assert_eq!(a * Tropical::NEG_INF, Tropical::NEG_INF);
    }

    #[test]
    fn test_tropical_distributive() {
        let a = Tropical(1.0);
        let b = Tropical(2.0);
        let c = Tropical(3.0);
        // a ⊗ (b ⊕ c) = a ⊗ b ⊕ a ⊗ c
        let left = a * (b + c);
        let right = (a * b) + (a * c);
        assert_eq!(left, right);
    }

    #[test]
    fn test_tropical_pow() {
        let a = Tropical(3.0);
        assert_eq!(a.pow(0), Tropical::ONE); // 0*3 = 0
        assert_eq!(a.pow(1), Tropical(3.0));
        assert_eq!(a.pow(2), Tropical(6.0)); // 3+3
        assert_eq!(a.pow(3), Tropical(9.0)); // 3+3+3
    }

    #[test]
    fn test_tropical_zero_pow() {
        let z = Tropical::NEG_INF;
        assert_eq!(z.pow(3), Tropical::NEG_INF);
    }

    #[test]
    fn test_tropical_div() {
        let a = Tropical(10.0);
        let b = Tropical(3.0);
        assert_eq!(a.div(b), Tropical(7.0)); // 10 - 3
    }

    #[test]
    fn test_serde_roundtrip() {
        let a = Tropical(3.14);
        let json = serde_json::to_string(&a).unwrap();
        let b: Tropical = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Tropical::NEG_INF), "-∞");
        assert_eq!(format!("{}", Tropical(3.5)), "3.5");
    }
}
