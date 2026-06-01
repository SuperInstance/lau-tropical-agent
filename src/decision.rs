//! Tropical decision-making: optimal choices via max-plus algebra.

use crate::semiring::Tropical;
use crate::matrix::TropicalMatrix;
use serde::{Deserialize, Serialize};

/// A tropical decision agent that selects actions using max-plus optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TropicalDecision {
    /// Number of states.
    n_states: usize,
    /// Number of actions.
    n_actions: usize,
    /// Reward matrix: reward[s][a] = tropical reward for taking action a in state s.
    rewards: TropicalMatrix,
    /// Discount factor (tropical: added to path costs).
    discount: f64,
}

impl TropicalDecision {
    /// Create a new tropical decision agent.
    pub fn new(rewards: TropicalMatrix, discount: f64) -> Self {
        let n_states = rewards.nrows();
        let n_actions = rewards.ncols();
        TropicalDecision { n_states, n_actions, rewards, discount }
    }

    /// Select the best action for a given state (tropical argmax).
    pub fn best_action(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut best_val = Tropical::NEG_INF;
        for a in 0..self.n_actions {
            let val = self.rewards.get(state, a);
            if val > best_val {
                best_val = val;
                best_action = a;
            }
        }
        best_action
    }

    /// Get the optimal value for a state (tropical max over actions).
    pub fn optimal_value(&self, state: usize) -> Tropical {
        let mut best = Tropical::NEG_INF;
        for a in 0..self.n_actions {
            best = best + self.rewards.get(state, a); // max
        }
        best
    }

    /// Value iteration in the tropical setting: V' = max_a(R(s,a) + γ⊗V).
    /// This is just tropical matrix multiplication with the reward matrix.
    pub fn value_iteration(&self, transition: &TropicalMatrix, iterations: usize) -> Vec<Tropical> {
        let n = self.n_states;
        let mut v = vec![Tropical::NEG_INF; n];
        for _ in 0..iterations {
            let v_new: Vec<Tropical> = (0..n).map(|s| {
                let mut best = Tropical::NEG_INF;
                for a in 0..self.n_actions {
                    let reward = self.rewards.get(s, a);
                    let mut future = Tropical::NEG_INF;
                    for (s2, v_val) in v.iter().enumerate().take(n) {
                        let t_idx = (s * self.n_actions + a) % transition.nrows();
                        let t_val = transition.get(t_idx, s2.min(transition.ncols() - 1));
                        future = future + Tropical(self.discount) * t_val * *v_val;
                    }
                    best = best + (reward + future);
                }
                best
            }).collect();
            v = v_new;
        }
        v
    }

    /// Get number of states.
    pub fn n_states(&self) -> usize { self.n_states }
    /// Get number of actions.
    pub fn n_actions(&self) -> usize { self.n_actions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_action() {
        let rewards = TropicalMatrix::from_f64(&[
            vec![1.0, 5.0, 3.0],
            vec![4.0, 2.0, 6.0],
        ]);
        let agent = TropicalDecision::new(rewards, 0.9);
        assert_eq!(agent.best_action(0), 1); // max(1,5,3) = 5 at index 1
        assert_eq!(agent.best_action(1), 2); // max(4,2,6) = 6 at index 2
    }

    #[test]
    fn test_optimal_value() {
        let rewards = TropicalMatrix::from_f64(&[
            vec![1.0, 5.0],
        ]);
        let agent = TropicalDecision::new(rewards, 0.9);
        assert_eq!(agent.optimal_value(0), Tropical(5.0));
    }

    #[test]
    fn test_decision_new() {
        let rewards = TropicalMatrix::zeros(3, 2);
        let agent = TropicalDecision::new(rewards, 0.95);
        assert_eq!(agent.n_states(), 3);
        assert_eq!(agent.n_actions(), 2);
    }

    #[test]
    fn test_value_iteration_runs() {
        let rewards = TropicalMatrix::from_f64(&[
            vec![1.0, 2.0],
            vec![3.0, 0.0],
        ]);
        let transition = TropicalMatrix::identity(2);
        let agent = TropicalDecision::new(rewards, 0.9);
        let v = agent.value_iteration(&transition, 10);
        assert_eq!(v.len(), 2);
        assert!(v[0].is_finite());
        assert!(v[1].is_finite());
    }

    #[test]
    fn test_single_state_single_action() {
        let rewards = TropicalMatrix::from_f64(&[vec![7.0]]);
        let agent = TropicalDecision::new(rewards, 0.5);
        assert_eq!(agent.best_action(0), 0);
        assert_eq!(agent.optimal_value(0), Tropical(7.0));
    }
}
