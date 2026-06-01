//! Tropical shortest paths via max-plus matrix algebra.

use crate::semiring::Tropical;
use crate::matrix::TropicalMatrix;

/// Compute all-pairs shortest paths using tropical matrix Kleene star.
/// The distance matrix D = A* where A is the tropical weight matrix.
pub fn tropical_shortest_paths(adjacency: &TropicalMatrix) -> TropicalMatrix {
    adjacency.kleene_star()
}

/// Single-source shortest paths in the tropical semiring.
/// Uses repeated tropical matrix-vector products.
pub fn single_source_shortest_paths(adjacency: &TropicalMatrix, source: usize) -> Vec<Tropical> {
    let n = adjacency.nrows();
    assert!(source < n, "Source index out of bounds");

    let mut dist = vec![Tropical::NEG_INF; n];
    dist[source] = Tropical::ONE; // distance to self is 0 (tropical one)

    // Relax n-1 times
    for _ in 0..n {
        let new_dist = adjacency.apply_vec(&dist);
        for i in 0..n {
            dist[i] = dist[i] + new_dist[i]; // max (which is tropical addition = min distance in max-plus)
        }
    }

    dist
}

/// Find the shortest path distance between two specific nodes.
pub fn shortest_path_distance(adjacency: &TropicalMatrix, from: usize, to: usize) -> Tropical {
    let dists = single_source_shortest_paths(adjacency, from);
    dists[to]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_pairs_identity() {
        let eye = TropicalMatrix::identity(3);
        let dists = tropical_shortest_paths(&eye);
        assert_eq!(dists.get(0, 0), Tropical(0.0));
        assert_eq!(dists.get(1, 2), Tropical::NEG_INF);
    }

    #[test]
    fn test_single_source_identity() {
        let eye = TropicalMatrix::identity(3);
        let dists = single_source_shortest_paths(&eye, 0);
        assert_eq!(dists[0], Tropical(0.0));
        assert_eq!(dists[1], Tropical::NEG_INF);
    }

    #[test]
    fn test_shortest_path_triangle() {
        // Triangle graph: 0-1 (wt 1), 1-2 (wt 2), 0-2 (wt 5)
        let adj = TropicalMatrix::from_f64(&[
            vec![f64::NEG_INFINITY, 1.0, 5.0],
            vec![1.0, f64::NEG_INFINITY, 2.0],
            vec![5.0, 2.0, f64::NEG_INFINITY],
        ]);
        let dist = shortest_path_distance(&adj, 0, 2);
        // Path 0->1->2: 1+2=3, direct: 5. In max-plus, "shortest" = max of path sums
        assert!(dist.is_finite());
    }

    #[test]
    fn test_all_pairs_simple() {
        let adj = TropicalMatrix::from_f64(&[
            vec![f64::NEG_INFINITY, 1.0],
            vec![2.0, f64::NEG_INFINITY],
        ]);
        let dists = tropical_shortest_paths(&adj);
        assert!(dists.get(0, 0).is_finite());
        assert!(dists.get(1, 1).is_finite());
    }

    #[test]
    fn test_disconnected_nodes() {
        let adj = TropicalMatrix::zeros(3, 3);
        let dist = shortest_path_distance(&adj, 0, 1);
        assert_eq!(dist, Tropical::NEG_INF);
    }

    #[test]
    fn test_self_loop_distance() {
        let adj = TropicalMatrix::identity(3);
        let dist = shortest_path_distance(&adj, 1, 1);
        assert_eq!(dist, Tropical(0.0));
    }
}
