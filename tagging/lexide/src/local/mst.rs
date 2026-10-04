//! Single-root Chu–Liu–Edmonds; exact port of tagger/mst.py.
use anyhow::{ensure, Result};

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

fn cycle(heads: &[usize]) -> Option<Vec<usize>> {
    let mut done = vec![false; heads.len()];
    done[0] = true;
    for start in 1..heads.len() {
        let mut path = Vec::new();
        let mut node = start;
        while !done[node] {
            if let Some(pos) = path.iter().position(|&n| n == node) {
                return Some(path[pos..].to_vec());
            }
            path.push(node);
            node = heads[node];
        }
        for node in path {
            done[node] = true;
        }
    }
    None
}

fn cle(scores: &[Vec<f64>]) -> Vec<usize> {
    let mut heads: Vec<_> = scores.iter().map(|r| argmax(r)).collect();
    heads[0] = 0;
    let Some(cycle) = cycle(&heads) else {
        return heads;
    };
    let outside: Vec<_> = (0..heads.len()).filter(|i| !cycle.contains(i)).collect();
    let c = outside.len();
    let mut reduced = vec![vec![f64::NEG_INFINITY; c + 1]; c + 1];
    let mut entry = vec![0; c];
    let mut exit = vec![0; c];
    for (i, &node) in outside.iter().enumerate() {
        for (j, &other) in outside.iter().enumerate() {
            reduced[i][j] = scores[node][other];
        }
        let values: Vec<_> = cycle
            .iter()
            .map(|&v| scores[v][node] - scores[v][heads[v]])
            .collect();
        entry[i] = cycle[argmax(&values)];
        reduced[c][i] = values[argmax(&values)];
        let values: Vec<_> = cycle.iter().map(|&v| scores[node][v]).collect();
        exit[i] = cycle[argmax(&values)];
        reduced[i][c] = values[argmax(&values)];
    }
    reduced[0].fill(f64::NEG_INFINITY);
    reduced[0][0] = 0.;
    let parents = cle(&reduced);
    for (i, &node) in outside.iter().enumerate().skip(1) {
        heads[node] = if parents[i] == c {
            exit[i]
        } else {
            outside[parents[i]]
        };
    }
    heads[entry[parents[c]]] = outside[parents[c]];
    heads
}

pub(super) fn single_root_mst(arcs: &[f32], n: usize) -> Result<Vec<usize>> {
    ensure!(arcs.len() == n * (n + 1), "Expected W x (W+1) arc scores");
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut scores = vec![vec![f64::NEG_INFINITY; n + 1]; n + 1];
    scores[0][0] = 0.;
    for i in 0..n {
        for j in 0..=n {
            scores[i + 1][j] = f64::from(arcs[i * (n + 1) + j]);
        }
        scores[i + 1][i + 1] = f64::NEG_INFINITY;
        ensure!(
            scores[i + 1].iter().any(|s| s.is_finite()),
            "A word has no finite head candidate"
        );
    }
    let heads = cle(&scores);
    if heads[1..].iter().filter(|&&h| h == 0).count() == 1 {
        return Ok(heads[1..].to_vec());
    }
    let mut best = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    for root in 1..=n {
        if !scores[root][0].is_finite() {
            continue;
        }
        let mut constrained = scores.clone();
        for row in &mut constrained[1..] {
            row[0] = f64::NEG_INFINITY;
        }
        constrained[root].fill(f64::NEG_INFINITY);
        constrained[root][0] = scores[root][0];
        let candidate = cle(&constrained);
        let value: f64 = (1..=n).map(|i| scores[i][candidate[i]]).sum();
        if value > best_score {
            best = candidate;
            best_score = value;
        }
    }
    ensure!(!best.is_empty(), "No finite single-root tree");
    Ok(best[1..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn agrees_with_exhaustive_single_root_trees() {
        let mut seed = 123_u64;
        for n in 1_usize..=4 {
            for _ in 0..15 {
                let arcs: Vec<f32> = (0..n * (n + 1))
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 32) % 101) as f32 - 50.
                    })
                    .collect();
                let actual = single_root_mst(&arcs, n).unwrap();
                let valid = |heads: &[usize]| {
                    if heads.iter().filter(|&&h| h == 0).count() != 1 {
                        return false;
                    }
                    (1..=n).all(|mut node| {
                        for _ in 0..n {
                            if node == 0 {
                                return true;
                            }
                            node = heads[node - 1];
                        }
                        node == 0
                    })
                };
                let value = |heads: &[usize]| {
                    heads
                        .iter()
                        .enumerate()
                        .map(|(i, &h)| arcs[i * (n + 1) + h])
                        .sum::<f32>()
                };
                assert!(valid(&actual));
                let mut best = f32::NEG_INFINITY;
                for mut code in 0..(n + 1).pow(n as u32) {
                    let heads: Vec<_> = (0..n)
                        .map(|_| {
                            let h = code % (n + 1);
                            code /= n + 1;
                            h
                        })
                        .collect();
                    if valid(&heads) {
                        best = best.max(value(&heads));
                    }
                }
                assert_eq!(value(&actual), best);
            }
        }
    }

    #[test]
    fn roots_and_cycles() {
        assert_eq!(single_root_mst(&[], 0).unwrap(), Vec::<usize>::new());
        assert_eq!(
            single_root_mst(&[5., 0., 4., 5., 4., 0.], 2).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            single_root_mst(&[1., 0., 8., 2., 9., 0.], 2).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            single_root_mst(&[1., 0., 8., 3., 9., 0.], 2).unwrap(),
            vec![2, 0]
        );
    }
}
