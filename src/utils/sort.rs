use core::mem;

/// Fast u32-key radix sort for SoA (inds/vals).
/// - Sorts by inds ascending
/// - Reorders vals accordingly
/// - Requires `N: Copy` for speed (good for f16/f32 etc.)
///
/// Complexity: 4 passes, each O(n + 256)
#[inline(always)]
pub fn radix_sort_u32_soa<N: Copy>(inds: &mut [u32], vals: &mut [N]) {
    assert_eq!(inds.len(), vals.len());
    let n = inds.len();
    if n <= 1 {
        return;
    }

    // Small sizes: insertion sort is often faster than allocating scratch.
    if n <= 32 {
        insertion_sort_u32_soa(inds, vals);
        return;
    }

    // Scratch buffers (allocate once per call)
    let mut inds_tmp = vec![0u32; n];
    let mut vals_tmp: Vec<N> = vec![unsafe { mem::zeroed() }; n];

    // Alternate between (src -> dst)
    let mut src_inds: &mut [u32] = inds;
    let mut src_vals: &mut [N] = vals;
    let mut dst_inds: &mut [u32] = &mut inds_tmp;
    let mut dst_vals: &mut [N] = &mut vals_tmp;

    // 4 passes: byte 0..3 (LSD)
    for shift in [0u32, 8, 16, 24] {
        let mut count = [0usize; 256];

        // Count
        for &k in src_inds.iter() {
            count[((k >> shift) & 0xFF) as usize] += 1;
        }

        // Prefix sum -> starting positions
        let mut sum = 0usize;
        for c in count.iter_mut() {
            let tmp = *c;
            *c = sum;
            sum += tmp;
        }

        // Distribute (stable)
        // NOTE: writing with raw pointers helps the optimizer a bit.
        unsafe {
            let s_i = src_inds.as_ptr();
            let s_v = src_vals.as_ptr();
            let d_i = dst_inds.as_mut_ptr();
            let d_v = dst_vals.as_mut_ptr();

            for idx in 0..n {
                let k = *s_i.add(idx);
                let b = ((k >> shift) & 0xFF) as usize;
                let pos = count[b];
                count[b] = pos + 1;

                *d_i.add(pos) = k;
                *d_v.add(pos) = *s_v.add(idx);
            }
        }

        // swap src/dst for next pass
        mem::swap(&mut src_inds, &mut dst_inds);
        mem::swap(&mut src_vals, &mut dst_vals);
    }

    // After 4 passes, result is in `src_*`.
    // Since we did 4 passes (even), `src_*` points back to original (inds/vals).
    // If you change pass count, you may need a copy-back.
    // (Here no-op.)
}

/// Tiny insertion sort for small n (SoA).
#[inline(always)]
fn insertion_sort_u32_soa<N: Copy>(inds: &mut [u32], vals: &mut [N]) {
    let n = inds.len();
    for i in 1..n {
        let mut j = i;
        while j > 0 && inds[j] < inds[j - 1] {
            inds.swap(j, j - 1);
            vals.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare with stable baseline:
    /// sort by key, and if key is equal, preserve original order (stable).
    fn baseline_stable_sort<N: Copy>(inds: &[u32], vals: &[N]) -> (Vec<u32>, Vec<N>) {
        let mut pairs: Vec<(u32, usize, N)> = inds
            .iter()
            .copied()
            .enumerate()
            .map(|(i, k)| (k, i, vals[i]))
            .collect();

        // stable baseline: sort by (key, original_index)
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut out_k = Vec::with_capacity(pairs.len());
        let mut out_v = Vec::with_capacity(pairs.len());
        for (k, _i, v) in pairs {
            out_k.push(k);
            out_v.push(v);
        }
        (out_k, out_v)
    }

    fn assert_sorted(keys: &[u32]) {
        for i in 1..keys.len() {
            assert!(keys[i - 1] <= keys[i], "not sorted at {i}: {} > {}", keys[i - 1], keys[i]);
        }
    }

    /// tiny deterministic PRNG (xorshift32)
    struct Rng(u32);
    impl Rng {
        fn new(seed: u32) -> Self { Self(seed) }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    #[test]
    fn radix_sort_handles_empty_and_single() {
        // empty
        let mut inds: Vec<u32> = vec![];
        let mut vals: Vec<u16> = vec![];
        radix_sort_u32_soa(&mut inds, &mut vals);
        assert!(inds.is_empty());
        assert!(vals.is_empty());

        // single
        let mut inds = vec![42u32];
        let mut vals = vec![7u16];
        radix_sort_u32_soa(&mut inds, &mut vals);
        assert_eq!(inds, vec![42u32]);
        assert_eq!(vals, vec![7u16]);
    }

    #[test]
    fn radix_sort_works_on_duplicates_and_preserves_pairing() {
        // Keys have duplicates; vals encode original position
        let mut inds = vec![3u32, 1, 3, 2, 1, 3, 0];
        let mut vals: Vec<u32> = (0..inds.len() as u32).collect();

        let (base_k, base_v) = baseline_stable_sort(&inds, &vals);

        radix_sort_u32_soa(&mut inds, &mut vals);

        assert_sorted(&inds);
        assert_eq!(inds, base_k);
        assert_eq!(vals, base_v, "radix sort should be stable in this implementation");
    }

    #[test]
    fn radix_sort_matches_baseline_many_sizes() {
        let mut rng = Rng::new(0x1234_5678);

        // Test a range of sizes, including small threshold area and larger sizes.
        for &n in &[0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024] {
            let mut inds = Vec::with_capacity(n);
            let mut vals = Vec::with_capacity(n);

            for i in 0..n {
                // Make duplicates somewhat likely by masking.
                let k = rng.next_u32() & 0x00FF_FFFF;
                inds.push(k);
                // Value carries identity to verify pairing
                vals.push((i as u32) ^ 0xA5A5_5A5A);
            }

            let (base_k, base_v) = baseline_stable_sort(&inds, &vals);

            radix_sort_u32_soa(&mut inds, &mut vals);

            assert_sorted(&inds);
            assert_eq!(inds, base_k, "keys mismatch at n={n}");
            assert_eq!(vals, base_v, "vals mismatch at n={n}");
        }
    }

    #[test]
    fn radix_sort_extremes() {
        let mut inds = vec![
            0u32,
            u32::MAX,
            1,
            u32::MAX - 1,
            0,
            2,
            u32::MAX,
        ];
        let mut vals: Vec<u32> = (0..inds.len() as u32).collect();

        let (base_k, base_v) = baseline_stable_sort(&inds, &vals);

        radix_sort_u32_soa(&mut inds, &mut vals);

        assert_sorted(&inds);
        assert_eq!(inds, base_k);
        assert_eq!(vals, base_v);
    }
}

