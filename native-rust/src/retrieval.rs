use anyhow::{bail, ensure, Context, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

/// Feature rows used for RVC index retrieval: a portable float32 bank or a FAISS `IndexIVFFlat`.
pub enum FeatureBank {
    Flat { channels: usize, rows: Vec<f32> },
    Ivf(IvfFlat),
}
impl FeatureBank {
    /// FAISS files are detected by their `IwFl` header; anything else is read as raw f32 rows.
    pub fn load(path: &Path, channels: usize) -> Result<Self> {
        let data =
            std::fs::read(path).with_context(|| format!("Load feature bank {}", path.display()))?;
        if data.starts_with(b"IwFl") {
            let index = IvfFlat::parse(&data).context("Parse FAISS index")?;
            ensure!(
                index.dim == channels,
                "FAISS index has {} dimensions but the model expects {channels}",
                index.dim
            );
            return Ok(Self::Ivf(index));
        }
        ensure!(
            !data.is_empty() && data.len() % (4 * channels) == 0,
            "Feature bank must be a FAISS IVF-Flat index or nonempty little-endian f32 rows of {channels} channels"
        );
        let rows = floats(&data);
        ensure!(
            rows.iter().all(|v| v.is_finite()),
            "Non-finite feature bank"
        );
        Ok(Self::Flat { channels, rows })
    }
    pub fn dim(&self) -> usize {
        match self {
            Self::Flat { channels, .. } => *channels,
            Self::Ivf(index) => index.dim,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Flat { channels, rows } => rows.len().checked_div(*channels).unwrap_or(0),
            Self::Ivf(index) => index.ntotal,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Stored row nearest to `query` by L2, i.e. `big_npy[index.search(q, 1)]` in the reference pipeline.
    pub fn nearest(&self, query: &[f32]) -> &[f32] {
        match self {
            Self::Flat { channels, rows } => rows
                .chunks_exact(*channels)
                .min_by(|a, b| l2(a, query).total_cmp(&l2(b, query)))
                .unwrap(),
            Self::Ivf(index) => {
                // FAISS reports id -1 when every probed list is empty; scan all lists instead.
                let (list, row) = index
                    .nearest(query, index.nprobe)
                    .or_else(|| index.nearest(query, index.ids.len()))
                    .unwrap();
                index.row(list, row)
            }
        }
    }
    /// Blend the `k` nearest feature rows to `query` by inverse squared L2 distance
    /// (`1.0 / (dist_sq + 1e-6)` normalized to sum to 1.0) and write the result into `out`.
    pub fn blend_nearest(&self, query: &[f32], k: usize, out: &mut [f32]) {
        let dim = self.dim();
        assert_eq!(
            query.len(),
            dim,
            "Query has {} features, expected {dim}",
            query.len()
        );
        assert_eq!(
            out.len(),
            dim,
            "Output slice has {} features, expected {dim}",
            out.len()
        );
        let len = self.len();
        if dim == 0 || len == 0 {
            out.fill(0.0);
            return;
        }
        let k = k.clamp(1, len);
        if k == 1 {
            out.copy_from_slice(self.nearest(query));
            return;
        }
        match self {
            Self::Flat { channels, rows } => {
                let mut heap = BinaryHeap::with_capacity(k);
                for (idx, row) in rows.chunks_exact(*channels).enumerate() {
                    push_top_k(&mut heap, k, l2(row, query), idx);
                }
                let top = heap.into_sorted_vec();
                accumulate_blend(
                    out,
                    top.iter()
                        .map(|c| (c.0, &rows[c.1 * *channels..][..*channels])),
                );
            }
            Self::Ivf(index) => {
                // FAISS reports id -1 when every probed list is empty; scan all lists instead.
                let mut top = index.nearest_k(query, index.nprobe, k);
                if top.is_empty() {
                    top = index.nearest_k(query, index.ids.len(), k);
                }
                accumulate_blend(
                    out,
                    top.iter()
                        .map(|&(dist, list, row)| (dist, index.row(list, row))),
                );
            }
        }
    }
}

/// FAISS `IndexIVFFlat` with an L2 flat quantizer and in-memory (`ilar`) inverted lists.
pub struct IvfFlat {
    pub dim: usize,
    pub ntotal: usize,
    pub nprobe: usize,
    centroids: Vec<f32>,
    codes: Vec<Vec<f32>>,
    ids: Vec<Vec<i64>>,
}
impl IvfFlat {
    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(&std::fs::read(path).with_context(|| format!("Read {}", path.display()))?)
    }
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut r = Reader { data, pos: 0 };
        r.fourcc(b"IwFl")?;
        let (dim, ntotal) = r.header()?;
        let nlist = r.size()?;
        let nprobe = r.size()?.clamp(1, nlist.max(1));
        r.fourcc(b"IxF2")
            .context("Only L2 flat quantizers are supported")?;
        let (qdim, qtotal) = r.header()?;
        ensure!(
            qdim == dim && qtotal == nlist,
            "Quantizer does not match the IVF header"
        );
        let count = r.size()?;
        ensure!(count == dim * nlist, "Unexpected quantizer size");
        let centroids = floats(r.take(count * 4)?);
        // Direct map: type byte, id array, and a hashtable of pairs when type == 2.
        let map_type = r.take(1)?[0];
        let entries = r.size()?;
        r.take(entries * 8)?;
        if map_type == 2 {
            let pairs = r.size()?;
            r.take(pairs * 16)?;
        }
        r.fourcc(b"ilar")
            .context("Only in-memory inverted lists are supported")?;
        ensure!(r.size()? == nlist, "Inverted list count mismatch");
        let code_size = r.size()?;
        ensure!(
            code_size == dim * 4,
            "Code size {code_size} is not float32 x {dim}"
        );
        let mut sizes = vec![0; nlist];
        match r.take(4)? {
            b"full" => {
                ensure!(r.size()? == nlist, "Inverted list size table mismatch");
                for size in &mut sizes {
                    *size = r.size()?;
                }
            }
            b"sprs" => {
                let pairs = r.size()?;
                ensure!(pairs % 2 == 0, "Malformed sparse inverted list table");
                for _ in 0..pairs / 2 {
                    let (list, size) = (r.size()?, r.size()?);
                    ensure!(list < nlist, "Inverted list number out of range");
                    sizes[list] = size;
                }
            }
            _ => bail!("Unknown inverted list layout"),
        }
        ensure!(
            sizes.iter().sum::<usize>() == ntotal,
            "Inverted lists do not hold ntotal vectors"
        );
        let mut codes = Vec::with_capacity(nlist);
        let mut ids = Vec::with_capacity(nlist);
        for size in sizes {
            codes.push(floats(r.take(size * code_size)?));
            ids.push(
                r.take(size * 8)?
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| i64::from_le_bytes(*v))
                    .collect(),
            );
        }
        ensure!(r.pos == data.len(), "Trailing bytes after FAISS index");
        ensure!(
            centroids
                .iter()
                .chain(codes.iter().flatten())
                .all(|v| v.is_finite()),
            "Non-finite FAISS index"
        );
        ensure!(ntotal > 0, "FAISS index is empty");
        Ok(Self {
            dim,
            ntotal,
            nprobe,
            centroids,
            codes,
            ids,
        })
    }
    fn probed_lists(&self, query: &[f32], probe: usize) -> Vec<usize> {
        if probe == 0 || self.centroids.is_empty() {
            return Vec::new();
        }
        let mut lists: Vec<(f32, usize)> = self
            .centroids
            .chunks_exact(self.dim)
            .map(|c| l2(c, query))
            .zip(0..)
            .collect();
        let probe = probe.min(lists.len());
        if probe < lists.len() {
            lists.select_nth_unstable_by(probe - 1, |a, b| a.0.total_cmp(&b.0));
        }
        lists[..probe].iter().map(|&(_, list)| list).collect()
    }
    /// (list, row) of the nearest stored vector among the `probe` closest lists.
    fn nearest(&self, query: &[f32], probe: usize) -> Option<(usize, usize)> {
        let mut best = None;
        let mut best_distance = f32::INFINITY;
        for list in self.probed_lists(query, probe) {
            for (row, code) in self.codes[list].chunks_exact(self.dim).enumerate() {
                let distance = l2(code, query);
                if distance < best_distance {
                    best_distance = distance;
                    best = Some((list, row));
                }
            }
        }
        best
    }
    /// Up to `k` `(squared L2 distance, list, row)` entries among the `probe` closest lists,
    /// ordered closest-first.
    fn nearest_k(&self, query: &[f32], probe: usize, k: usize) -> Vec<(f32, usize, usize)> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k);
        for list in self.probed_lists(query, probe) {
            for (row, code) in self.codes[list].chunks_exact(self.dim).enumerate() {
                push_top_k(&mut heap, k, l2(code, query), (list, row));
            }
        }
        heap.into_sorted_vec()
            .into_iter()
            .map(|Candidate(dist, (list, row))| (dist, list, row))
            .collect()
    }
    fn row(&self, list: usize, row: usize) -> &[f32] {
        &self.codes[list][row * self.dim..(row + 1) * self.dim]
    }
    /// FAISS `search(query, 1)`: (squared L2 distance, id), or None when probed lists are empty.
    pub fn search(&self, query: &[f32]) -> Option<(f32, i64)> {
        assert_eq!(query.len(), self.dim);
        let (list, row) = self.nearest(query, self.nprobe)?;
        Some((l2(self.row(list, row), query), self.ids[list][row]))
    }
    /// FAISS `search(query, k)`: up to `k` `(squared L2 distance, id)` pairs ordered closest-first.
    pub fn search_k(&self, query: &[f32], k: usize) -> Vec<(f32, i64)> {
        assert_eq!(query.len(), self.dim);
        self.nearest_k(query, self.nprobe, k)
            .into_iter()
            .map(|(dist, list, row)| (dist, self.ids[list][row]))
            .collect()
    }
    /// All stored vectors ordered by id, like `reconstruct_n(0, ntotal)` for sequentially added data.
    pub fn reconstruct_all(&self) -> Result<Vec<f32>> {
        let mut out = vec![f32::NAN; self.ntotal * self.dim];
        for (list, ids) in self.ids.iter().enumerate() {
            for (row, &id) in ids.iter().enumerate() {
                ensure!(
                    (0..self.ntotal as i64).contains(&id),
                    "Index ids are not 0..ntotal"
                );
                out[id as usize * self.dim..][..self.dim].copy_from_slice(self.row(list, row));
            }
        }
        ensure!(
            out.iter().all(|v| !v.is_nan()),
            "Index ids are not 0..ntotal"
        );
        Ok(out)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= self.data.len());
        let end = end.context("Truncated FAISS index")?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
    fn fourcc(&mut self, expected: &[u8; 4]) -> Result<()> {
        let found = self.take(4)?;
        ensure!(
            found == expected,
            "Expected FAISS {:?}, found {:?}",
            String::from_utf8_lossy(expected),
            String::from_utf8_lossy(found)
        );
        Ok(())
    }
    fn size(&mut self) -> Result<usize> {
        let value = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        usize::try_from(value)
            .ok()
            .filter(|v| *v <= self.data.len())
            .context("FAISS size out of range")
    }
    /// Common index header: d, ntotal, two reserved fields, is_trained, metric (L2 only).
    fn header(&mut self) -> Result<(usize, usize)> {
        let dim = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
        let ntotal = i64::from_le_bytes(self.take(8)?.try_into().unwrap());
        self.take(17)?;
        let metric = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
        ensure!(metric == 1, "Only METRIC_L2 FAISS indexes are supported");
        ensure!(dim > 0 && ntotal >= 0, "Invalid FAISS header");
        Ok((dim as usize, ntotal as usize))
    }
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a_chunks, a_rem) = a[..n].as_chunks::<8>();
    let (b_chunks, b_rem) = b[..n].as_chunks::<8>();
    // Eight independent accumulators break loop-carried fp add dependencies for SIMD.
    let mut acc = [0.0_f32; 8];
    for (ca, cb) in a_chunks.iter().zip(b_chunks) {
        for i in 0..8 {
            let d = ca[i] - cb[i];
            acc[i] += d * d;
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in a_rem.iter().zip(b_rem) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[derive(Clone, Copy)]
struct Candidate<T>(f32, T);

impl<T: Eq> PartialEq for Candidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq() && self.1 == other.1
    }
}

impl<T: Eq> Eq for Candidate<T> {}

impl<T: Ord> PartialOrd for Candidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for Candidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}

fn push_top_k<T: Ord>(heap: &mut BinaryHeap<Candidate<T>>, k: usize, dist: f32, item: T) {
    if heap.len() < k {
        heap.push(Candidate(dist, item));
    } else if let Some(worst_dist) = heap.peek().map(|c| c.0) {
        if dist.total_cmp(&worst_dist).is_lt() {
            *heap.peek_mut().unwrap() = Candidate(dist, item);
        }
    }
}

fn accumulate_blend<'a, I>(out: &mut [f32], items: I)
where
    I: IntoIterator<Item = (f32, &'a [f32])>,
    I::IntoIter: Clone,
{
    let iter = items.into_iter();
    let mut weight_sum = 0.0_f64;
    for (dist_sq, _) in iter.clone() {
        weight_sum += 1.0_f64 / (dist_sq.max(0.0) as f64 + 1e-6);
    }
    out.fill(0.0);
    if weight_sum == 0.0 || !weight_sum.is_finite() {
        let mut count = 0usize;
        for (_, row) in iter {
            for (o, &v) in out.iter_mut().zip(row) {
                *o += v;
            }
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
        return;
    }
    let inv_sum = 1.0_f64 / weight_sum;
    for (dist_sq, row) in iter {
        let w = ((1.0_f64 / (dist_sq.max(0.0) as f64 + 1e-6)) * inv_sum) as f32;
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn l2_matches_scalar_across_lengths() {
        for len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 64, 768] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.73).cos() * 1.5).collect();
            assert_eq!(l2(&a, &a), 0.0, "len={len}");
            let got = l2(&a, &b);
            let want = scalar_l2(&a, &b);
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "len={len}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn blend_nearest_distinct_weights_math() {
        let bank = FeatureBank::Flat {
            channels: 2,
            rows: vec![
                1.0, 0.0, // dist_sq = 1.0
                0.0, 2.0, // dist_sq = 4.0
                10.0, 10.0, // dist_sq = 200.0
            ],
        };
        let query = [0.0, 0.0];
        let mut out = [0.0; 2];
        bank.blend_nearest(&query, 2, &mut out);

        let w0 = 1.0_f64 / (1.0 + 1e-6);
        let w1 = 1.0_f64 / (4.0 + 1e-6);
        let expected_x = (w0 / (w0 + w1)) as f32;
        let expected_y = (2.0 * w1 / (w0 + w1)) as f32;
        assert!((out[0] - expected_x).abs() < 1e-6);
        assert!((out[1] - expected_y).abs() < 1e-6);
    }

    #[test]
    fn blend_nearest_exact_match_is_finite_and_dominant() {
        let bank = FeatureBank::Flat {
            channels: 3,
            rows: vec![
                1.0, -2.0, 3.0, // exact match
                2.0, 0.0, 3.0, // dist_sq = 5.0
            ],
        };
        let query = [1.0, -2.0, 3.0];
        let mut out = [0.0; 3];
        bank.blend_nearest(&query, 8, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        for (got, want) in out.iter().zip(query) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }

        // Duplicate exact matches average cleanly with no NaN/Inf.
        let dup_bank = FeatureBank::Flat {
            channels: 2,
            rows: vec![1.5, -0.5, 1.5, -0.5, 10.0, 10.0],
        };
        let mut dup_out = [0.0; 2];
        dup_bank.blend_nearest(&[1.5, -0.5], 3, &mut dup_out);
        assert!(dup_out.iter().all(|v| v.is_finite()));
        assert!((dup_out[0] - 1.5).abs() < 1e-5);
        assert!((dup_out[1] + 0.5).abs() < 1e-5);
    }

    #[test]
    fn blend_nearest_clamps_k_and_handles_empty_bank() {
        let bank = FeatureBank::Flat {
            channels: 2,
            rows: vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0],
        };
        let query = [0.1, 0.0];
        let mut out_k0 = [0.0; 2];
        bank.blend_nearest(&query, 0, &mut out_k0);
        assert_eq!(out_k0, bank.nearest(&query));

        let mut out_large_k = [0.0; 2];
        let mut out_len_k = [0.0; 2];
        bank.blend_nearest(&query, 100, &mut out_large_k);
        bank.blend_nearest(&query, bank.len(), &mut out_len_k);
        assert_eq!(out_large_k, out_len_k);

        let empty = FeatureBank::Flat {
            channels: 2,
            rows: Vec::new(),
        };
        let mut out_empty = [42.0; 2];
        empty.blend_nearest(&query, 8, &mut out_empty);
        assert_eq!(out_empty, [0.0, 0.0]);
    }

    #[test]
    fn blend_nearest_ivf_matches_flat_and_falls_back_on_empty_probe() {
        let row0 = [0.5, 0.0, 0.0, 0.0];
        let row1 = [1.0, 1.0, 0.0, 0.0];
        let row2 = [10.0, 0.5, 0.0, 0.0];
        let row3 = [10.0, 2.0, 0.0, 0.0];

        let mut all_rows = Vec::new();
        for r in [row0, row1, row2, row3] {
            all_rows.extend_from_slice(&r);
        }
        let flat = FeatureBank::Flat {
            channels: 4,
            rows: all_rows,
        };

        let ivf_full = FeatureBank::Ivf(IvfFlat {
            dim: 4,
            ntotal: 4,
            nprobe: 2,
            centroids: vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            codes: vec![[row0, row1].concat(), [row2, row3].concat()],
            ids: vec![vec![0, 1], vec![2, 3]],
        });

        let query = [0.2, 0.3, 0.0, 0.0];
        for k in [1, 2, 3, 8] {
            let mut out_flat = [0.0; 4];
            let mut out_ivf = [0.0; 4];
            flat.blend_nearest(&query, k, &mut out_flat);
            ivf_full.blend_nearest(&query, k, &mut out_ivf);
            for (a, b) in out_flat.iter().zip(out_ivf) {
                assert!((a - b).abs() < 1e-6, "k={k}: flat={a}, ivf={b}");
            }
        }

        if let FeatureBank::Ivf(ref index) = ivf_full {
            let hits = index.search_k(&query, 3);
            assert_eq!(hits.len(), 3);
            assert_eq!(hits[0].1, 0);
            assert!(hits.windows(2).all(|w| w[0].0 <= w[1].0));
        }

        // Probing list 0 (empty) must fall back to scanning all lists.
        let ivf_sparse = IvfFlat {
            dim: 4,
            ntotal: 2,
            nprobe: 1,
            centroids: vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            codes: vec![Vec::new(), [row2, row3].concat()],
            ids: vec![Vec::new(), vec![0, 1]],
        };
        assert!(ivf_sparse.search_k(&query, 2).is_empty());
        let flat_sparse = FeatureBank::Flat {
            channels: 4,
            rows: [row2, row3].concat(),
        };
        let ivf_sparse_bank = FeatureBank::Ivf(ivf_sparse);
        let mut out_ivf_sparse = [0.0; 4];
        let mut out_flat_sparse = [0.0; 4];
        ivf_sparse_bank.blend_nearest(&query, 8, &mut out_ivf_sparse);
        flat_sparse.blend_nearest(&query, 8, &mut out_flat_sparse);
        for (a, b) in out_ivf_sparse.iter().zip(out_flat_sparse) {
            assert!((a - b).abs() < 1e-6, "sparse fallback mismatch: {a} vs {b}");
        }
    }
}
