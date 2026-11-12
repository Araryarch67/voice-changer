use anyhow::{bail, ensure, Context, Result};
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
    pub fn len(&self) -> usize {
        match self {
            Self::Flat { channels, rows } => rows.len() / channels,
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
    /// (list, row) of the nearest stored vector among the `probe` closest lists.
    fn nearest(&self, query: &[f32], probe: usize) -> Option<(usize, usize)> {
        let mut lists: Vec<(f32, usize)> = self
            .centroids
            .chunks_exact(self.dim)
            .map(|c| l2(c, query))
            .zip(0..)
            .collect();
        let probe = probe.min(lists.len());
        lists.select_nth_unstable_by(probe - 1, |a, b| a.0.total_cmp(&b.0));
        let mut best = None;
        let mut best_distance = f32::INFINITY;
        for &(_, list) in &lists[..probe] {
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
    fn row(&self, list: usize, row: usize) -> &[f32] {
        &self.codes[list][row * self.dim..(row + 1) * self.dim]
    }
    /// FAISS `search(query, 1)`: (squared L2 distance, id), or None when probed lists are empty.
    pub fn search(&self, query: &[f32]) -> Option<(f32, i64)> {
        assert_eq!(query.len(), self.dim);
        let (list, row) = self.nearest(query, self.nprobe)?;
        Some((l2(self.row(list, row), query), self.ids[list][row]))
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
    a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
}
