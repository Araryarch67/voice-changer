use std::path::{Path, PathBuf};
use voice_changer::retrieval::{FeatureBank, IvfFlat};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Expected results were produced by `faiss.read_index(...).search(q, 1)` with FAISS 1.7.3.
fn expected() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(fixture("ivf-expected.json")).unwrap()).unwrap()
}

fn queries(expected: &serde_json::Value) -> Vec<Vec<f32>> {
    serde_json::from_value(expected["queries"].clone()).unwrap()
}

#[test]
fn ivf_search_matches_faiss_for_full_and_sparse_lists() {
    let expected = expected();
    for name in ["ivf-nprobe1", "ivf-nprobe3", "ivf-sparse"] {
        let index = IvfFlat::load(&fixture(&format!("{name}.index"))).unwrap();
        let ids: Vec<i64> = serde_json::from_value(expected[name]["ids"].clone()).unwrap();
        let distances: Vec<f32> =
            serde_json::from_value(expected[name]["distances"].clone()).unwrap();
        for ((query, id), distance) in queries(&expected).iter().zip(ids).zip(distances) {
            // FAISS reports id -1 when all probed lists are empty.
            let Some((found_distance, found_id)) = index.search(query) else {
                assert_eq!(id, -1, "{name}");
                continue;
            };
            assert_eq!(found_id, id, "{name}");
            assert!((found_distance - distance).abs() < 1e-4, "{name}");
        }
    }
}

#[test]
fn ivf_reconstruction_matches_added_rows_and_bank_returns_stored_rows() {
    let base = std::fs::read(fixture("ivf-base.bank")).unwrap();
    let base: Vec<f32> = base
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let index = IvfFlat::load(&fixture("ivf-nprobe3.index")).unwrap();
    assert_eq!((index.dim, index.ntotal, index.nprobe), (8, 200, 3));
    assert_eq!(index.reconstruct_all().unwrap(), base);

    let ivf = FeatureBank::load(&fixture("ivf-nprobe1.index"), 8).unwrap();
    let flat = FeatureBank::load(&fixture("ivf-base.bank"), 8).unwrap();
    let sparse = FeatureBank::load(&fixture("ivf-sparse.index"), 8).unwrap();
    for query in queries(&expected()) {
        assert!(base[..24]
            .as_chunks::<8>()
            .0
            .iter()
            .any(|row| row == sparse.nearest(&query)));
    }
    assert_eq!((ivf.len(), flat.len()), (200, 200));
    for row in base.as_chunks::<8>().0 {
        assert_eq!(ivf.nearest(row), row);
        assert_eq!(flat.nearest(row), row);
    }
}

#[test]
fn malformed_or_mismatched_indexes_are_rejected() {
    assert!(FeatureBank::load(&fixture("ivf-nprobe1.index"), 768).is_err());
    let data = std::fs::read(fixture("ivf-nprobe1.index")).unwrap();
    assert!(IvfFlat::parse(&data[..data.len() - 1]).is_err());
    let mut extra = data.clone();
    extra.push(0);
    assert!(IvfFlat::parse(&extra).is_err());
    let mut inner_product = data;
    inner_product[33..37].copy_from_slice(&0_i32.to_le_bytes());
    assert!(IvfFlat::parse(&inner_product).is_err());
}

/// Uses the RVC training indexes in `../voice-rvc` when present (not distributed with the crate).
/// Expected sums come from FAISS `reconstruct_n(0, ntotal)`.
#[test]
#[ignore = "requires ../voice-rvc kobo and zeta indexes"]
fn real_rvc_indexes_load_and_match_faiss_reconstruction() {
    for (file, ntotal, expected_sum) in [
        (
            "kobo/added_IVF454_Flat_nprobe_1_kobov2_v2.index",
            17711,
            -120977.13525491953,
        ),
        (
            "zeta/added_IVF462_Flat_nprobe_1_zetaTest_v2.index",
            18033,
            -111998.11656492949,
        ),
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../voice-rvc")
            .join(file);
        let index = IvfFlat::load(&path).unwrap();
        assert_eq!((index.dim, index.ntotal, index.nprobe), (768, ntotal, 1));
        let rows = index.reconstruct_all().unwrap();
        let sum: f64 = rows.iter().map(|v| *v as f64).sum();
        assert!((sum - expected_sum).abs() < 1e-3, "{file}");
        for id in [0, 1516, 9000, ntotal - 1] {
            let (distance, _) = index.search(&rows[id * 768..][..768]).unwrap();
            assert_eq!(distance, 0.0, "{file}");
        }
    }
}
