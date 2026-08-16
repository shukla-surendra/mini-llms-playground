//! Fetch one dataset's parquet shards from the Hugging Face Hub — the Rust
//! counterpart to `prepare.py::download_source()`.
//!
//! Not every dataset repo publishes parquet on its main branch — some (e.g.
//! databricks/databricks-dolly-15k) only ever shipped a raw .jsonl file. Hugging Face
//! auto-converts every public dataset to parquet anyway, published on a standard
//! `refs/convert/parquet` ref regardless of the source repo's native format — falling
//! back to that ref when the main branch has no parquet means this pipeline can ingest
//! any public HF dataset uniformly.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::HFClientSync;

use crate::sources::DatasetSource;
use crate::walk;

/// Download `source`'s parquet shards into `<raw_root>/<slug>/`, preserving the
/// repository's own path structure (`local_dir` semantics — see hf-hub's
/// `snapshot_download` docs). Returns the list of parquet files found there once done.
///
/// A real error on the first (main-branch) attempt propagates immediately — matching
/// `prepare.py`, where a raised exception during the first `attempt()` call is never
/// caught inside `download_source()` itself, only one level up by the caller. Only a
/// *successful* attempt that yields zero parquet files triggers the
/// `refs/convert/parquet` retry.
pub fn download_source(
    client: &HFClientSync,
    source: &DatasetSource,
    raw_root: &Path,
    mut log: impl FnMut(String),
) -> Result<Vec<PathBuf>> {
    let (owner, name) = hf_hub::split_id(source.hf_id);
    let repo = client.dataset(owner, name);
    let out_dir = raw_root.join(source.slug());
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;

    fn attempt(
        repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeDataset>,
        hf_id: &str,
        out_dir: &Path,
        revision: Option<String>,
        log: &mut dyn FnMut(String),
    ) -> Result<()> {
        log(format!(
            "[download] {} (revision={}) -> {}",
            hf_id,
            revision.as_deref().unwrap_or("main"),
            out_dir.display()
        ));
        repo.snapshot_download()
            .allow_patterns(vec!["*.parquet".to_string()])
            .local_dir(out_dir.to_path_buf())
            .maybe_revision(revision)
            .send()
            .with_context(|| format!("downloading {hf_id}"))?;
        Ok(())
    }

    attempt(&repo, source.hf_id, &out_dir, None, &mut log)?;
    let mut files = parquet_files(&out_dir);
    if files.is_empty() {
        log(format!(
            "[info] no parquet on main for {}; trying refs/convert/parquet",
            source.hf_id
        ));
        attempt(&repo, source.hf_id, &out_dir, Some("refs/convert/parquet".to_string()), &mut log)?;
        files = parquet_files(&out_dir);
    }
    Ok(files)
}

/// Every `.parquet` file already on disk under `dir` — used both after a fresh
/// download and for `--skip-download` (reuse whatever's already there).
pub fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut files = walk::find_files(dir, &["parquet".to_string()].into_iter().collect());
    files.sort();
    files
}
