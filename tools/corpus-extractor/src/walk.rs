//! Recursive directory scan, filtered to the requested extensions.
//!
//! Uses the `ignore` crate (the same walker ripgrep is built on) rather than a plain
//! recursive walk, specifically so build/dependency noise gets skipped automatically —
//! pointed at a real software project (this tool's primary target, given .rs/.js/.py are
//! in its default extension list), a naive walk picks up `target/`, `node_modules/`,
//! `.venv/`, and similar generated/vendored trees, which are not source material for a
//! training corpus and can vastly outnumber and duplicate the real source. `.gitignore`/
//! `.ignore` rules are honored even when `root` isn't itself a git repository
//! (`require_git(false)`), and hidden files/directories (`.git`, dotfiles) are skipped by
//! default, matching what a developer scanning their own project would expect.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// Parse a comma-separated extension list ("pdf,txt,md") into a lookup set.
pub fn parse_extensions(raw: &str) -> HashSet<String> {
    raw.split(',')
        .map(|s| s.trim().trim_start_matches('.').to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Every file under `root` (recursively) whose extension is in `extensions`, skipping
/// anything `.gitignore`-style rules would exclude. Silently skips entries it can't read
/// (permission errors, broken symlinks) rather than aborting the whole scan over one bad
/// entry.
pub fn find_files(root: &Path, extensions: &HashSet<String>) -> Vec<PathBuf> {
    WalkBuilder::new(root)
        .require_git(false)
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.contains(&ext.to_lowercase()))
                .unwrap_or(false)
        })
        .collect()
}
