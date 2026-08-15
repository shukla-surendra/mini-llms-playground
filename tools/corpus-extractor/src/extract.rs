//! Turn one file, of a known type, into raw extracted text.
//!
//! Deliberately one function per format rather than a single generic "read file" —
//! each format fails differently (PDFs can be scanned images with no text layer at all,
//! HTML needs tag-stripping, source files can have non-UTF-8 bytes from an odd editor),
//! and keeping them separate means a PDF-specific failure can't silently degrade how
//! plain-text files are read.

use std::fs;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use epub::doc::EpubDoc;

/// Plain-text-shaped files: read as UTF-8, falling back to lossy conversion rather than
/// erroring out — a source file with one stray non-UTF-8 byte (common in older code with
/// mixed encodings in a comment) shouldn't sink the entire file's otherwise-good content.
pub fn extract_plain_text(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// HTML: strip markup down to readable text. A large fixed wrap width is used because
/// this output gets re-chunked by token count downstream anyway (see chunk.rs) — the
/// intermediate line-wrapping html2text applies is irrelevant, not a formatting choice
/// that needs to be right.
pub fn extract_html(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(html2text::from_read(bytes.as_slice(), 10_000))
}

/// PDF: extract whatever text layer exists. Scanned/image-only PDFs have no text layer
/// at all and will correctly extract to an empty (or near-empty) string — this is not a
/// bug to work around here; it's a real limitation of text-layer extraction (no OCR),
/// documented in the README rather than silently masked.
pub fn extract_pdf(path: &Path) -> Result<String> {
    pdf_extract::extract_text(path).with_context(|| format!("extracting PDF text from {}", path.display()))
}

/// EPUB: each chapter is XHTML inside a zip container, ordered by the book's own spine
/// (its table of contents order, not filename order — `EpubDoc` reads this from the
/// package's OPF manifest). Every chapter is run through the same `html2text` conversion
/// `extract_html` uses, then joined — an EPUB is, mechanically, just HTML with a
/// reading-order index on top, so it reuses that path rather than a separate parser.
pub fn extract_epub(path: &Path) -> Result<String> {
    let mut doc = EpubDoc::new(path)
        .with_context(|| format!("opening EPUB {}", path.display()))?;

    let mut out = String::new();
    let spine_len = doc.spine.len();
    for i in 0..spine_len {
        // get_current_str() reads whatever chapter go_next()/the initial position points
        // at; content is (String, mime type) — only the text matters here.
        if let Some((content, _mime)) = doc.get_current_str() {
            let chapter_text = html2text::from_read(content.as_bytes(), 10_000);
            if !chapter_text.trim().is_empty() {
                let _ = write!(out, "{chapter_text}\n\n");
            }
        }
        if i + 1 < spine_len {
            doc.go_next();
        }
    }
    Ok(out)
}

/// Dispatch by extension. Returns `Ok(None)` for an extension this module doesn't know
/// how to handle (should not happen given `walk.rs` already filtered by extension, but
/// kept explicit rather than panicking on an unreachable case).
pub fn extract(path: &Path) -> Result<Option<String>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let text = match ext.as_deref() {
        Some("pdf") => extract_pdf(path)?,
        Some("epub") => extract_epub(path)?,
        Some("html" | "htm") => extract_html(path)?,
        Some("txt" | "md" | "rs" | "js" | "py") => extract_plain_text(path)?,
        _ => return Ok(None),
    };
    Ok(Some(text))
}
