//! Text normalization and the cheap quality gate applied before a chunk is kept.
//!
//! Deliberately minimal — this is not trying to be a full quality pipeline (no
//! dedicated per-language handling, no code-aware formatting). It exists to catch the
//! same class of problem custom-gpt-10m's own `is_quality_text()` catches for its chat
//! corpus: extraction noise that would teach a model garbage rather than language, kept
//! cheap enough to run on every chunk of a large folder without becoming the bottleneck.

/// Collapse repeated whitespace and normalize line endings. Applied to raw extracted
/// text *before* chunking, so token boundaries in the chunker aren't computed against
/// noise (e.g. a PDF extractor's stray runs of spaces from column layout).
pub fn normalize_whitespace(text: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");

    let mut out = String::with_capacity(unified.len());
    let mut blank_run = 0u32;
    for line in unified.split('\n') {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run > 2 {
                continue; // collapse 3+ consecutive blank lines down to 2
            }
        } else {
            blank_run = 0;
        }
        out.push_str(trimmed);
        out.push('\n');
    }
    out.trim().to_string()
}

/// Whether a chunk is worth keeping: long enough, and not dominated by non-ASCII bytes
/// (a proxy for garbled extraction — real prose/code in this pipeline's target
/// extensions is overwhelmingly ASCII; a chunk that isn't is far more likely to be
/// extraction noise than legitimate non-English content, given this tool's fixed
/// extension list).
pub fn is_quality_text(text: &str, min_chars: usize, min_ascii_ratio: f64) -> bool {
    let char_count = text.chars().count();
    if char_count < min_chars {
        return false;
    }
    if char_count == 0 {
        return false;
    }
    let ascii_count = text.chars().filter(|c| c.is_ascii()).count();
    let ascii_ratio = ascii_count as f64 / char_count as f64;
    if ascii_ratio < min_ascii_ratio {
        return false;
    }
    // Require *some* alphabetic content — filters chunks that are pure punctuation/
    // table-of-contents dot-leaders/page-number noise, a common PDF-extraction artifact.
    let alpha_count = text.chars().filter(|c| c.is_alphabetic()).count();
    alpha_count >= (char_count / 20).max(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_excess_blank_lines() {
        let input = "a\n\n\n\n\nb";
        let out = normalize_whitespace(input);
        assert_eq!(out, "a\n\n\nb");
    }

    #[test]
    fn rejects_short_text() {
        assert!(!is_quality_text("hi", 40, 0.5));
    }

    #[test]
    fn rejects_mostly_non_ascii() {
        let text = "文".repeat(50);
        assert!(!is_quality_text(&text, 10, 0.5));
    }

    #[test]
    fn accepts_normal_prose() {
        let text = "This is a perfectly ordinary sentence about nothing in particular.";
        assert!(is_quality_text(text, 10, 0.5));
    }
}
