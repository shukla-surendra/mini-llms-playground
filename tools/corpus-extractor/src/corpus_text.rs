//! Turn normalization, cleaning, and quality filtering for chat-conversation text —
//! the `build-corpus` counterpart to `clean.rs`'s simpler length+ASCII filter used by
//! `extract`. Kept as its own module rather than folded into `clean.rs` because the
//! two filters serve different pipelines with different tolerances: `extract`'s filter
//! is a cheap proxy against garbled PDF/HTML extraction, while this one mirrors
//! custom-gpt-153m's `prepare.py::is_quality_text()` — printable-ratio, alphabetic
//! density, and redacted-placeholder rejection tuned for HF chat/instruction datasets,
//! not local-file extraction noise.

use std::collections::HashMap;
use std::sync::LazyLock;

/// Role a turn is attributed to in the normalized (role, text) pipeline shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
        }
    }
}

static ROLE_MAP: LazyLock<HashMap<&'static str, Role>> = LazyLock::new(|| {
    HashMap::from([
        ("user", Role::User),
        ("assistant", Role::Assistant),
        ("system", Role::System),
        ("human", Role::User),
        ("gpt", Role::Assistant),
    ])
});

/// Some sources (notably LMSYS) redact entities into placeholders like NAME_1. Those
/// leak into generations as nonsense, so drop any turn containing them.
const PLACEHOLDER_PATTERNS: [&str; 4] = ["NAME_", "PERSON_", "EMAIL_", "URL_"];

/// Normalize a raw role string (any case/whitespace) to one of the three roles this
/// pipeline keeps. Returns `None` for anything unrecognized (e.g. a schema's own
/// internal role labels this project doesn't train on).
pub fn normalize_role(raw: &str) -> Option<Role> {
    ROLE_MAP.get(raw.trim().to_lowercase().as_str()).copied()
}

/// CRLF/CR -> LF, drop the U+FFFD replacement character, collapse runs of
/// spaces/tabs to one space and runs of 3+ newlines down to exactly 2 (one blank line
/// max), trim ends — direct port of `prepare.py::clean_text()`'s two regex
/// substitutions (`[ \t]+` -> " ", `\n{3,}` -> "\n\n").
pub fn clean_text(text: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");
    let no_replacement = unified.replace('\u{FFFD}', " ");

    let mut collapsed_spaces = String::with_capacity(no_replacement.len());
    let mut last_was_hspace = false;
    for ch in no_replacement.chars() {
        if ch == ' ' || ch == '\t' {
            if !last_was_hspace {
                collapsed_spaces.push(' ');
            }
            last_was_hspace = true;
        } else {
            collapsed_spaces.push(ch);
            last_was_hspace = false;
        }
    }

    let mut out = String::with_capacity(collapsed_spaces.len());
    let mut newline_run = 0usize;
    for ch in collapsed_spaces.chars() {
        if ch == '\n' {
            newline_run += 1;
            continue;
        }
        if newline_run > 0 {
            out.push_str(&"\n".repeat(newline_run.min(2)));
            newline_run = 0;
        }
        out.push(ch);
    }
    if newline_run > 0 {
        out.push_str(&"\n".repeat(newline_run.min(2)));
    }

    out.trim().to_string()
}

/// Reject turns that would teach the model noise rather than language — direct port of
/// `prepare.py::is_quality_text()`. Deliberately stricter than `clean::is_quality_text`
/// (extract's filter): chat/instruction datasets carry different failure modes
/// (redacted placeholders, markdown-fence spam) than local PDF/HTML extraction noise.
pub fn is_quality_conversation_text(text: &str, min_chars: usize, min_ascii_ratio: f64) -> bool {
    let char_count = text.chars().count();
    if char_count < min_chars || char_count == 0 {
        return false;
    }

    let printable_count = text
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .count();
    if (printable_count as f64) / (char_count as f64) < 0.98 {
        return false;
    }

    let ascii_count = text.chars().filter(|c| c.is_ascii()).count();
    if (ascii_count as f64) / (char_count as f64) < min_ascii_ratio {
        return false;
    }

    let alpha_count = text.chars().filter(|c| c.is_alphabetic()).count();
    if alpha_count < (char_count / 25).max(5) {
        return false;
    }

    if PLACEHOLDER_PATTERNS.iter().any(|p| text.contains(p)) {
        return false;
    }

    if text.matches("```").count() > 2 {
        return false;
    }

    true
}

/// "Role: text" lines joined by "\n" — the flat document shape written to
/// train.txt/test.txt.
pub fn turns_to_text(turns: &[(Role, String)]) -> String {
    turns
        .iter()
        .map(|(role, text)| format!("{}: {text}", role.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a held-out prompt that stops right before the final assistant reply — direct
/// port of `prepare.py::prompt_from_turns()`.
pub fn prompt_from_turns(turns: &[(Role, String)]) -> Option<String> {
    if turns.len() < 2 {
        return None;
    }

    let last_assistant_idx = turns.iter().rposition(|(role, _)| *role == Role::Assistant);

    let prefix: &[(Role, String)] = match last_assistant_idx {
        Some(idx) if idx > 0 => &turns[..idx],
        _ => turns,
    };
    if prefix.is_empty() {
        return None;
    }
    Some(format!("{}\nAssistant:", turns_to_text(prefix)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_known_roles_case_insensitively() {
        assert_eq!(normalize_role("User"), Some(Role::User));
        assert_eq!(normalize_role("gpt"), Some(Role::Assistant));
        assert_eq!(normalize_role("human"), Some(Role::User));
        assert_eq!(normalize_role("narrator"), None);
    }

    #[test]
    fn clean_text_collapses_whitespace_and_blank_runs() {
        assert_eq!(clean_text("a\r\nb\r\r\nc"), "a\nb\n\nc");
        assert_eq!(clean_text("a\n\n\n\n\nb"), "a\n\nb");
        assert_eq!(clean_text("a\n\nb"), "a\n\nb");
        assert_eq!(clean_text("a   b\t\tc"), "a b c");
    }

    #[test]
    fn quality_filter_rejects_placeholders_and_short_text() {
        assert!(!is_quality_conversation_text("hi", 24, 0.995));
        assert!(!is_quality_conversation_text(
            "Contact NAME_1 for details please thanks",
            10,
            0.995
        ));
        assert!(is_quality_conversation_text(
            "This is a perfectly ordinary sentence about nothing in particular.",
            10,
            0.995
        ));
    }

    #[test]
    fn quality_filter_rejects_excess_code_fences() {
        let text = "```one``` and ```two``` and ```three``` are too many fences here";
        assert!(!is_quality_conversation_text(text, 10, 0.995));
    }

    #[test]
    fn prompt_from_turns_stops_before_final_assistant_reply() {
        let turns = vec![
            (Role::User, "hi".to_string()),
            (Role::Assistant, "hello".to_string()),
            (Role::User, "how are you".to_string()),
            (Role::Assistant, "good".to_string()),
        ];
        let prompt = prompt_from_turns(&turns).unwrap();
        assert_eq!(prompt, "User: hi\nAssistant: hello\nUser: how are you\nAssistant:");
    }

    #[test]
    fn prompt_from_turns_needs_at_least_two_turns() {
        assert!(prompt_from_turns(&[(Role::User, "hi".to_string())]).is_none());
    }
}
