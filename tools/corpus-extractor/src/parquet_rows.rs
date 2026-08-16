//! Parse HF dataset parquet shards into `(Role, text)` conversations — the Rust
//! counterpart to `prepare.py`'s `extract_turns_conversation` / `extract_turns_instruction`
//! / `load_oasst_conversations`. Reads `RecordBatch`es columnarly via `parquet`'s arrow
//! integration rather than row-by-row like Python's `datasets` library does — this is
//! the real throughput win, not just "no GIL".

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, RecordBatch, StringArray, StructArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::corpus_text::{self, Role};
use crate::sources::Schema;

fn find_column(batch: &RecordBatch, names: &[&str]) -> Option<ArrayRef> {
    for name in names {
        if let Ok(idx) = batch.schema().index_of(name) {
            return Some(batch.column(idx).clone());
        }
    }
    None
}

fn utf8_value(arr: &dyn Array, idx: usize) -> Option<String> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return Some(a.value(idx).to_string());
    }
    if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
        return Some(a.value(idx).to_string());
    }
    None
}

fn int_value(arr: &dyn Array, idx: usize) -> Option<i64> {
    if arr.is_null(idx) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(a.value(idx));
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(a.value(idx) as i64);
    }
    None
}

fn list_elements(col: &ArrayRef, row: usize) -> Option<ArrayRef> {
    if col.is_null(row) {
        return None;
    }
    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        return Some(list.value(row));
    }
    if let Some(list) = col.as_any().downcast_ref::<LargeListArray>() {
        return Some(list.value(row));
    }
    None
}

/// Schema: a list of {role/from, content/value} structs (UltraChat, SmolTalk, No
/// Robots, LMSYS) — direct port of `extract_turns_conversation`.
fn conversation_turns(batch: &RecordBatch, row: usize, min_chars: usize, min_ascii_ratio: f64) -> Vec<(Role, String)> {
    let Some(col) = find_column(batch, &["conversation", "conversations", "messages"]) else {
        return Vec::new();
    };
    let Some(elements) = list_elements(&col, row) else {
        return Vec::new();
    };
    let Some(structs) = elements.as_any().downcast_ref::<StructArray>() else {
        return Vec::new();
    };

    let role_col = structs.column_by_name("role").or_else(|| structs.column_by_name("from"));
    let content_col = structs.column_by_name("content").or_else(|| structs.column_by_name("value"));
    let (Some(role_col), Some(content_col)) = (role_col, content_col) else {
        return Vec::new();
    };

    let mut turns = Vec::new();
    for i in 0..structs.len() {
        let Some(raw_role) = utf8_value(role_col.as_ref(), i) else { continue };
        let Some(role) = corpus_text::normalize_role(&raw_role) else { continue };
        let Some(raw_text) = utf8_value(content_col.as_ref(), i) else { continue };
        let text = corpus_text::clean_text(&raw_text);
        if text.is_empty() || !corpus_text::is_quality_conversation_text(&text, min_chars, min_ascii_ratio) {
            continue;
        }
        turns.push((role, text));
    }
    turns
}

/// Schema: flat instruction/input/output columns (Dolly, GSM8K) — direct port of
/// `extract_turns_instruction`.
fn instruction_turns(batch: &RecordBatch, row: usize, min_chars: usize, min_ascii_ratio: f64) -> Vec<(Role, String)> {
    let Some(instruction_col) = find_column(batch, &["instruction", "prompt", "question"]) else {
        return Vec::new();
    };
    let Some(output_col) = find_column(batch, &["output", "response", "answer", "completion"]) else {
        return Vec::new();
    };
    let input_col = find_column(batch, &["input"]);

    let Some(instruction) = utf8_value(instruction_col.as_ref(), row) else { return Vec::new() };
    let Some(output) = utf8_value(output_col.as_ref(), row) else { return Vec::new() };

    let mut user_text = corpus_text::clean_text(&instruction);
    if let Some(input_text) = input_col.and_then(|c| utf8_value(c.as_ref(), row)) {
        if !input_text.is_empty() {
            user_text = format!("{}\n{}", user_text, corpus_text::clean_text(&input_text));
        }
    }
    let assistant_text = corpus_text::clean_text(&output);

    if !corpus_text::is_quality_conversation_text(&user_text, min_chars, min_ascii_ratio) {
        return Vec::new();
    }
    if !corpus_text::is_quality_conversation_text(&assistant_text, min_chars, min_ascii_ratio) {
        return Vec::new();
    }
    vec![(Role::User, user_text), (Role::Assistant, assistant_text)]
}

fn keeps_conversation(turns: &[(Role, String)], min_turns: usize) -> bool {
    if turns.len() < min_turns {
        return false;
    }
    let roles: HashSet<Role> = turns.iter().map(|(r, _)| *r).collect();
    roles.contains(&Role::Assistant) && roles.contains(&Role::User)
}

fn rows_from_file(
    path: &Path,
    schema: Schema,
    remaining_cap: usize,
    min_turns: usize,
    min_chars: usize,
    min_ascii_ratio: f64,
) -> Result<Vec<Vec<(Role, String)>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata for {}", path.display()))?
        .build()
        .with_context(|| format!("building arrow reader for {}", path.display()))?;

    let mut collected = Vec::new();
    for batch_result in reader {
        if collected.len() >= remaining_cap {
            break;
        }
        let batch = batch_result.with_context(|| format!("reading a batch from {}", path.display()))?;
        for row in 0..batch.num_rows() {
            if collected.len() >= remaining_cap {
                break;
            }
            let turns = match schema {
                Schema::Conversation => conversation_turns(&batch, row, min_chars, min_ascii_ratio),
                Schema::Instruction => instruction_turns(&batch, row, min_chars, min_ascii_ratio),
                Schema::OasstTree => unreachable!("OASST's reply-tree schema uses load_oasst_conversations, not this generic loader"),
            };
            if keeps_conversation(&turns, min_turns) {
                collected.push(turns);
            }
        }
    }
    Ok(collected)
}

/// Parse parquet shards into conversations, stopping once `cap` are collected —
/// direct port of `prepare.py::load_conversations()`. A shard that fails to parse is
/// skipped with a warning rather than aborting the whole source, matching Python's
/// `except Exception` -> `[warn] skipping unreadable parquet file ...` behavior.
pub fn load_rows(
    files: &[PathBuf],
    schema: Schema,
    cap: usize,
    min_turns: usize,
    min_chars: usize,
    min_ascii_ratio: f64,
) -> Vec<Vec<(Role, String)>> {
    let mut collected = Vec::new();
    for path in files {
        if collected.len() >= cap {
            break;
        }
        let remaining = cap - collected.len();
        match rows_from_file(path, schema, remaining, min_turns, min_chars, min_ascii_ratio) {
            Ok(rows) => collected.extend(rows),
            Err(err) => eprintln!("[warn] skipping unreadable parquet file {}: {err:#}", path.display()),
        }
    }
    collected
}

const OASST_ENGLISH: &str = "en";

fn oasst_role(raw: &str) -> Option<Role> {
    match raw {
        "prompter" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        _ => None,
    }
}

struct OasstMessage {
    role: Option<Role>,
    text: String,
    rank: Option<i64>,
}

/// OASST1-specific: its raw schema is a flat table of individual messages forming
/// reply trees (message_id/parent_id/rank), not a pre-assembled conversation — direct
/// port of `prepare.py::load_oasst_conversations()`. Reconstructs one linear
/// conversation per tree by walking from each root message, picking the best-ranked
/// reply at each level (rank 0 = highest quality) rather than an arbitrary child.
/// English-only by default, matching the rest of this corpus/tokenizer's focus.
fn oasst_conversations_from_file(
    path: &Path,
    remaining_cap: usize,
    min_turns: usize,
    min_chars: usize,
    min_ascii_ratio: f64,
) -> Result<Vec<Vec<(Role, String)>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata for {}", path.display()))?
        .build()
        .with_context(|| format!("building arrow reader for {}", path.display()))?;

    let mut by_id: HashMap<String, OasstMessage> = HashMap::new();
    let mut children: HashMap<Option<String>, Vec<String>> = HashMap::new();

    for batch_result in reader {
        let batch = batch_result.with_context(|| format!("reading a batch from {}", path.display()))?;
        let message_id_col = find_column(&batch, &["message_id"]);
        let parent_id_col = find_column(&batch, &["parent_id"]);
        let role_col = find_column(&batch, &["role"]);
        let text_col = find_column(&batch, &["text"]);
        let lang_col = find_column(&batch, &["lang"]);
        let rank_col = find_column(&batch, &["rank"]);
        let (Some(message_id_col), Some(role_col), Some(text_col)) = (&message_id_col, &role_col, &text_col) else {
            continue;
        };

        for i in 0..batch.num_rows() {
            if let Some(lang_col) = &lang_col {
                if utf8_value(lang_col.as_ref(), i).as_deref() != Some(OASST_ENGLISH) {
                    continue;
                }
            }
            let Some(message_id) = utf8_value(message_id_col.as_ref(), i) else { continue };
            let parent_id = parent_id_col.as_ref().and_then(|c| utf8_value(c.as_ref(), i));
            let role = utf8_value(role_col.as_ref(), i).as_deref().and_then(oasst_role);
            let text = utf8_value(text_col.as_ref(), i)
                .map(|t| corpus_text::clean_text(&t))
                .unwrap_or_default();
            let rank = rank_col.as_ref().and_then(|c| int_value(c.as_ref(), i));

            children.entry(parent_id).or_default().push(message_id.clone());
            by_id.insert(message_id, OasstMessage { role, text, rank });
        }
    }

    // Best-ranked (rank 0) reply first at every branch point; unranked replies sort last.
    for kids in children.values_mut() {
        kids.sort_by_key(|id| by_id.get(id).and_then(|m| m.rank).unwrap_or(10_000));
    }

    let mut collected = Vec::new();
    for root_id in children.get(&None).cloned().unwrap_or_default() {
        if collected.len() >= remaining_cap {
            break;
        }
        let mut turns = Vec::new();
        let mut node_id = Some(root_id);
        while let Some(id) = node_id {
            let Some(message) = by_id.get(&id) else { break };
            if let Some(role) = message.role {
                if !message.text.is_empty()
                    && corpus_text::is_quality_conversation_text(&message.text, min_chars, min_ascii_ratio)
                {
                    turns.push((role, message.text.clone()));
                }
            }
            node_id = children.get(&Some(id.clone())).and_then(|kids| kids.first().cloned());
        }
        if keeps_conversation(&turns, min_turns) {
            collected.push(turns);
        }
    }
    Ok(collected)
}

/// Per-shard reply-tree reconstruction, capped and skip-on-error exactly like
/// [`load_rows`] — kept as a separate entry point (rather than a `Schema::OasstTree`
/// branch inside `rows_from_file`) since the tree walk needs the whole file's rows in
/// scope at once, not a per-row dispatch.
pub fn load_oasst_conversations(
    files: &[PathBuf],
    cap: usize,
    min_turns: usize,
    min_chars: usize,
    min_ascii_ratio: f64,
) -> Vec<Vec<(Role, String)>> {
    let mut collected = Vec::new();
    for path in files {
        if collected.len() >= cap {
            break;
        }
        let remaining = cap - collected.len();
        match oasst_conversations_from_file(path, remaining, min_turns, min_chars, min_ascii_ratio) {
            Ok(rows) => collected.extend(rows),
            Err(err) => eprintln!("[warn] skipping unreadable parquet file {}: {err:#}", path.display()),
        }
    }
    collected
}
