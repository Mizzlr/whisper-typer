//! Offline spelling cleanup. SymSpell proposes one-edit candidates; Hunspell
//! validates word forms. Only unambiguous duplicated-letter deletions are
//! applied automatically. Other spelling/grammar decisions need context.

use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;

use regex::Regex;
use serde::{Deserialize, Serialize};
use spellbook::Dictionary;
use symspell::{SymSpell, SymSpellBuilder, UnicodeStringStrategy, Verbosity};
use tracing::{info, warn};

use crate::config::SpellingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellingEdit {
    /// Byte offsets in the input to this cleanup pass.
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub replacement: String,
}

pub struct SpellingResult<'a> {
    pub text: Cow<'a, str>,
    pub edits: Vec<SpellingEdit>,
}

pub struct SpellCorrector {
    index: SymSpell<UnicodeStringStrategy>,
    dictionary: Dictionary,
    words: Regex,
    chunks: Regex,
    code: Regex,
}

impl SpellCorrector {
    /// Read/index data once at startup. Missing or malformed data disables this
    /// advisory pass rather than preventing dictation.
    pub fn load(config: &SpellingConfig) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let loaded = (|| {
            let aff = read(&config.aff_path)?;
            let dic = read(&config.dic_path)?;
            let frequency = if config.frequency_path.is_empty() {
                None
            } else {
                match read(&config.frequency_path) {
                    Ok(data) => Some(data),
                    Err(error) => {
                        warn!("Optional spelling frequencies not loaded: {error}");
                        None
                    }
                }
            };
            Self::from_data(&aff, &dic, frequency.as_deref())
        })();
        match loaded {
            Ok(corrector) => {
                info!("Offline spelling cleanup ready (SymSpell, edit distance 1, duplicated letters only)");
                Some(corrector)
            }
            Err(error) => {
                warn!("Offline spelling cleanup disabled: {error}");
                None
            }
        }
    }

    pub fn from_data(aff: &str, dic: &str, frequency: Option<&str>) -> Result<Self, String> {
        let dictionary = Dictionary::new(aff, dic).map_err(|error| error.to_string())?;
        let mut index = SymSpellBuilder::<UnicodeStringStrategy>::default()
            .max_dictionary_edit_distance(1)
            .build()
            .map_err(|error| error.to_string())?;
        // Supplement frequency data with valid system-dictionary stems, so
        // modern words such as "dataset" need no manually written mappings.
        for line in dic.lines().skip(1) {
            let stem = line
                .split_whitespace()
                .next()
                .unwrap_or("")
                .split('/')
                .next()
                .unwrap_or("");
            if stem.bytes().all(|byte| byte.is_ascii_lowercase())
                && !stem.is_empty()
                && dictionary.check(stem)
            {
                index.load_dictionary_line(&format!("{stem} 1"), 0, 1, " ");
            }
        }
        if let Some(frequency) = frequency {
            for line in frequency.lines() {
                let mut fields = line.split_whitespace();
                let (Some(word), Some(count)) = (fields.next(), fields.next()) else {
                    continue;
                };
                if fields.next().is_none()
                    && word.bytes().all(|byte| byte.is_ascii_lowercase())
                    && count.parse::<i64>().is_ok_and(|count| count > 0)
                    && dictionary.check(word)
                {
                    index.load_dictionary_line(&format!("{word} {count}"), 0, 1, " ");
                }
            }
        }
        Ok(Self {
            index,
            dictionary,
            words: Regex::new(r"\b[A-Za-z]+\b").unwrap(),
            chunks: Regex::new(r"\S+").unwrap(),
            // Preserve fenced/inline code even when its contents look English.
            code: Regex::new(r"(?s)```.*?(?:```|$)|`[^`]*(?:`|$)").unwrap(),
        })
    }

    pub fn apply<'a>(&self, text: &'a str, protectors: &[Regex]) -> SpellingResult<'a> {
        let mut protected = self
            .code
            .find_iter(text)
            .map(|m| (m.start(), m.end()))
            .collect::<Vec<_>>();
        protected.extend(
            protectors
                .iter()
                .flat_map(|regex| regex.find_iter(text).map(|m| (m.start(), m.end()))),
        );
        for chunk in self.chunks.find_iter(text) {
            let inner = chunk.as_str().trim_matches(|ch: char| {
                matches!(
                    ch,
                    '.' | ','
                        | '!'
                        | '?'
                        | ';'
                        | ':'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '"'
                        | '\''
                )
            });
            // Protect identifiers, addresses, flags, paths, hyphenated terms,
            // numbers/versions, and non-ASCII text without changing offsets.
            if inner.chars().any(|ch| {
                !ch.is_ascii()
                    || ch.is_ascii_digit()
                    || matches!(
                        ch,
                        '_' | '/' | '\\' | '@' | '.' | '-' | '=' | '+' | '$' | '#' | '\'' | '’'
                    )
            }) {
                protected.push((chunk.start(), chunk.end()));
            }
        }
        protected.sort_unstable();
        let mut protected_index = 0;
        let mut output: Option<String> = None;
        let mut copied = 0;
        let mut edits = Vec::new();
        for matched in self.words.find_iter(text) {
            while protected_index < protected.len()
                && protected[protected_index].1 <= matched.start()
            {
                protected_index += 1;
            }
            if protected
                .get(protected_index)
                .is_some_and(|&(start, end)| start < matched.end() && matched.start() < end)
            {
                continue;
            }
            let word = matched.as_str();
            // Lowercase or ordinary sentence capitalization only. Arbitrary
            // mixed-case words and acronyms may be technical identifiers.
            if word.len() < 4
                || word.len() > 64
                || !word.as_bytes()[1..].iter().all(u8::is_ascii_lowercase)
            {
                continue;
            }
            let lower = word.to_ascii_lowercase();
            if !lower.as_bytes().windows(2).any(|pair| pair[0] == pair[1])
                || self.dictionary.check(word)
                || self.dictionary.check(&lower)
            {
                continue;
            }
            let mut candidates = self
                .index
                .lookup(&lower, Verbosity::Closest, 1)
                .into_iter()
                .filter(|suggestion| {
                    suggestion.distance == 1
                        && is_duplicate_deletion(&lower, &suggestion.term)
                        && self.dictionary.check(&suggestion.term)
                        // Short rare words/abbreviations are easy accidental
                        // matches (e.g. "theool" -> "theol"). Defer them to
                        // context instead of changing them automatically.
                        && (suggestion.term.len() >= 6 || suggestion.count >= 100_000)
                });
            let Some(candidate) = candidates.next() else {
                continue;
            };
            if candidates.next().is_some() {
                continue;
            }
            let mut replacement = candidate.term;
            if word.as_bytes()[0].is_ascii_uppercase() {
                replacement[..1].make_ascii_uppercase();
            }
            let buffer = output.get_or_insert_with(|| String::with_capacity(text.len()));
            buffer.push_str(&text[copied..matched.start()]);
            buffer.push_str(&replacement);
            copied = matched.end();
            edits.push(SpellingEdit {
                start: matched.start(),
                end: matched.end(),
                original: word.into(),
                replacement,
            });
        }
        let text = match output {
            Some(mut buffer) => {
                buffer.push_str(&text[copied..]);
                Cow::Owned(buffer)
            }
            None => Cow::Borrowed(text),
        };
        SpellingResult { text, edits }
    }
}

fn is_duplicate_deletion(original: &str, candidate: &str) -> bool {
    let original = original.as_bytes();
    let candidate = candidate.as_bytes();
    if original.len() != candidate.len() + 1 {
        return false;
    }
    // A doubled initial can instead be an attached article/partial start;
    // it needs sentence context, so this pass only removes later duplicates.
    (2..original.len()).any(|i| {
        original[i] == original[i - 1]
            && original[..i] == candidate[..i]
            && original[i + 1..] == candidate[i..]
    })
}

fn read(path: &str) -> Result<String, String> {
    let path = path
        .strip_prefix("~/")
        .map(|relative| dirs::home_dir().unwrap_or_default().join(relative))
        .unwrap_or_else(|| PathBuf::from(path));
    fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corrector() -> SpellCorrector {
        SpellCorrector::from_data(
            "SET UTF-8\nSFX S Y 1\nSFX S 0 s .\n",
            "9\nchart\ndataset\nspecified\nplaywright\nletter\nfill/S\nfree\nfreed\ntree/S\n",
            Some("chart 1000000\nletter 1000000\n"),
        )
        .unwrap()
    }

    #[test]
    fn fixes_unlisted_duplicate_errors_and_preserves_layout() {
        let result = corrector().apply("Speccified: chartt, dataaset!\nplaywrright?", &[]);
        assert_eq!(result.text, "Specified: chart, dataset!\nplaywright?");
        assert_eq!(result.edits.len(), 4);
    }

    #[test]
    fn preserves_valid_inflections_acronyms_identifiers_and_code() {
        let input = "letter fills freed trees LLM SSD Bool DataAset chartt_id ./chartt.txt https://x/chartt name@chartt.com --chartt v2-chartt `chartt`\n```\nchartt\n```";
        let result = corrector().apply(input, &[]);
        assert_eq!(result.text, input);
        assert!(matches!(result.text, Cow::Borrowed(_)));
    }

    #[test]
    fn honors_existing_protection_rules_and_unicode_boundaries() {
        let result = corrector().apply(
            "keep chartt; caféchartt chartt",
            &[Regex::new("keep chartt").unwrap()],
        );
        assert_eq!(result.text, "keep chartt; caféchartt chart");
    }

    #[test]
    fn does_not_apply_ambiguous_deletions_or_other_edit_types() {
        let corrector =
            SpellCorrector::from_data("SET UTF-8\n", "3\nxabbcd\nxaabcd\nchart\n", None).unwrap();
        assert_eq!(
            corrector.apply("xaabbcd chrat chartx", &[]).text,
            "xaabbcd chrat chartx"
        );
    }

    #[test]
    fn defers_leading_duplicates_and_rare_short_matches_to_context() {
        let corrector = SpellCorrector::from_data(
            "SET UTF-8\n",
            "4\nache\naport\ntheol\nchart\n",
            Some("ache 1000000\ntheol 48358\nchart 1000000\n"),
        )
        .unwrap();
        assert_eq!(
            corrector
                .apply("aache apport theool cchart chartt", &[])
                .text,
            "aache apport theool cchart chart"
        );
    }

    #[test]
    fn missing_dictionary_is_fail_open() {
        assert!(SpellCorrector::load(&SpellingConfig {
            enabled: true,
            dic_path: "/nonexistent/dictation.dic".into(),
            ..SpellingConfig::default()
        })
        .is_none());
    }
}
