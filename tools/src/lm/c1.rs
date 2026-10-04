//! C1 pilot: does giving the decoder the previous committed word help?
//!
//! See plans/C1-eval-plan.md in the tw_input project. `c1_prepare` turns
//! documents into scored input units; `c1_run` decodes each unit from an
//! empty start (E1) and from the previous word (E2) and compares them.

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, bail};
use chewing::{
    conversion::{
        ChewingEngine, Composition, ConversionEngine, Decoder, Lattice, LatticeBuilder, Symbol,
    },
    dictionary::{CompositeDict, LookupStrategy, StringTable},
    lm::{LoadMode, StaticDict, StaticLm},
    model::{Candidate, WordOrig},
    user::{HistoryDict, UserDict},
    zhuyin::Syllable,
};
use serde::{Deserialize, Serialize};

const PUNCTUATION: &[char] = &['，', '。', '！', '？', '；', '：', '、'];
const MAX_UNIT_CHARS: usize = 39;
const LAMBDA: f64 = 0.6;
const N_BEST: u8 = 10;
const BOOTSTRAP_ROUNDS: usize = 1000;
const BOOTSTRAP_SEED: u64 = 20261005;

#[derive(Deserialize)]
struct Doc {
    doc_id: String,
    text: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Word {
    text: String,
    reading: String,
    /// `unique` or `dominant`
    kind: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Unit {
    id: String,
    doc_id: String,
    para: u32,
    split: String,
    text: String,
    words: Vec<Word>,
    symbols: Vec<String>,
    prev_word: String,
    /// ok, ambiguous, oov, non_han, too_long
    status: String,
}

/// System data loaded by full path, never through the user search path.
struct Model {
    string_table: StringTable,
    static_dict: StaticDict,
    rare_dict: StaticDict,
    lm: StaticLm,
}

impl Model {
    fn open(dir: &Path) -> Result<Model> {
        let file = |name: &str| dir.join(name);
        Ok(Model {
            string_table: StringTable::open_bin(file("static_words.bin"))
                .context("open static_words.bin")?,
            static_dict: StaticDict::open(file("static_dict.bin"))
                .context("open static_dict.bin")?,
            rare_dict: StaticDict::open(file("rare_dict.bin")).context("open rare_dict.bin")?,
            lm: StaticLm::open(file("static_lm.bin"), LoadMode::Lazy)
                .context("open static_lm.bin")?,
        })
    }
    fn decoder(&self) -> Decoder {
        Decoder {
            lm: self.lm.clone(),
            lambda: LAMBDA,
        }
    }
    fn lattice_builder(&self) -> LatticeBuilder {
        let dict = CompositeDict::new(
            self.static_dict.clone(),
            self.rare_dict.clone(),
            HistoryDict::new(self.string_table.clone()),
            UserDict::new(self.string_table.clone()),
        );
        LatticeBuilder {
            dict,
            lookup_strategy: LookupStrategy::Standard,
        }
    }
    fn text_of(&self, cand: &Candidate) -> String {
        match cand {
            Candidate::Word { wid, .. } => self.string_table.get_text(*wid).unwrap_or_default(),
            Candidate::Grapheme(c) => c.to_string(),
            Candidate::None => String::new(),
        }
    }
}

// ---- reading lookup ---------------------------------------------------------

/// Word → readings from `tsi_dict.csv` (`word, bopomofo`), the rare
/// readings from `rare_dict.csv`, and the candidate priority from the old
/// `tsi.csv` (`word,priority,bopomofo`).
pub(crate) struct Readings {
    all: HashMap<String, Vec<String>>,
    rare: HashMap<String, Vec<String>>,
    priority: HashMap<(String, String), i64>,
}

fn read_pairs(path: &Path) -> Result<Vec<(String, String)>> {
    let mut out = vec![];
    for line in
        BufReader::new(File::open(path).with_context(|| path.display().to_string())?).lines()
    {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some((word, reading)) = line.split_once(',') {
            out.push((word.trim().to_string(), reading.trim().to_string()));
        }
    }
    Ok(out)
}

impl Readings {
    fn open(tsi_dict: &Path, rare_dict: &Path, old_tsi: &Path) -> Result<Readings> {
        let mut all: HashMap<String, Vec<String>> = HashMap::new();
        for (w, r) in read_pairs(tsi_dict)? {
            let e = all.entry(w).or_default();
            if !e.contains(&r) {
                e.push(r);
            }
        }
        let mut rare: HashMap<String, Vec<String>> = HashMap::new();
        for (w, r) in read_pairs(rare_dict)? {
            rare.entry(w).or_default().push(r);
        }
        let mut priority = HashMap::new();
        for (w, rest) in read_pairs(old_tsi)? {
            if let Some((p, r)) = rest.split_once(',') {
                if let Ok(p) = p.trim().parse::<i64>() {
                    priority.insert((w, r.trim().to_string()), p);
                }
            }
        }
        Ok(Readings {
            all,
            rare,
            priority,
        })
    }

    /// Reading of one word, or Err(status) when it cannot be chosen.
    pub(crate) fn choose(&self, word: &str) -> Result<Word, &'static str> {
        let Some(readings) = self.all.get(word) else {
            return Err("oov");
        };
        if readings.len() == 1 {
            return Ok(Word {
                text: word.to_string(),
                reading: readings[0].clone(),
                kind: "unique".into(),
            });
        }
        let rare = self.rare.get(word);
        let mut candidates: Vec<&String> = readings
            .iter()
            .filter(|r| !rare.is_some_and(|rs| rs.contains(r)))
            .collect();
        if candidates.is_empty() {
            candidates = readings.iter().collect();
        }
        if candidates.len() == 1 {
            return Ok(Word {
                text: word.to_string(),
                reading: candidates[0].clone(),
                kind: "dominant".into(),
            });
        }
        let mut best: Option<(i64, &String)> = None;
        let mut tie = false;
        for r in candidates {
            let Some(&p) = self.priority.get(&(word.to_string(), r.clone())) else {
                return Err("ambiguous");
            };
            match best {
                Some((bp, _)) if p == bp => tie = true,
                Some((bp, _)) if p < bp => {}
                _ => {
                    best = Some((p, r));
                    tie = false;
                }
            }
        }
        match best {
            Some((_, r)) if !tie => Ok(Word {
                text: word.to_string(),
                reading: r.clone(),
                kind: "dominant".into(),
            }),
            _ => Err("ambiguous"),
        }
    }

    /// Readings for a segmented unit; words missing from the dictionary
    /// fall back to their characters.
    pub(crate) fn annotate(&self, words: &[String]) -> Result<Vec<Word>, &'static str> {
        let mut out = vec![];
        for w in words {
            match self.choose(w) {
                Ok(word) => out.push(word),
                Err("oov") if w.chars().count() > 1 => {
                    for c in w.chars() {
                        out.push(self.choose(&c.to_string())?);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

// ---- unit splitting -----------------------------------------------------------

/// Splits on punctuation; runs of punctuation count as one boundary and
/// empty pieces are dropped.
pub(crate) fn split_units(text: &str) -> Vec<String> {
    text.split(|c| PUNCTUATION.contains(&c))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FFFF)
}

/// What a unit leaves behind as context for the next unit.
pub(crate) enum UnitOutcome {
    /// Not segmented; the next unit starts without context.
    TooLong,
    /// Last word of the segmentation if it is a static dictionary word.
    Segmented(Option<String>),
}

fn is_static_word(c: &Candidate) -> bool {
    matches!(c, Candidate::Word { wid, .. } if matches!(wid.orig(), WordOrig::Static))
}

/// `prev_word` of each unit in one paragraph: the last word of the
/// immediately preceding unit, whatever its status; empty for the first
/// unit, after a too-long unit, or when that word is not a static word.
pub(crate) fn prev_words(outcomes: &[UnitOutcome]) -> Vec<String> {
    let mut prev = String::new();
    outcomes
        .iter()
        .map(|o| {
            let current = std::mem::take(&mut prev);
            prev = match o {
                UnitOutcome::TooLong | UnitOutcome::Segmented(None) => String::new(),
                UnitOutcome::Segmented(Some(w)) => w.clone(),
            };
            current
        })
        .collect()
}

/// Assigns each document to dev or test: documents sorted by a seeded
/// hash, first half dev.
fn assign_splits(doc_ids: &[String], seed: u64) -> HashMap<String, &'static str> {
    let mut keyed: Vec<(u64, &String)> = doc_ids
        .iter()
        .map(|id| {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            id.hash(&mut h);
            (h.finish(), id)
        })
        .collect();
    keyed.sort();
    let half = keyed.len() / 2;
    keyed
        .into_iter()
        .enumerate()
        .map(|(i, (_, id))| (id.clone(), if i < half { "dev" } else { "test" }))
        .collect()
}

pub(crate) fn c1_prepare(
    data_dir: &Path,
    tsi_dict: &Path,
    rare_csv: &Path,
    old_tsi: &Path,
    docs: &Path,
    seed: u64,
    output: &Path,
) -> Result<()> {
    let model = Model::open(data_dir)?;
    let readings = Readings::open(tsi_dict, rare_csv, old_tsi)?;
    let decoder = model.decoder();

    let docs: Vec<Doc> = BufReader::new(File::open(docs)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()?;
    let splits = assign_splits(
        &docs.iter().map(|d| d.doc_id.clone()).collect::<Vec<_>>(),
        seed,
    );

    let mut out = BufWriter::new(File::create(output)?);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for doc in &docs {
        let split = splits[&doc.doc_id];
        // Each document is one paragraph: the source has no paragraph breaks.
        let mut units = vec![];
        let mut outcomes = vec![];
        for (i, text) in split_units(&doc.text).into_iter().enumerate() {
            let n_chars = text.chars().count();
            let mut unit = Unit {
                id: format!("{}-{:03}", doc.doc_id, i),
                doc_id: doc.doc_id.clone(),
                para: 0,
                split: split.into(),
                text: text.clone(),
                words: vec![],
                symbols: vec![],
                prev_word: String::new(),
                status: String::new(),
            };
            if n_chars > MAX_UNIT_CHARS {
                unit.status = "too_long".into();
                outcomes.push(UnitOutcome::TooLong);
            } else {
                let lattice = Lattice::from_str(&text, |s| model.string_table.get_wid(s));
                let best = decoder.decoden(lattice, 1).remove(0);
                let words: Vec<String> = best.candidates.iter().map(|c| model.text_of(c)).collect();
                outcomes.push(UnitOutcome::Segmented(
                    best.candidates
                        .last()
                        .filter(|c| is_static_word(c))
                        .map(|c| model.text_of(c)),
                ));
                if !text.chars().all(is_han) {
                    unit.status = "non_han".into();
                } else {
                    match readings.annotate(&words) {
                        Ok(annotated) => {
                            unit.symbols = annotated
                                .iter()
                                .flat_map(|w| w.reading.split(' ').map(str::to_string))
                                .collect();
                            unit.words = annotated;
                            unit.status = if unit.symbols.len() == n_chars {
                                "ok".into()
                            } else {
                                // A reading with the wrong number of syllables.
                                "ambiguous".into()
                            };
                        }
                        Err(status) => unit.status = status.into(),
                    }
                }
            }
            units.push(unit);
        }
        for (unit, prev) in units.iter_mut().zip(prev_words(&outcomes)) {
            if unit.status == "ok" && prev.is_empty() {
                *counts
                    .entry(format!("{}:ok_with_empty_prev", unit.split))
                    .or_default() += 1;
            }
            unit.prev_word = prev;
            *counts
                .entry(format!("{}:{}", unit.split, unit.status))
                .or_default() += 1;
            serde_json::to_writer(&mut out, &unit)?;
            out.write_all(b"\n")?;
        }
    }
    out.flush()?;
    for (k, v) in counts {
        eprintln!("{k}\t{v}");
    }
    Ok(())
}

// ---- run ---------------------------------------------------------------------

#[derive(Serialize)]
struct UnitResult<'a> {
    id: &'a str,
    doc_id: &'a str,
    reference: &'a str,
    e1: String,
    e2: String,
    e1_dist: usize,
    e2_dist: usize,
}

pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Per-document totals, the unit of bootstrap resampling.
#[derive(Default, Clone, Copy)]
pub(crate) struct DocStat {
    pub(crate) units: usize,
    pub(crate) chars: usize,
    pub(crate) e1_err: usize,
    pub(crate) e2_err: usize,
    pub(crate) e1_dist: usize,
    pub(crate) e2_dist: usize,
}

impl DocStat {
    fn add(&mut self, o: &DocStat) {
        self.units += o.units;
        self.chars += o.chars;
        self.e1_err += o.e1_err;
        self.e2_err += o.e2_err;
        self.e1_dist += o.e1_dist;
        self.e2_dist += o.e2_dist;
    }
    /// Relative reduction of sentence error rate, None when E1 has no error.
    fn d(&self) -> Option<f64> {
        (self.e1_err > 0).then(|| 1.0 - self.e2_err as f64 / self.e1_err as f64)
    }
    /// CER change E2 − E1.
    fn cer_change(&self) -> Option<f64> {
        (self.chars > 0).then(|| (self.e2_dist as f64 - self.e1_dist as f64) / self.chars as f64)
    }
}

/// SplitMix64, enough for reproducible resampling.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// 95% percentile intervals for d and the CER change, resampling
/// documents with replacement. None when any resample is undefined.
pub(crate) fn bootstrap(
    docs: &[DocStat],
    rounds: usize,
    seed: u64,
) -> Option<((f64, f64), (f64, f64))> {
    if docs.is_empty() {
        return None;
    }
    let mut rng = Rng(seed);
    let mut ds = Vec::with_capacity(rounds);
    let mut cers = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut total = DocStat::default();
        for _ in 0..docs.len() {
            total.add(&docs[(rng.next() % docs.len() as u64) as usize]);
        }
        ds.push(total.d()?);
        cers.push(total.cer_change()?);
    }
    let ci = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let lo = v[(v.len() as f64 * 0.025) as usize];
        let hi = v[((v.len() as f64 * 0.975) as usize).min(v.len() - 1)];
        (lo, hi)
    };
    Some((ci(&mut ds), ci(&mut cers)))
}

pub(crate) fn c1_run(data_dir: &Path, units: &Path, split: &str, output: &Path) -> Result<()> {
    let model = Model::open(data_dir)?;
    let builder = model.lattice_builder();
    let decoder = model.decoder();
    let engine = ChewingEngine {
        word_lattice_builder: model.lattice_builder(),
        decoder: model.decoder(),
        string_table: model.string_table.clone(),
    };

    let units: Vec<Unit> = BufReader::new(File::open(units)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()?;

    let mut out = BufWriter::new(File::create(output)?);
    let mut per_doc: BTreeMap<String, DocStat> = BTreeMap::new();
    let mut engine_mismatch = 0;
    let mut failed = 0;
    for unit in units
        .iter()
        .filter(|u| u.split == split && u.status == "ok")
    {
        let mut comp = Composition::new();
        for s in &unit.symbols {
            let syl: Syllable = s
                .parse()
                .with_context(|| format!("bad syllable {s} in {}", unit.id))?;
            comp.push(Symbol::Syllable(syl));
        }
        let decode = |start: Candidate| -> String {
            let lattice = builder.build_lattice(&comp);
            decoder
                .decoden_from(lattice, N_BEST, start)
                .first()
                .map(|h| h.candidates.iter().map(|c| model.text_of(c)).collect())
                .unwrap_or_default()
        };
        let e1 = decode(Candidate::None);
        let start = match model.string_table.get_wid(&unit.prev_word) {
            Some(wid) if !unit.prev_word.is_empty() => Candidate::Word {
                wid,
                hist_prob: f64::NEG_INFINITY,
                user_pref: None,
            },
            _ => Candidate::None,
        };
        let e2 = decode(start);

        // E1 must equal what the stock engine outputs.
        let stock: String = engine
            .convert(&comp)
            .first()
            .map(|o| o.intervals.iter().map(|i| i.text.to_string()).collect())
            .unwrap_or_default();
        if stock != e1 {
            engine_mismatch += 1;
        }
        if e1.is_empty() || e2.is_empty() {
            failed += 1;
        }

        let chars = unit.text.chars().count();
        // A failed conversion counts as entirely wrong.
        let dist = |r: &str| {
            if r.is_empty() {
                chars
            } else {
                levenshtein(&unit.text, r)
            }
        };
        let (e1_dist, e2_dist) = (dist(&e1), dist(&e2));
        let stat = per_doc.entry(unit.doc_id.clone()).or_default();
        stat.add(&DocStat {
            units: 1,
            chars,
            e1_err: usize::from(e1 != unit.text),
            e2_err: usize::from(e2 != unit.text),
            e1_dist,
            e2_dist,
        });
        serde_json::to_writer(
            &mut out,
            &UnitResult {
                id: &unit.id,
                doc_id: &unit.doc_id,
                reference: &unit.text,
                e1,
                e2,
                e1_dist,
                e2_dist,
            },
        )?;
        out.write_all(b"\n")?;
    }
    out.flush()?;

    if engine_mismatch > 0 {
        bail!("{engine_mismatch} units: empty-start decoding differs from ChewingEngine output");
    }
    let docs: Vec<DocStat> = per_doc.values().copied().collect();
    let mut total = DocStat::default();
    docs.iter().for_each(|d| total.add(d));
    let rate = |n: usize| n as f64 / total.units.max(1) as f64;
    println!("split\t{split}");
    println!("documents\t{}", docs.len());
    println!("units\t{}", total.units);
    println!("failed\t{failed}");
    println!("e1_sentence_error\t{:.4}", rate(total.e1_err));
    println!("e2_sentence_error\t{:.4}", rate(total.e2_err));
    println!(
        "e1_cer\t{:.4}",
        total.e1_dist as f64 / total.chars.max(1) as f64
    );
    println!(
        "e2_cer\t{:.4}",
        total.e2_dist as f64 / total.chars.max(1) as f64
    );
    match (
        total.d(),
        bootstrap(&docs, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED),
    ) {
        (Some(d), Some(((d_lo, d_hi), (c_lo, c_hi)))) => {
            println!("d\t{d:.4}\t95%CI\t{d_lo:.4}\t{d_hi:.4}");
            println!(
                "cer_change\t{:.4}\t95%CI\t{c_lo:.4}\t{c_hi:.4}",
                total.cer_change().unwrap_or(0.0)
            );
        }
        _ => println!("d\tN/A"),
    }
    Ok(())
}

// ---- recall (S0) ---------------------------------------------------------------

/// Keeps the first occurrence of each candidate text, preserving order.
/// Decoder output is sorted by cost, so the kept one is the cheapest.
pub(crate) fn dedup_texts(texts: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    texts
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Recall {
    pub(crate) units: usize,
    /// Units whose first candidate equals the reference.
    pub(crate) top1: usize,
    /// Units whose reference is among the candidates.
    pub(crate) found: usize,
    pub(crate) candidates_total: usize,
    pub(crate) candidates_max: usize,
}

impl Recall {
    pub(crate) fn add(&mut self, reference: &str, candidates: &[String]) {
        self.units += 1;
        self.top1 += usize::from(candidates.first().is_some_and(|c| c == reference));
        self.found += usize::from(candidates.iter().any(|c| c == reference));
        self.candidates_total += candidates.len();
        self.candidates_max = self.candidates_max.max(candidates.len());
    }
    /// `(found − baseline) ÷ (units − baseline)`; None when undefined.
    pub(crate) fn oracle(&self, baseline: usize) -> Option<f64> {
        let room = self.units.checked_sub(baseline).filter(|r| *r > 0)?;
        Some((self.found as f64 - baseline as f64) / room as f64)
    }
}

pub(crate) fn c1_recall(
    data_dir: &Path,
    units: &Path,
    split: &str,
    k: u8,
    baseline: Option<usize>,
    check_e1: Option<&Path>,
) -> Result<()> {
    let model = Model::open(data_dir)?;
    let builder = model.lattice_builder();
    let decoder = model.decoder();

    let e1: Option<HashMap<String, String>> = match check_e1 {
        Some(path) => {
            #[derive(Deserialize)]
            struct E1 {
                id: String,
                e1: String,
            }
            let mut map = HashMap::new();
            for line in BufReader::new(File::open(path)?).lines() {
                let r: E1 = serde_json::from_str(&line?)?;
                map.insert(r.id, r.e1);
            }
            Some(map)
        }
        None => None,
    };

    let units: Vec<Unit> = BufReader::new(File::open(units)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()?;

    let mut recall = Recall::default();
    let mut e1_mismatch = 0;
    for unit in units
        .iter()
        .filter(|u| u.split == split && u.status == "ok")
    {
        let mut comp = Composition::new();
        for s in &unit.symbols {
            let syl: Syllable = s
                .parse()
                .with_context(|| format!("bad syllable {s} in {}", unit.id))?;
            comp.push(Symbol::Syllable(syl));
        }
        let texts = decoder
            .decoden(builder.build_lattice(&comp), k)
            .into_iter()
            .map(|h| h.candidates.iter().map(|c| model.text_of(c)).collect())
            .collect();
        let candidates = dedup_texts(texts);
        if let Some(e1) = &e1 {
            let expected = e1
                .get(&unit.id)
                .with_context(|| format!("{} missing from E1 results", unit.id))?;
            if candidates.first() != Some(expected) {
                e1_mismatch += 1;
            }
        }
        recall.add(&unit.text, &candidates);
    }

    if e1_mismatch > 0 {
        bail!("{e1_mismatch} units: first candidate differs from the E1 results");
    }
    let base = baseline.unwrap_or(recall.top1);
    println!("split\t{split}");
    println!("k\t{k}");
    println!("units\t{}", recall.units);
    println!("top1_correct_this_k\t{}", recall.top1);
    println!("baseline\t{base}");
    println!("found_in_candidates\t{}", recall.found);
    match recall.oracle(base) {
        Some(o) => println!("oracle_improvement\t{o:.4}"),
        None => println!("oracle_improvement\tN/A"),
    }
    println!(
        "candidates_mean\t{:.2}",
        recall.candidates_total as f64 / recall.units.max(1) as f64
    );
    println!("candidates_max\t{}", recall.candidates_max);
    if check_e1.is_some() {
        println!("e1_check\tok");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_keeps_first_and_order() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            dedup_texts(v(&["甲", "乙", "甲", "丙", "乙"])),
            v(&["甲", "乙", "丙"])
        );
    }

    #[test]
    fn recall_counts_and_oracle() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut r = Recall::default();
        r.add("今天", &v(&["今天", "金天"])); // first is correct
        r.add("天氣", &v(&["添氣", "天器", "天氣"])); // correct at position 3
        r.add("出門", &v(&["初門"])); // missing
        assert_eq!(
            r,
            Recall {
                units: 3,
                top1: 1,
                found: 2,
                candidates_total: 6,
                candidates_max: 3,
            }
        );
        // (2 − 1) ÷ (3 − 1)
        assert_eq!(r.oracle(1), Some(0.5));
        assert_eq!(r.oracle(3), None);
    }

    #[test]
    fn levenshtein_counts_each_edit() {
        assert_eq!(levenshtein("今天天氣", "今天天氣"), 0);
        assert_eq!(levenshtein("今天天氣", "今天添氣"), 1); // substitution
        assert_eq!(levenshtein("今天天氣", "今天氣"), 1); // deletion
        assert_eq!(levenshtein("今天氣", "今天天氣"), 1); // insertion
        assert_eq!(levenshtein("", "今天"), 2);
    }

    #[test]
    fn split_merges_punctuation_runs_and_drops_empty() {
        assert_eq!(split_units("你好，世界。"), vec!["你好", "世界"]);
        assert_eq!(split_units("真的嗎？！好。"), vec!["真的嗎", "好"]);
        assert_eq!(split_units("。，"), Vec::<String>::new());
    }

    #[test]
    fn han_detection() {
        assert!("今天天氣".chars().all(is_han));
        assert!(!"今天A".chars().all(is_han));
        assert!(!"115年".chars().all(is_han));
    }

    fn readings() -> Readings {
        let all = HashMap::from([
            ("天".to_string(), vec!["ㄊㄧㄢ".to_string()]),
            (
                "的".to_string(),
                vec!["ㄉㄜ˙".into(), "ㄉㄧˊ".into(), "ㄉㄧˋ".into()],
            ),
            ("重".to_string(), vec!["ㄓㄨㄥˋ".into(), "ㄔㄨㄥˊ".into()]),
            ("行".to_string(), vec!["ㄒㄧㄥˊ".into(), "ㄏㄤˊ".into()]),
            ("為".to_string(), vec!["ㄨㄟˋ".into(), "ㄨㄟˊ".into()]),
            ("今天".to_string(), vec!["ㄐㄧㄣ ㄊㄧㄢ".into()]),
        ]);
        let rare = HashMap::from([
            ("的".to_string(), vec!["ㄉㄧˋ".to_string()]),
            ("重".to_string(), vec!["ㄔㄨㄥˊ".to_string()]),
        ]);
        let priority = HashMap::from([
            (("的".to_string(), "ㄉㄜ˙".to_string()), 147270),
            (("的".to_string(), "ㄉㄧˊ".to_string()), 14727),
            (("的".to_string(), "ㄉㄧˋ".to_string()), 1),
            (("為".to_string(), "ㄨㄟˋ".to_string()), 100),
            (("為".to_string(), "ㄨㄟˊ".to_string()), 100),
        ]);
        Readings {
            all,
            rare,
            priority,
        }
    }

    #[test]
    fn reading_rules() {
        let r = readings();
        // unique
        let w = r.choose("天").unwrap();
        assert_eq!((w.reading.as_str(), w.kind.as_str()), ("ㄊㄧㄢ", "unique"));
        // dominant by priority after removing rare readings
        let w = r.choose("的").unwrap();
        assert_eq!((w.reading.as_str(), w.kind.as_str()), ("ㄉㄜ˙", "dominant"));
        // only one non-rare reading left
        assert_eq!(r.choose("重").unwrap().reading, "ㄓㄨㄥˋ");
        // no priority data
        assert_eq!(r.choose("行").err(), Some("ambiguous"));
        // priority tie
        assert_eq!(r.choose("為").err(), Some("ambiguous"));
        // missing
        assert_eq!(r.choose("鑫").err(), Some("oov"));
    }

    #[test]
    fn unknown_word_falls_back_to_characters() {
        let r = readings();
        let words = r.annotate(&["天天".to_string()]).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(r.annotate(&["天鑫".to_string()]).err(), Some("oov"));
    }

    #[test]
    fn prev_word_rules() {
        let w = |s: &str| UnitOutcome::Segmented(Some(s.to_string()));
        let outcomes = [
            w("今天"),                    // first unit of the paragraph
            w("天氣"),                    // gets 今天
            UnitOutcome::TooLong,         // gets 天氣
            w("很好"),                    // after too long: empty
            UnitOutcome::Segmented(None), // gets 很好; last word not static
            w("出門"),                    // empty: previous word had no static id
        ];
        assert_eq!(
            prev_words(&outcomes),
            vec!["", "今天", "天氣", "", "很好", ""]
        );
    }

    #[test]
    fn only_static_words_count_as_context() {
        use chewing::model::WordId;
        let word = |id: u32| Candidate::Word {
            wid: WordId(id),
            hist_prob: 0.0,
            user_pref: None,
        };
        assert!(is_static_word(&word(42)));
        assert!(!is_static_word(&word(0x0100_0001))); // user word id range
        assert!(!is_static_word(&Candidate::Grapheme('A')));
        assert!(!is_static_word(&Candidate::None));
    }

    #[test]
    fn bootstrap_is_reproducible_and_reports_na() {
        let docs: Vec<DocStat> = (0..20)
            .map(|i| DocStat {
                units: 10,
                chars: 50,
                e1_err: 2 + i % 3,
                e2_err: 1 + i % 2,
                e1_dist: 4,
                e2_dist: 3,
            })
            .collect();
        let a = bootstrap(&docs, 200, 7);
        assert!(a.is_some());
        assert_eq!(a, bootstrap(&docs, 200, 7));
        let no_errors = vec![DocStat {
            units: 5,
            chars: 20,
            ..Default::default()
        }];
        assert_eq!(bootstrap(&no_errors, 10, 7), None);
    }
}
