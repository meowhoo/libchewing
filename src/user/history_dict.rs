//! Automatic learned user vocabulary list
//!
//! The auto user vocabulary list stores new words learned from user interactions

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    ops::Bound::{Excluded, Included},
    path::Path,
    sync::{Arc, RwLock},
};

use scoped_error::{bail, expect_error, impl_context_error};

use crate::{
    bare::{BareDecoder, BareEncoder},
    dictionary::{LookupStrategy, StringTable},
    model::WordId,
    zhuyin::{Syllable, SyllableVec},
};

/// Automatic learned user vocabulary list
#[derive(Debug, Clone)]
pub struct HistoryDict {
    inner: Arc<RwLock<HistoryDictInner>>,
}

#[derive(Debug)]
struct HistoryDictInner {
    string_table: StringTable,
    half_life: u32,
    generation: u64,
    records: BTreeMap<SyllableVec, Vec<HistoryDictEntry>>,
}

#[derive(Debug)]
struct HistoryDictEntry {
    wid: WordId,
    seen: u32,
    epoch: u64,
}

impl HistoryDict {
    pub const HALF_LIFE: u32 = 50_000;
    pub const COLD_THRESHOLD: f64 = 2000.0;

    /// Returns an empty HistoryDict
    pub fn new(string_table: StringTable) -> HistoryDict {
        HistoryDict {
            inner: Arc::new(RwLock::new(HistoryDictInner {
                string_table,
                half_life: Self::HALF_LIFE,
                generation: 0,
                records: BTreeMap::new(),
            })),
        }
    }
    /// Initialize an empty HistoryDict on the filesystem.
    ///
    /// If a file already exists then it will be truncated.
    pub fn init<P: AsRef<Path>>(path: P) -> Result<(), HistoryDictError> {
        expect_error("Failed to initialize HistoryDict", || {
            let dict = Self::new(StringTable::new());
            let file = File::create(path)?;
            dict.to_writer(file)?;
            Ok(())
        })
    }
    /// Open an HistoryDict file and read from it.
    pub fn open<P: AsRef<Path>>(
        path: P,
        string_table: StringTable,
    ) -> Result<HistoryDict, HistoryDictError> {
        expect_error("Failed to open user history dictionary", || {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            Ok(HistoryDict::from_reader(reader, string_table)?)
        })
    }
    /// Reads history dict from the IO stream
    pub fn from_reader<R: BufRead>(
        reader: R,
        string_table: StringTable,
    ) -> Result<HistoryDict, HistoryDictError> {
        expect_error("Failed to parse history dict", || {
            let mut decoder = BareDecoder::new(reader);

            // Read file magic
            let magic = decoder.read_data_exact(4)?;
            if magic != b"CHHD" {
                bail!("Invalid file header");
            }
            // Read HistoryDictFile union version
            let version = decoder.read_uint()?;
            if version != 0 {
                bail!("Unknown file version");
            }
            // Read HistoryDictHeader flags
            let flags = decoder.read_u32()?;
            if flags != 0 {
                bail!("Unknown header flags");
            }
            let half_life = decoder.read_u32()?;
            let generation = decoder.read_u64()?;
            let count = decoder.read_u32()?;
            let mut records = BTreeMap::new();

            // Read record frames
            //
            // Older writers stored the number of distinct syllable keys
            // instead of the number of frames. That count can be smaller
            // than the real number of frames (homophones) or larger (empty
            // keys left by remove). Read until a clean EOF so those files
            // are recovered instead of truncated or rejected.
            let mut read = 0;
            loop {
                if decoder.is_eof()? {
                    break;
                }
                if read >= count {
                    log::debug!("history dict has more frames than its header count {count}");
                }
                read += 1;
                let len = decoder.read_uint()? as usize;
                let mut syllables = SyllableVec::new();
                for _ in 0..len {
                    let syl = Syllable::try_from(decoder.read_u16()?)?;
                    syllables.push(syl);
                }
                let raw_word = decoder.read_data()?;
                let word = str::from_utf8(&raw_word)?;
                let seen = decoder.read_u32()?;
                let epoch = decoder.read_u64()?;

                let wid = string_table.intern(word);

                let word_entries = records.entry(syllables).or_insert(vec![]);
                word_entries.push(HistoryDictEntry { wid, seen, epoch });
            }

            Ok(HistoryDict {
                inner: Arc::new(RwLock::new(HistoryDictInner {
                    string_table,
                    half_life,
                    generation,
                    records,
                })),
            })
        })
    }
    pub fn to_writer<W: Write>(&self, writer: W) -> Result<(), HistoryDictError> {
        expect_error("Unable to serialize the history dictionary", || {
            let lock = self
                .inner
                .read()
                .expect("Unable to acquire HistoryDict reader lock");
            let mut encoder = BareEncoder::new(writer);

            // Write file magic
            encoder.write_data_exact(b"CHHD")?;
            // Write HistoryDictFile union version
            encoder.write_uint(0)?;
            // Write HistoryDictHeader flags
            encoder.write_u32(0)?;
            encoder.write_u32(lock.half_life)?;
            encoder.write_u64(lock.generation)?;
            let frames: usize = lock.records.values().map(Vec::len).sum();
            encoder.write_u32(frames as u32)?;

            for (syllables, entries) in lock.records.iter() {
                for entry in entries {
                    encoder.write_uint(syllables.len() as u64)?;
                    for syl in syllables {
                        encoder.write_u16(syl.to_u16())?;
                    }
                    let word = lock
                        .string_table
                        .get_text(entry.wid)
                        .expect("Should have this word");
                    encoder.write_data(word.as_bytes())?;
                    encoder.write_u32(entry.seen)?;
                    encoder.write_u64(entry.epoch)?;
                }
            }
            Ok(())
        })
    }
    pub(crate) fn observe(&self, syllables: &[Syllable], word: &str) {
        let mut lock = self
            .inner
            .write()
            .expect("Unable to acquire HistoryDict writer lock");
        lock.generation += 1;
        let wid = lock.string_table.intern(word);
        let generation = lock.generation;
        let hist_entries = lock.records.entry(syllables.into()).or_default();
        if let Some(pos) = hist_entries.iter().position(|e| e.wid == wid) {
            hist_entries[pos].seen += 1;
        } else {
            hist_entries.push(HistoryDictEntry {
                wid,
                seen: 1,
                epoch: generation,
            });
        }
    }
    pub fn remove(&self, syllables: &[Syllable], word: &str) {
        let mut lock = self
            .inner
            .write()
            .expect("Unable to acquire UserDict writer lock");
        let wid = lock.string_table.intern(word);
        let key: SyllableVec = syllables.into();
        let Some(hist_entries) = lock.records.get_mut(&key) else {
            return;
        };
        if let Some(pos) = hist_entries.iter().position(|e| e.wid == wid) {
            hist_entries.remove(pos);
        }
        if hist_entries.is_empty() {
            lock.records.remove(&key);
        }
    }
    // Return all words and their history based unigram log10 prob
    pub(crate) fn lookup(
        &self,
        syllables: &[Syllable],
        strategy: LookupStrategy,
    ) -> Vec<(WordId, f64)> {
        let lock = self
            .inner
            .read()
            .expect("Unable to acquire HistoryDict reader lock");
        let total = lock.generation as f64 + Self::COLD_THRESHOLD;
        match strategy {
            LookupStrategy::Standard => lock
                .records
                .get(syllables)
                .map(|entries| {
                    entries
                        .iter()
                        .map(|e| {
                            let count =
                                true_count(lock.half_life, lock.generation, e.seen, e.epoch);
                            let log10prob = (count as f64 / total).log10();
                            (e.wid, log10prob)
                        })
                        .collect()
                })
                .unwrap_or_default(),
            LookupStrategy::FuzzyPartialPrefix => {
                let mut end = syllables.to_vec();
                // NB: relies on the syllable encoding to
                // ensure Syllable::EMPTY is greater than all real syllables.
                end.push(Syllable::new());
                lock.records
                    .range::<[Syllable], _>((Included(syllables), Excluded(end.as_slice())))
                    .flat_map(|(_, entries)| {
                        entries.iter().map(|e| {
                            let count =
                                true_count(lock.half_life, lock.generation, e.seen, e.epoch);
                            let log10prob = (count as f64 / total).log10();
                            (e.wid, log10prob)
                        })
                    })
                    .collect()
            }
        }
    }
}

// Linear approximation (first-order)
fn true_count(h: u32, g: u64, c_i: u32, b_i: u64) -> u32 {
    let h = h as u64;
    let elapsed = g - b_i;
    let num_halvings = elapsed / h;
    let reminder = elapsed % h;
    let base = c_i as u64 >> num_halvings;
    (base - (base * reminder) / (2 * h)) as u32
}

impl_context_error!(pub HistoryDictError);

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Cursor};

    use super::HistoryDict;
    use crate::{dictionary::StringTable, zhuyin::SyllableVec};

    fn roundtrip(dict: &HistoryDict) -> Result<HistoryDict, Box<dyn Error>> {
        let mut buf = vec![];
        dict.to_writer(&mut buf)?;
        Ok(HistoryDict::from_reader(
            Cursor::new(buf),
            StringTable::new(),
        )?)
    }

    fn words(dict: &HistoryDict) -> Vec<String> {
        let lock = dict.inner.read().unwrap();
        let mut out: Vec<String> = lock
            .records
            .values()
            .flatten()
            .map(|e| lock.string_table.get_text(e.wid).unwrap())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn roundtrip_keeps_homophones() -> Result<(), Box<dyn Error>> {
        let dict = HistoryDict::new(StringTable::new());
        let syl: SyllableVec = "ㄕˋ ㄕˊ".parse()?;
        dict.observe(syl.as_ref(), "事實");
        dict.observe(syl.as_ref(), "事時");
        let loaded = roundtrip(&dict)?;
        assert_eq!(words(&loaded), vec!["事實".to_string(), "事時".to_string()]);
        Ok(())
    }

    /// Files written by older versions stored the number of syllable keys
    /// as the frame count. Simulate that by patching the header.
    fn with_header_count(dict: &HistoryDict, count: u32) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut buf = vec![];
        dict.to_writer(&mut buf)?;
        // "CHHD" (4) + version uint (1) + flags (4) + half_life (4) + generation (8)
        const COUNT_OFFSET: usize = 21;
        buf[COUNT_OFFSET..COUNT_OFFSET + 4].copy_from_slice(&count.to_le_bytes());
        Ok(buf)
    }

    #[test]
    fn read_legacy_file_with_too_small_count() -> Result<(), Box<dyn Error>> {
        let dict = HistoryDict::new(StringTable::new());
        let syl: SyllableVec = "ㄕˋ ㄕˊ".parse()?;
        dict.observe(syl.as_ref(), "事實");
        dict.observe(syl.as_ref(), "事時");
        let buf = with_header_count(&dict, 1)?;
        let loaded = HistoryDict::from_reader(Cursor::new(buf), StringTable::new())?;
        assert_eq!(words(&loaded), vec!["事實".to_string(), "事時".to_string()]);
        Ok(())
    }

    #[test]
    fn read_legacy_file_with_too_large_count() -> Result<(), Box<dyn Error>> {
        let dict = HistoryDict::new(StringTable::new());
        let syl: SyllableVec = "ㄚ ㄇㄠ".parse()?;
        dict.observe(syl.as_ref(), "阿貓");
        let buf = with_header_count(&dict, 3)?;
        let loaded = HistoryDict::from_reader(Cursor::new(buf), StringTable::new())?;
        assert_eq!(words(&loaded), vec!["阿貓".to_string()]);
        Ok(())
    }

    #[test]
    fn truncated_frame_is_still_an_error() -> Result<(), Box<dyn Error>> {
        let dict = HistoryDict::new(StringTable::new());
        let syl: SyllableVec = "ㄚ ㄇㄠ".parse()?;
        dict.observe(syl.as_ref(), "阿貓");
        let mut buf = vec![];
        dict.to_writer(&mut buf)?;
        buf.truncate(buf.len() - 3);
        assert!(HistoryDict::from_reader(Cursor::new(buf), StringTable::new()).is_err());
        Ok(())
    }

    #[test]
    fn roundtrip_after_removing_unknown_word() -> Result<(), Box<dyn Error>> {
        let dict = HistoryDict::new(StringTable::new());
        let a: SyllableVec = "ㄚ ㄇㄠ".parse()?;
        let b: SyllableVec = "ㄚ ㄍㄡˇ".parse()?;
        dict.observe(a.as_ref(), "阿貓");
        dict.remove(b.as_ref(), "阿狗");
        let loaded = roundtrip(&dict)?;
        assert_eq!(words(&loaded), vec!["阿貓".to_string()]);
        Ok(())
    }
}
