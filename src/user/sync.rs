//! Keep the user dictionaries consistent across processes.
//!
//! On Windows the input method is loaded into every application process,
//! so several [`Editor`](crate::editor::Editor) instances share one user
//! directory. Each keeps the dictionaries in memory. Writing the whole
//! in-memory copy back would let the last writer erase what other
//! processes learned, so every sync does a three-way merge under an
//! exclusive file lock:
//!
//! - *base*: the file content this process last read or wrote
//! - *ours*: the current in-memory content
//! - *theirs*: the current file content, possibly written by others
//!
//! An entry changed only on one side takes that side; an entry changed on
//! both sides keeps the local value. The merged result is written to a
//! unique temporary file and renamed over the target, then applied back to
//! memory so this process also sees what others learned.

use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    fs::{self, OpenOptions},
    hash::{Hash, Hasher},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use log::warn;
use scoped_error::{expect_error, impl_context_error};

use super::{HistoryDict, UserDict, dict::UserSnapshot, history_dict::HistorySnapshot};
use crate::dictionary::StringTable;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const USER_DICT_FILE: &str = "user_dict.csv";
const HISTORY_DICT_FILE: &str = "history_dict.bin";
const LOCK_FILE: &str = "user_data.lock";

/// Merge state of one process for one user directory.
#[derive(Debug)]
pub(crate) struct UserDataSync {
    dir: PathBuf,
    user: Tracked<UserSnapshot>,
    hist: Tracked<HistorySnapshot>,
}

/// What this process knows about one file in the user directory.
#[derive(Debug)]
struct Tracked<S> {
    /// File content as of the last sync.
    base: S,
    /// Hash of the bytes this process last wrote or read, used to skip
    /// parsing when nobody else touched the file.
    disk_hash: Option<u64>,
}

/// State of a file on disk relative to [`Tracked::base`].
enum Disk<S> {
    /// Same bytes as last time: equal to base.
    Unchanged,
    /// Current content and its hash; `None` hash means the file is missing.
    Content(S, Option<u64>),
    /// The file could not be parsed and was moved aside.
    Unusable,
}

impl<S: Clone + PartialEq + Default> Tracked<S> {
    fn new(base: S) -> Tracked<S> {
        Tracked {
            base,
            disk_hash: None,
        }
    }

    /// Merges `ours` with the file at `path`, writes the result if it
    /// differs from what is on disk, and returns it.
    fn sync(
        &mut self,
        path: &Path,
        ours: &S,
        parse: impl FnOnce(&[u8]) -> Result<S, BoxError>,
        merge: impl FnOnce(&S, &S, &S) -> S,
        serialize: impl FnOnce(&S, &mut Vec<u8>) -> Result<(), BoxError>,
    ) -> Result<S, BoxError> {
        let disk = self.read(path, parse)?;
        let (merged, needs_write, disk_hash) = match disk {
            Disk::Unchanged => (ours.clone(), *ours != self.base, self.disk_hash),
            Disk::Content(theirs, hash) => {
                let merged = merge(&self.base, ours, &theirs);
                let needs_write = merged != theirs;
                (merged, needs_write, hash)
            }
            Disk::Unusable => (ours.clone(), true, None),
        };
        self.disk_hash = disk_hash;
        if needs_write {
            let mut buf = vec![];
            serialize(&merged, &mut buf)?;
            write_atomic(path, &buf)?;
            self.disk_hash = Some(hash_bytes(&buf));
        }
        self.base = merged.clone();
        Ok(merged)
    }

    /// Reads the file at `path`.
    ///
    /// A missing file reads as empty, so deleting the file resets the
    /// dictionary. A file that cannot be parsed is renamed to a
    /// `.corrupt-<time>` backup so local data is written back instead of
    /// being replaced by nothing. Any other read error aborts the sync so a
    /// file we cannot see is never overwritten.
    fn read(
        &self,
        path: &Path,
        parse: impl FnOnce(&[u8]) -> Result<S, BoxError>,
    ) -> Result<Disk<S>, BoxError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Ok(Disk::Content(S::default(), None));
            }
            Err(err) => return Err(err.into()),
        };
        let hash = hash_bytes(&bytes);
        if self.disk_hash == Some(hash) {
            return Ok(Disk::Unchanged);
        }
        match parse(&bytes) {
            Ok(theirs) => Ok(Disk::Content(theirs, Some(hash))),
            Err(err) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                let backup = path.with_file_name(format!("{name}.corrupt-{}", unix_time()));
                warn!(
                    "Unable to parse {}: {err}; keeping it as {}",
                    path.display(),
                    backup.display()
                );
                fs::rename(path, &backup)?;
                Ok(Disk::Unusable)
            }
        }
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

impl UserDataSync {
    /// Starts tracking from the content currently in memory, which is
    /// what the editor loaded from `dir`.
    pub(crate) fn new(dir: PathBuf, user: &UserDict, hist: &HistoryDict) -> UserDataSync {
        UserDataSync {
            dir,
            user: Tracked::new(user.snapshot()),
            hist: Tracked::new(hist.snapshot()),
        }
    }

    /// Merges memory with the files on disk, writes the result and updates
    /// memory with changes made by other processes.
    pub(crate) fn sync(&mut self, user: &UserDict, hist: &HistoryDict) -> Result<(), SyncError> {
        expect_error("Failed to sync user data", || {
            fs::create_dir_all(&self.dir)?;
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(self.dir.join(LOCK_FILE))?;
            lock.lock()?;

            // Only file work happens under the file lock. Memory is updated
            // after the lock is released so a process that is slow to give
            // up its in-memory locks never stalls the others.
            let user_ours = user.snapshot();
            let user_merged = self.user.sync(
                &self.dir.join(USER_DICT_FILE),
                &user_ours,
                |bytes| Ok(UserDict::from_reader(bytes, StringTable::new())?.snapshot()),
                merge_user,
                |snapshot, buf| Ok(UserDict::write_snapshot(snapshot, buf)?),
            )?;

            let hist_ours = hist.snapshot();
            let hist_merged = self.hist.sync(
                &self.dir.join(HISTORY_DICT_FILE),
                &hist_ours,
                |bytes| Ok(HistoryDict::from_reader(bytes, StringTable::new())?.snapshot()),
                merge_history,
                |snapshot, buf| Ok(HistoryDict::write_snapshot(snapshot, buf)?),
            )?;

            lock.unlock()?;
            drop(lock);

            user.apply_merge(&user_ours, &user_merged);
            hist.apply_merge(&hist_ours, &hist_merged);
            Ok(())
        })
    }
}

impl Default for HistorySnapshot {
    fn default() -> Self {
        HistorySnapshot {
            half_life: HistoryDict::HALF_LIFE,
            generation: 0,
            entries: BTreeMap::new(),
        }
    }
}

/// Picks the value for one key: a side that did not change from base
/// yields to the other side; when both changed, ours wins.
fn pick<'a, V: PartialEq>(
    base: Option<&'a V>,
    ours: Option<&'a V>,
    theirs: Option<&'a V>,
) -> Option<&'a V> {
    if ours == base { theirs } else { ours }
}

fn merge_user(base: &UserSnapshot, ours: &UserSnapshot, theirs: &UserSnapshot) -> UserSnapshot {
    let keys: BTreeSet<_> = base
        .keys()
        .chain(ours.keys())
        .chain(theirs.keys())
        .collect();
    keys.into_iter()
        .filter_map(|k| pick(base.get(k), ours.get(k), theirs.get(k)).map(|v| (k.clone(), *v)))
        .collect()
}

fn merge_history(
    base: &HistorySnapshot,
    ours: &HistorySnapshot,
    theirs: &HistorySnapshot,
) -> HistorySnapshot {
    // Observations made locally since base continue the timeline of
    // theirs. Shift local epochs by how far theirs moved from base.
    let local_observations = ours.generation.saturating_sub(base.generation);
    let generation = theirs.generation + local_observations;
    let shift = theirs.generation as i128 - base.generation as i128;
    let shifted = |(seen, epoch): (u32, u64)| {
        let epoch = (epoch as i128 + shift).clamp(0, generation as i128) as u64;
        (seen, epoch)
    };

    let keys: BTreeSet<_> = base
        .entries
        .keys()
        .chain(ours.entries.keys())
        .chain(theirs.entries.keys())
        .collect();
    let mut entries = BTreeMap::new();
    for k in keys {
        let (b, o, t) = (
            base.entries.get(k),
            ours.entries.get(k),
            theirs.entries.get(k),
        );
        let value = if o == b {
            t.copied()
        } else if t == b {
            o.copied().map(shifted)
        } else {
            match (b, o, t) {
                // Both observed the word: add the local observations to theirs.
                (_, Some(&(o_seen, o_epoch)), Some(&(t_seen, t_epoch))) => {
                    let b_seen = b.map(|v| v.0).unwrap_or(0);
                    let seen = t_seen.saturating_add(o_seen.saturating_sub(b_seen));
                    // Keep the first time the word was seen on either side.
                    let epoch = if b.is_some() {
                        t_epoch
                    } else {
                        t_epoch.min(shifted((0, o_epoch)).1)
                    };
                    Some((seen, epoch))
                }
                // Removed locally: an explicit removal wins.
                (_, None, _) => None,
                // Removed elsewhere but observed again locally: keep ours.
                (_, Some(&v), None) => Some(shifted(v)),
            }
        };
        if let Some(v) = value {
            entries.insert(k.clone(), v);
        }
    }
    HistorySnapshot {
        half_life: theirs.half_life,
        generation,
        entries,
    }
}

/// Writes `path` by writing a uniquely named temporary file in the same
/// directory and renaming it over the target.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), SyncError> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    expect_error("Failed to write user data file", || {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("invalid user data path")?;
        let tmp = path.with_file_name(format!(
            "{file_name}.{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(target_family = "unix")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let result = (|| -> Result<(), BoxError> {
            let mut file = options.open(&tmp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        Ok(result?)
    })
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl_context_error!(pub SyncError);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zhuyin::SyllableVec;

    fn key(syl: &str, word: &str) -> (SyllableVec, String) {
        (syl.parse().unwrap(), word.to_string())
    }

    #[test]
    fn user_merge_keeps_both_sides() {
        let base = UserSnapshot::new();
        let ours = UserSnapshot::from([(key("ㄚ ㄇㄠ", "阿貓"), 10)]);
        let theirs = UserSnapshot::from([(key("ㄚ ㄍㄡˇ", "阿狗"), 10)]);
        let merged = merge_user(&base, &ours, &theirs);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn user_merge_honours_removals() {
        let base =
            UserSnapshot::from([(key("ㄚ ㄇㄠ", "阿貓"), 10), (key("ㄚ ㄍㄡˇ", "阿狗"), 10)]);
        let mut ours = base.clone();
        ours.remove(&key("ㄚ ㄇㄠ", "阿貓"));
        let mut theirs = base.clone();
        theirs.remove(&key("ㄚ ㄍㄡˇ", "阿狗"));
        assert!(merge_user(&base, &ours, &theirs).is_empty());
    }

    #[test]
    fn user_merge_conflict_keeps_ours() {
        let base = UserSnapshot::from([(key("ㄚ ㄇㄠ", "阿貓"), 10)]);
        let ours = UserSnapshot::from([(key("ㄚ ㄇㄠ", "阿貓"), 100)]);
        let theirs = UserSnapshot::from([(key("ㄚ ㄇㄠ", "阿貓"), 0)]);
        assert_eq!(
            merge_user(&base, &ours, &theirs)[&key("ㄚ ㄇㄠ", "阿貓")],
            100
        );
    }

    #[test]
    fn history_merge_adds_observations() {
        let k = key("ㄚ ㄇㄠ", "阿貓");
        let base = HistorySnapshot {
            generation: 10,
            entries: BTreeMap::from([(k.clone(), (2, 5))]),
            ..Default::default()
        };
        // We observed it 3 more times, others observed it 4 more times.
        let ours = HistorySnapshot {
            generation: 13,
            entries: BTreeMap::from([(k.clone(), (5, 5))]),
            ..Default::default()
        };
        let theirs = HistorySnapshot {
            generation: 14,
            entries: BTreeMap::from([(k.clone(), (6, 5))]),
            ..Default::default()
        };
        let merged = merge_history(&base, &ours, &theirs);
        assert_eq!(merged.generation, 17);
        assert_eq!(merged.entries[&k], (9, 5));
    }

    #[test]
    fn history_merge_shifts_new_local_epochs() {
        let k = key("ㄚ ㄇㄠ", "阿貓");
        let base = HistorySnapshot {
            generation: 10,
            ..Default::default()
        };
        let ours = HistorySnapshot {
            generation: 11,
            entries: BTreeMap::from([(k.clone(), (1, 11))]),
            ..Default::default()
        };
        let theirs = HistorySnapshot {
            generation: 20,
            ..Default::default()
        };
        let merged = merge_history(&base, &ours, &theirs);
        assert_eq!(merged.generation, 21);
        assert_eq!(merged.entries[&k], (1, 21));
    }
}

/// Two `UserDataSync` instances on one directory behave like two
/// processes sharing a user directory.
#[cfg(test)]
mod process_tests {
    use std::error::Error;

    use super::*;
    use crate::zhuyin::SyllableVec;

    struct Process {
        user: UserDict,
        hist: HistoryDict,
        sync: UserDataSync,
    }

    impl Process {
        fn open(dir: &Path) -> Process {
            let st = StringTable::new();
            let user = UserDict::open(dir.join(USER_DICT_FILE), st.clone())
                .unwrap_or_else(|_| UserDict::new(st.clone()));
            let hist = HistoryDict::open(dir.join(HISTORY_DICT_FILE), st.clone())
                .unwrap_or_else(|_| HistoryDict::new(st));
            let sync = UserDataSync::new(dir.to_path_buf(), &user, &hist);
            Process { user, hist, sync }
        }
        fn learn(&self, syl: &str, word: &str) {
            let syl: SyllableVec = syl.parse().unwrap();
            self.user.insert(syl.as_ref(), word);
            self.hist.observe(syl.as_ref(), word);
        }
        fn flush(&mut self) -> Result<(), SyncError> {
            self.sync.sync(&self.user, &self.hist)
        }
        fn words(&self) -> Vec<String> {
            let mut w: Vec<_> = self.user.entries().map(|(_, w)| w).collect();
            w.sort();
            w
        }
    }

    #[test]
    fn concurrent_writers_keep_each_others_words() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let mut a = Process::open(dir.path());
        let mut b = Process::open(dir.path());
        a.learn("ㄚ ㄇㄠ", "阿貓");
        a.flush()?;
        b.learn("ㄚ ㄍㄡˇ", "阿狗");
        b.flush()?;
        // b picked up a's word while merging.
        assert_eq!(b.words(), vec!["阿狗", "阿貓"]);
        let fresh = Process::open(dir.path());
        assert_eq!(fresh.words(), vec!["阿狗", "阿貓"]);
        assert_eq!(fresh.hist.snapshot().entries.len(), 2);
        // a sees b's word after its next sync.
        a.flush()?;
        assert_eq!(a.words(), vec!["阿狗", "阿貓"]);
        Ok(())
    }

    #[test]
    fn removal_in_one_process_propagates() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let mut a = Process::open(dir.path());
        a.learn("ㄚ ㄇㄠ", "阿貓");
        a.flush()?;
        let mut b = Process::open(dir.path());
        let syl: SyllableVec = "ㄚ ㄇㄠ".parse()?;
        b.user.remove(syl.as_ref(), "阿貓");
        b.flush()?;
        // a did not touch the word since its last sync, so the removal wins.
        a.flush()?;
        assert!(a.words().is_empty());
        Ok(())
    }

    #[test]
    fn corrupt_file_is_backed_up_and_local_data_kept() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let mut a = Process::open(dir.path());
        a.learn("ㄚ ㄇㄠ", "阿貓");
        a.flush()?;
        fs::write(dir.path().join(USER_DICT_FILE), "garbage without commas\n")?;
        a.learn("ㄚ ㄍㄡˇ", "阿狗");
        a.flush()?;
        let fresh = Process::open(dir.path());
        assert_eq!(fresh.words(), vec!["阿狗", "阿貓"]);
        let backups = fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
            .count();
        assert_eq!(backups, 1);
        Ok(())
    }

    #[test]
    fn leftover_temp_files_do_not_corrupt_output() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        // Leftovers from older versions and from a crashed writer.
        fs::write(dir.path().join("182e2c74c37b7090.tmp"), "x".repeat(4096))?;
        fs::write(
            dir.path().join(format!("{USER_DICT_FILE}.1-0.tmp")),
            "x".repeat(4096),
        )?;
        let mut a = Process::open(dir.path());
        a.learn("ㄚ ㄇㄠ", "阿貓");
        a.flush()?;
        let content = fs::read_to_string(dir.path().join(USER_DICT_FILE))?;
        assert!(!content.contains('x'));
        assert_eq!(Process::open(dir.path()).words(), vec!["阿貓"]);
        Ok(())
    }
}
