use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about)]
pub(crate) struct ChewingCli {
    #[command(subcommand)]
    pub(crate) subcommand: ChewingCliCmd,
}

#[derive(Subcommand)]
pub(crate) enum ChewingCliCmd {
    /// Display information about the file
    Info(Info),
    /// List detected system and user file
    List(List),
    /// Convert a binary format to its text format
    Dump(Dump),
    /// Subcommands for dictionary index manipulation
    #[command(subcommand)]
    Index(Index),
    /// Subcommands for language model manipulation
    #[command(subcommand)]
    Lm(Lm),
}

#[derive(Subcommand)]
pub(crate) enum Index {
    /// Create dictionary index file
    CreateDict(IndexCreateDict),
    /// Create string table index file
    CreateStringTable(IndexCreateStringTable),
}

#[derive(Args)]
pub(crate) struct IndexCreateDict {
    /// Path to the dictionary source file (tsi.csv)
    pub(crate) tsi_csv: PathBuf,
    /// Path to the words list (static_words.txt)
    pub(crate) words_txt: PathBuf,
    /// Path to the output file (static_dict.bin)
    pub(crate) dict_output: PathBuf,
}

#[derive(Args)]
pub(crate) struct IndexCreateStringTable {
    /// Path to the words list (static_words.txt)
    pub(crate) words_txt: PathBuf,
    /// Path to the output file (static_words.bin)
    pub(crate) words_output: PathBuf,
}

#[derive(Subcommand)]
pub(crate) enum Lm {
    /// Create binary language model from ARPA file
    Compile(LmCompile),
    /// Segment a string
    Segment(LmSegment),
    /// Learn unigram and bigram language model and output ARPA file
    Learn(LmLearn),
    /// Prepare data for evaluation
    Prepare(LmPrepare),
    /// Evaluate the conversion accuracy of a model
    Eval(LmEval),
    /// C1 pilot: build scored input units from documents (JSONL)
    C1Prepare(LmC1Prepare),
    /// C1 pilot: decode units from an empty start and from the previous word
    C1Run(LmC1Run),
}

#[derive(Args)]
pub(crate) struct LmC1Prepare {
    /// Directory with static_words.bin, static_dict.bin, rare_dict.bin, static_lm.bin
    pub(crate) data_dir: PathBuf,
    /// Word readings (chewing_v4/tsi_dict.csv)
    pub(crate) tsi_dict: PathBuf,
    /// Rare readings (chewing_v4/rare_dict.csv)
    pub(crate) rare_dict: PathBuf,
    /// Old dictionary with candidate priorities (chewing/tsi.csv)
    pub(crate) old_tsi: PathBuf,
    /// Documents, one JSON object per line with doc_id and text
    pub(crate) docs: PathBuf,
    /// Output units (JSONL)
    pub(crate) output: PathBuf,
    /// Seed for the dev/test document split
    #[arg(long, default_value_t = 20261005)]
    pub(crate) seed: u64,
}

#[derive(Args)]
pub(crate) struct LmC1Run {
    /// Directory with static_words.bin, static_dict.bin, rare_dict.bin, static_lm.bin
    pub(crate) data_dir: PathBuf,
    /// Units produced by c1-prepare
    pub(crate) units: PathBuf,
    /// Which split to evaluate (dev or test)
    pub(crate) split: String,
    /// Output per-unit results (JSONL)
    pub(crate) output: PathBuf,
}

#[derive(Args)]
pub(crate) struct LmCompile {
    /// Path to the language model ARPA file (static_lm.arpa)
    pub(crate) lm_arpa: PathBuf,
    /// Path to the words list (static_words.txt)
    pub(crate) words_txt: PathBuf,
    /// Path to the output file (static_lm.bin)
    pub(crate) output: PathBuf,
}

#[derive(Args)]
pub(crate) struct LmSegment {
    /// Path to the language model file (static_lm.bin)
    pub(crate) static_lm: PathBuf,
    /// Path to the words list (static_words.txt)
    pub(crate) words_txt: PathBuf,
}

#[derive(Args)]
pub(crate) struct LmLearn {
    /// Path to the words list (static_words.txt)
    pub(crate) words_txt: PathBuf,
    /// Path to the output ARPA file (static_lm.arpa)
    pub(crate) output: PathBuf,
    /// Minimum bigram count to consider (hard floor).
    #[arg(default_value_t = 10)]
    pub min_count: u64,
    /// Fraction of bigrams to keep after pruning (e.g. 0.2 = keep 20%).
    /// If `None`, falls back to the min_count threshold only.
    pub(crate) keep_fraction: Option<f64>,
}

#[derive(Args)]
pub(crate) struct LmPrepare {
    /// Path to the dictionary source file (tsi.csv)
    pub(crate) tsi_csv: PathBuf,
    /// Path to the rare word source file (rare.csv)
    pub(crate) rare_csv: PathBuf,
}

#[derive(Args)]
pub(crate) struct LmEval {
    /// Model file search path
    pub(crate) search_path: String,
    /// File name of the language model
    pub(crate) model_bin: String,
    /// Decoder hyperparameter alpha
    pub(crate) alpha: f64,
    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Args)]
pub(crate) struct Info {
    /// Location of the file
    #[arg(short, long)]
    pub(crate) path: PathBuf,
    /// Output in JSON format
    #[arg(short, long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct List {
    /// Display information of detected user dictionary
    #[arg(short, long)]
    pub(crate) user: bool,
    /// Display information of detected system dictionary
    #[arg(short, long)]
    pub(crate) system: bool,
    /// Output in JSON format
    #[arg(short, long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct Dump {
    /// Location of the dictionary file
    pub(crate) path: PathBuf,
    /// Location of the output file
    ///
    /// If OUTPUT equals to `-` then standard output will be used.
    pub(crate) output: Option<PathBuf>,
}
