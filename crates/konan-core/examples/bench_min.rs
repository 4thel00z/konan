//! Min-of-N throughput per strategy, in a form that runs unchanged against any
//! revision (it only touches the public `chunk` API).
//!
//! Criterion's cross-invocation comparisons are not trustworthy on a thermally
//! throttling laptop — the same config measured 521 µs cold and 1.66 ms warm
//! during this work. Reporting the minimum over many rounds and alternating
//! whole binaries between revisions is what actually survives that drift.
//!
//! Run: `cargo run --release -p konan-core --example bench_min`

use std::hint::black_box;
use std::time::{Duration, Instant};

use konan_core::{
    Chunker, FixedSizeChunker, MarkdownChunker, NaiveChunker, RecursiveChunker, SentenceChunker,
    TokenChunker,
};

const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dogs",
    "while",
    "seventy",
    "wizards",
    "briskly",
    "mix",
    "jugs",
    "of",
    "liquid",
    "oxygen",
    "under",
    "pale",
    "moonlight",
];

fn make_prose(n_bytes: usize) -> String {
    let mut out = String::with_capacity(n_bytes + 64);
    let mut w = 0usize;
    let mut sentence = 0usize;
    while out.len() < n_bytes {
        let len = 6 + (sentence * 7) % 12;
        for k in 0..len {
            let word = WORDS[(w + k * k) % WORDS.len()];
            if k == 0 {
                let mut chars = word.chars();
                let first = chars.next().expect("non-empty word").to_uppercase();
                out.extend(first);
                out.push_str(chars.as_str());
            } else {
                out.push_str(word);
            }
            out.push(if k + 1 == len { '.' } else { ' ' });
        }
        out.push(' ');
        w += len;
        sentence += 1;
        if sentence.is_multiple_of(8) {
            out.push_str("\n\n");
        }
    }
    out.truncate(n_bytes);
    out
}

fn make_markdown(n_bytes: usize) -> String {
    let mut out = String::with_capacity(n_bytes + 256);
    let mut section = 0usize;
    while out.len() < n_bytes {
        section += 1;
        out.push_str(&format!("# Section {section}\n\n"));
        for sub in 1..=2 {
            out.push_str(&format!("## Topic {section}.{sub}\n\n"));
            out.push_str(&make_prose(700));
            out.push_str("\n\n");
            if (section + sub).is_multiple_of(3) {
                out.push_str("```python\nfor i in range(10):\n    print(i)\n```\n\n");
            }
        }
    }
    out.truncate(n_bytes);
    out
}

/// CJK corpus: exercises the non-ASCII `count_chars` path, which the ASCII
/// prose corpus never touches.
fn make_cjk(n_bytes: usize) -> String {
    let mut out = String::with_capacity(n_bytes + 64);
    let mut n = 0usize;
    while out.len() < n_bytes {
        n += 1;
        out.push_str("日本語の文書です。これはテストの文章になります。");
        if n.is_multiple_of(6) {
            out.push_str("\n\n");
        }
    }
    while !out.is_char_boundary(n_bytes.min(out.len())) {
        out.pop();
    }
    out
}

fn bench(label: &str, rounds: usize, bytes: usize, mut f: impl FnMut()) {
    for _ in 0..2 {
        f();
    }
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed());
    }
    let s = best.as_secs_f64();
    println!(
        "{label:<26} {:>9.3} ms  {:>8.0} MB/s",
        s * 1e3,
        bytes as f64 / s / 1e6
    );
}

fn main() {
    let prose = make_prose(1_000_000);
    let markdown = make_markdown(1_000_000);
    let cjk = make_cjk(1_000_000);
    let rounds = 25;

    println!("# 1 MB corpora, min of {rounds} rounds");

    let naive = NaiveChunker::new(200).expect("valid config");
    bench("naive", rounds, prose.len(), || {
        black_box(naive.chunk(black_box(&prose)).expect("chunks"));
    });

    let fixed = FixedSizeChunker::new(1000, 200, true).expect("valid config");
    bench("fixed_size", rounds, prose.len(), || {
        black_box(fixed.chunk(black_box(&prose)).expect("chunks"));
    });

    let recursive = RecursiveChunker::new(1000, 200, None).expect("valid config");
    bench("recursive", rounds, prose.len(), || {
        black_box(recursive.chunk(black_box(&prose)).expect("chunks"));
    });

    let sentence = SentenceChunker::new(1000, 1).expect("valid config");
    bench("sentence", rounds, prose.len(), || {
        black_box(sentence.chunk(black_box(&prose)).expect("chunks"));
    });

    let md = MarkdownChunker::new(1000, 200).expect("valid config");
    bench("markdown", rounds, markdown.len(), || {
        black_box(md.chunk(black_box(&markdown)).expect("chunks"));
    });

    let token = TokenChunker::new(512, 64, "cl100k_base").expect("valid config");
    bench("token", 8, prose.len(), || {
        black_box(token.chunk(black_box(&prose)).expect("chunks"));
    });

    bench("fixed_size/cjk", rounds, cjk.len(), || {
        black_box(fixed.chunk(black_box(&cjk)).expect("chunks"));
    });

    bench("recursive/cjk", rounds, cjk.len(), || {
        black_box(recursive.chunk(black_box(&cjk)).expect("chunks"));
    });

    // Parallel path: 64 documents through rayon.
    let docs: Vec<String> = (0..64).map(|_| make_prose(256_000)).collect();
    let total: usize = docs.iter().map(String::len).sum();
    bench("chunk_many/64x256kb", 10, total, || {
        black_box(konan_core::chunk_many(&recursive, black_box(&docs)).expect("chunks"));
    });
}
