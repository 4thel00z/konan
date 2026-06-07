//! Criterion micro-benches for every CPU-bound strategy.
//!
//! Run: `cargo bench -p konan-core`

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

use konan_core::{
    chunk_many, Chunker, FixedSizeChunker, MarkdownChunker, NaiveChunker, RecursiveChunker,
    SentenceChunker, TokenChunker,
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

/// Deterministic ~`n_bytes` of prose: sentences of varying length, paragraph
/// breaks every few sentences. No RNG dependency needed.
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
                let first = chars.next().unwrap().to_uppercase();
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

fn bench_strategies(c: &mut Criterion) {
    let prose = make_prose(1_000_000);
    let markdown = make_markdown(1_000_000);

    let mut group = c.benchmark_group("chunk_1mb");
    group.throughput(Throughput::Bytes(prose.len() as u64));
    group.sample_size(20);

    let naive = NaiveChunker::new(200).unwrap();
    group.bench_function("naive", |b| {
        b.iter(|| naive.chunk(black_box(&prose)).unwrap())
    });

    let fixed = FixedSizeChunker::new(1000, 200, true).unwrap();
    group.bench_function("fixed_size", |b| {
        b.iter(|| fixed.chunk(black_box(&prose)).unwrap())
    });

    let recursive = RecursiveChunker::new(1000, 200, None).unwrap();
    group.bench_function("recursive", |b| {
        b.iter(|| recursive.chunk(black_box(&prose)).unwrap())
    });

    let sentence = SentenceChunker::new(1000, 1).unwrap();
    group.bench_function("sentence", |b| {
        b.iter(|| sentence.chunk(black_box(&prose)).unwrap())
    });

    let md = MarkdownChunker::new(1000, 200).unwrap();
    group.bench_function("markdown", |b| {
        b.iter(|| md.chunk(black_box(&markdown)).unwrap())
    });

    let token = TokenChunker::new(512, 64, "cl100k_base").unwrap();
    group.bench_function("token", |b| {
        b.iter(|| token.chunk(black_box(&prose)).unwrap())
    });

    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let docs: Vec<String> = (0..64).map(|_| make_prose(256_000)).collect();
    let total: usize = docs.iter().map(String::len).sum();
    let recursive = RecursiveChunker::new(1000, 200, None).unwrap();

    let mut group = c.benchmark_group("chunk_many_64x256kb");
    group.throughput(Throughput::Bytes(total as u64));
    group.sample_size(20);

    group.bench_function("sequential", |b| {
        b.iter(|| {
            docs.iter()
                .map(|d| recursive.chunk(black_box(d)).unwrap())
                .collect::<Vec<_>>()
        })
    });
    group.bench_function("rayon", |b| {
        b.iter(|| chunk_many(&recursive, black_box(&docs)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_strategies, bench_parallel);
criterion_main!(benches);
