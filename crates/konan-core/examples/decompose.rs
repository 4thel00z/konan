//! Throwaway decomposition of FixedSizeChunker + SentenceChunker cost.
//! Run: `cargo run --release -p konan-core --example decompose`

use std::hint::black_box;
use std::time::Instant;

use konan_core::{Chunker, FixedSizeChunker, SentenceChunker};

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

fn time<F: FnMut()>(label: &str, mut f: F) {
    // Warm up once, then measure best-of-5.
    f();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("{label:32} {:8.3} ms", best * 1e3);
}

fn main() {
    let text = make_prose(4 << 20);
    println!("corpus: {} bytes", text.len());

    let re = regex::Regex::new(r"[.!?]+\s+").unwrap();
    time("regex find_iter only", || {
        black_box(re.find_iter(&text).count());
    });

    let fixed = FixedSizeChunker::new(1000, 200, true).unwrap();
    time("fixed_size total", || {
        black_box(fixed.chunk(&text).unwrap());
    });

    let fixed_no_sent = FixedSizeChunker::new(1000, 200, false).unwrap();
    time("fixed_size respect=false", || {
        black_box(fixed_no_sent.chunk(&text).unwrap());
    });

    let sentence = SentenceChunker::new(1000, 1).unwrap();
    time("sentence total", || {
        black_box(sentence.chunk(&text).unwrap());
    });

    // Sentence segmentation alone (the dominant sentence-chunker cost).
    let segmenter = icu_segmenter::SentenceSegmenter::new(Default::default());
    time("icu sentence bounds only", || {
        black_box(segmenter.segment_str(&text).count());
    });
}
