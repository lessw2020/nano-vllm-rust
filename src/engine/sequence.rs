use std::sync::atomic::{AtomicUsize, Ordering};

// sampling params
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.0,
            top_k: 50,
            max_tokens: 100,
            ignore_eos: false,
            stop_tokens: Vec::new(),
        }
    }
}
