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


/// Status of a sequence in the generation pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
    Error(sequenceError),
}

/// Types of errors that can occur during generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceError {
    MaxTokensExceeded,
    TokenizationFailed,
    GenerationFailed,
    InvalidBlockSize,
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub seq_id: usize,
    pub status: SequenceStatus,
    // All token ids (prompt + generated) in the sequence
    token_ids: Vec<u32>,
    // most recently generated token
    last_token: u32,
    pub num_prompt_tokens: usize,
    // Number of tokens cached in blocks
    pub num_cached_tokens: usize,
    // block ids allocated for this sequence
    pub block_ids: Vec<usize>,
    sampling_params: SamplingParams,
    created_at: std::time::Instant,
}

impl Sequence {
    pub const DEFAULT_BLOCK_SIZE: usize = 256;
    // Global sequence counter, unique ids for each sequence
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Create a new sequence with the given prompt and sampling parameters
    pub fn new(token_ids: Vec<u32>, sampling_params: Option<SamplingParams>) -> Result<Self, SequenceError> {
        if token_ids.is_empty() {
            return Err(SequenceError::TokenizationFailed);
        }

        let params = sampling_params.unwrap_or_default();
        let last_token = *token_ids.last().unwrap();
        let num_prompt_tokens = token_ids.len();

        Ok(Self {
            seq_id: Self::COUNTER.fetch_add(1, Ordering::Relaxed),
            status: SequenceStatus::Waiting,
            last_token,
            num_prompt_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            sampling_params: params,
            token_ids,
            created_at: std::time::Instant::now(),
        })
    }

    // getters
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
