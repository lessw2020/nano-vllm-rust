use std::collections::HashMap;
use std::sync::{Arc, Mutex}
use std::thread;
use std::time::{Duration, Instant};

// TODO - calling magical things atm, but mimicking the python code
use crate::{
    Sequence, SamplingParams, SequenceStatus,
    scheduler::{Scheduler, SchedulerConfig, ScheduleResult},
    model_runner::{ModelRunner, ModelRunnerConfig, ModelRunnerResult},
}

//config for LLM Engine

#[derive(Debug, Clone)]
pub struct LLMEngineConfig {
    pub model_path: String,
    pub tensor_parallel_size: usize,
    pub max_num_steps: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
    pub block_size: i32,
    pub gpu_memory_utilization: f32,
    pub enforce_eager: bool,
    pub eos_token_id: Option<u32>, // 64?
    pub device_ids: Vec<i32>,
}

impl Default for LLMEngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tensor_parallel_size: 1,
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_model_len: 4096,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            enforce_eager: false,
            eos_token_id: None,
            device_ids: vec![0],
        }
    }
}

// Error types for LLM Engine
#[derive(Debug)]
pub enum LLMEngineError {
    InitializationError(String),
    TokenizeError(String),
    SchedulerError(String),
    ModelRunnerError(String),
    GenerationError(String),
}

impl std::fmt::Display for LLMEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMEngineError::InitializationError(msg) => write!(f, "LLMEngineError: {}", msg),
            LLMEngineError::TokenizeError(msg) => write!(f, "LLMEngineError: {}", msg),
            LLMEngineError::SchedulerError(msg) => write!(f, "LLMEngineError: {}", msg),
            LLMEngineError::ModelRunnerError(msg) => write!(f, "LLMEngineError: {}", msg),
            LLMEngineError::GenerationError(msg) => write!(f, "LLMEngineError: {}", msg),
        }
    }
}

impl std::error::Error for LLMEngineError {}

impl From<crate::scheeduler::SchedulerError> for LLMEngineError {
    fn from (err: crate::scheeduler::SchedulerError) -> Self {
        LLMEngineError::SchedulerError(err.to_string())
    }
}

impl From<crate::model_runner::ModelRunnerError> for LLMEngineError {
    fn from (err: crate::model_runner::ModelRunnerError) -> Self {
        LLMEngineError::ModelRunnerError(err.to_string())
    }
}

pub type LLMEngineResult<T> = Result<T, LLMEngineError>;

// Mock Tokenizer for now
pub struct Tokenizer {
    vocab_size: usize,
    eos_token_id: u32,
    vocab: HashMap<String, u32>,,
    inverse_vocab: HashMap<u32, String>,
}

impl Tokenizer {
    pub fn new(vocab_size: usize, eos_token_id: u32) -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();

        //simple vocab
        for i in 0..vocab_size {
            let token = format!("token_{}", i);
            vocab.insert(token.clone9(), i as u32);
            inverse_vocab.insert(i as u32, token);
        }
       //special tokens
       vocab.insert("<eos>".to_string(), eos_token_id);
       inverse_vocab.insert(eos_token_id, "<eos>".to_string());

        Self {
            vocab_size,
            eos_token_id,
            vocab,
            inverse_vocab,
        }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        //word based for now
        text.split_whitespace()
            .map(|token| *self.vocab.get(token).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        token_ids.iter()
            .filter_map(|&id| self.inverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

}

// Generation stats
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prefill_throughput: f32,
    pub decode_throughput: f32,
    pub total_tokens_generated: usize,
    pub total_time: Duration,
    pub sequencess_completed: usize,
}


// output of LLM Engine Generation Reequest
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    Error,
}

// Progress bar for generation
pub type ProgressCallback = Box<dyn Fn(usize, usize, &GenerationStats) + Send + Sync>;


// Main LLM Engine struct
pub struct LLMEngine {
    config: LLMEngineConfig,
    tokenizer: Tokenizer,
    scheduler: Scheduler,
    model_runner: ModelRunner,
    worker_handles: Vec<thread::JoinHandle<()>>,
    stats: Arc<Mutex<GenerationStats>>,
}

impl LLMEngine {
    pub fn new(config: LLMEngineConfig) -> LLMEngineResult<Self> {
        // init tokeniszer - TODO - this is standin tokenizer for now
        let eos_token_id = config.eos_token_id.unwrap_or(2);
        let tokenizer = Tokenizer::new(32_000, eos_token_id);

        //Create scheduler config
        let scheduler_config = SchedulerConfig {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos_token_id,
            num_kvcache_blocks: 1000, // TODO - this is hardcoded for now
            kvcache_block_size: config.block_size,
        };

        //init scheduler
        // TODO - continue here.........
    }
}
