use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc}
use std::thread{self, JoinHandle};
use std::time::{Duration, Instant};

// TODO - calling magical things atm, but mimicking the python code
use crate::{
    Sequence, SamplingParams, SequenceStatus, SequenceError,
    scheduler::{Scheduler, SchedulerConfig, ScheduleResult, SchedulerError},
    model_runner::{ModelRunner, ModelRunnerConfig, ModelRunnerResult, ModelRunnerError},
}

//config for LLM Engine

#[derive(Debug, Clone)]
pub struct LLMEngineConfig {
    pub model_path: String,
    pub tensor_parallel_size: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
    pub block_size: i32,
    pub gpu_memory_utilization: f32,
    pub enforce_eager: bool,
    pub eos_token_id: Option<u32>, // 64?
    pub device_ids: Vec<i32>,
    pub max_chunked_prefill_size: usize,
    pub worker_timeout: Duration,
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
            enable_chunked_prefill: true,
            max_chunked_prefill_size: 4096,
            worker_timeout: Duration::from_secs(30),
        }
    }
}

//update error tpes
#[derive(Debug, thiserror::Error)]
pub enum LLMEngineError {
    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("Tokenize error: {0}")]
    Tokenizer(String),

    #[error("Scheduler error: {source}")]
    Scheduler {
        #[from]
        source: SchedulerError,
    },

    #[error("Model runner error: {source}")]
    ModelRunner {
        #[from]
        source: ModelRunnerError,
    },

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Sequence error: {source}")]
    Sequence {
        #[from]
        source: SequenceError,
    },

    #[error("Worker thread error: {0}",) ]
    Worker(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Configuration error: {0}")]
    Config(String),
}



pub type LLMEngineResult<T> = Result<T, LLMEngineError>;

// Mock Tokenizer for now
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, LLMEngineError>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, LLMEngineError>;
    fn eos_token_id(&self) -> u32;
    fn vocab_size(&self) -> usize;
}



// Generation stats
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prefill_throughput: f32,
    pub decode_throughput: f32,
    pub total_tokens_generated: usize,
    pub total_time: Duration,
    pub sequencess_completed: usize,
    pub sequences_failed: usize,
    pub average_latency: Duration,
    pub peak_memory_usage: usize,
    pub cache_hit_rate: f32,
}

impl GenerationStats {
    pub fn update_throughput(&mut self, tokens: usize, duration: Duration, is_prefill: bool) {
        let throughput = tokens as f32 / duration.as_secs_f32();
        if is_prefill {
            self.prefill_throughput = throughput;
        } else {
            self.decode_throughput = throughput;
        }
    }

    pub fn complete_sequence(&mut self, tokens_generated: usize, latency: Duration) {
        self.sequences_completed += 1;
        self.total_tokens_generated += tokens_generated;
        self.average_latency = if self.sequences_completed == 1 {
            latency
        } else {
            Duration::from_nanos(
                (self.average_latency.as_nanos() as u64 + latency.as_nanos() as u64) / 2
            )
        };
    }

    pub fn fail_sequence(&mut self) {
        self.sequences_failed += 1;
    }
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
