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
