use std::default;

use monoio::time::Instant;

#[macro_export]
macro_rules! metric {
    ($($body:tt)*) => {
        #[cfg(feature = "metrics")]
        {
            $($body)*
        }
    };
}

#[macro_export]
macro_rules! access_pattern {
    ($($body:tt)*) => {
        // #[cfg(feature = "access_pattern")]
        // {
        //     $($body)*
        // }
    };
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerfMetrics {
    /// Total time to process query in micros
    pub total_us: usize,
    /// Total time spent in IO
    pub io_us: usize,
    /// Total time spent in CPU
    pub cpu_us: usize,

    /// total # of IOs issued
    pub n_ios: usize,
    /// # cache hits
    pub n_cache_hits: usize,
    /// # search hops
    pub n_hops: usize,

    /// # cmps
    pub n_cmps: usize,
    /// cmps saved
    pub n_cmps_saved: usize,

    /// prefetch time
    pub prefetch_us: usize,

    /// IO timer
    pub io_timer: Instant,
    /// CPU timer
    pub cpu_timer: Instant,
}

impl default::Default for PerfMetrics {
    fn default() -> Self {
        Self {
            total_us: 0,
            io_us: 0,
            cpu_us: 0,
            n_ios: 0,
            n_cache_hits: 0,
            n_hops: 0,
            n_cmps: 0,
            n_cmps_saved: 0,
            prefetch_us: 0,
            io_timer: Instant::now(),
            cpu_timer: Instant::now(),
        }
    }
}

impl PerfMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.total_us = 0;
        self.io_us = 0;
        self.cpu_us = 0;
        self.n_ios = 0;
        self.n_cache_hits = 0;
        self.n_hops = 0;
        self.n_cmps = 0;
        self.n_cmps_saved = 0;
        self.prefetch_us = 0;
    }
}
