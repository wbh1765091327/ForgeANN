use std::ops::Range;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::common::AnnResult;

/// based on thread_num, execute the task in parallel using Rayon or serial
#[inline]
pub fn execute_with_rayon<F>(range: Range<usize>, num_threads: u32, f: F) -> AnnResult<()>
where
    F: Fn(usize) -> AnnResult<()> + Sync + Send + Copy,
{
    if num_threads == 1 {
        for i in range {
            f(i)?;
        }
        Ok(())
    } else {
        range.into_par_iter().try_for_each(f)
    }
}

/// set the thread count of Rayon, otherwise it will use threads as many as logical cores.
#[inline]
pub fn set_rayon_num_threads(num_threads: u32) {
    unsafe { std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string()) };
}
