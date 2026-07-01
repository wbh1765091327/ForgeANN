use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::common::{AnnError, AnnResult};

#[inline]
pub fn with_rayon_thread_pool<R, F>(num_threads: u32, f: F) -> AnnResult<R>
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    if num_threads == 0 || num_threads == 1 {
        Ok(f())
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .map_err(|e| {
                AnnError::log_index_error(format!("Failed to create Rayon thread pool: {}", e))
            })?;
        Ok(pool.install(f))
    }
}

/// 使用Rayon线程池的并行执行函数
#[inline]
pub fn execute_with_rayon_thread_pool<F>(
    range: std::ops::Range<usize>,
    num_threads: u32,
    f: F,
) -> AnnResult<()>
where
    F: Fn(usize) -> AnnResult<()> + Sync + Send + Copy,
{
    if num_threads == 1 {
        for i in range {
            f(i)?;
        }
        Ok(())
    } else if num_threads == 0 {
        let tasks: Vec<usize> = range.collect();
        let results: Result<Vec<_>, _> = tasks.into_par_iter().map(f).collect();
        results.map(|_| ())
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .map_err(|e| {
                AnnError::log_index_error(format!("Failed to create Rayon thread pool: {}", e))
            })?;

        let tasks: Vec<usize> = range.collect();

        pool.install(|| {
            let results: Result<Vec<_>, _> = tasks.into_par_iter().map(f).collect();

            results.map(|_| ())
        })
    }
}

/// 使用Rayon线程池并支持线程局部状态初始化的并行执行函数
#[inline]
pub fn execute_with_rayon_thread_pool_init<F, I, S>(
    range: std::ops::Range<usize>,
    num_threads: u32,
    init: I,
    f: F,
) -> AnnResult<()>
where
    I: Fn() -> S + Sync + Send + Copy,
    F: Fn(&mut S, usize) -> AnnResult<()> + Sync + Send + Copy,
    S: Send,
{
    if num_threads == 1 {
        let mut state = init();
        for i in range {
            f(&mut state, i)?;
        }
        Ok(())
    } else if num_threads == 0 {
        let tasks: Vec<usize> = range.collect();
        let results: Result<Vec<_>, _> = tasks
            .into_par_iter()
            .map_init(init, |state, i| f(state, i))
            .collect();
        results.map(|_| ())
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .map_err(|e| {
                AnnError::log_index_error(format!("Failed to create Rayon thread pool: {}", e))
            })?;

        let tasks: Vec<usize> = range.collect();

        pool.install(|| {
            let results: Result<Vec<_>, _> = tasks
                .into_par_iter()
                .map_init(init, |state, i| f(state, i))
                .collect();

            results.map(|_| ())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use rayon::prelude::*;

    use super::with_rayon_thread_pool;
    use crate::common::AnnResult;

    #[test]
    fn nested_with_rayon_thread_pool_can_use_dedicated_inner_threads() -> AnnResult<()> {
        let outer_threads = Arc::new(Mutex::new(HashSet::new()));
        let inner_threads = Arc::new(Mutex::new(HashSet::new()));

        with_rayon_thread_pool(4, {
            let outer_threads = Arc::clone(&outer_threads);
            let inner_threads = Arc::clone(&inner_threads);
            move || -> AnnResult<()> {
                (0..64usize).into_par_iter().for_each(|_| {
                    outer_threads.lock().insert(std::thread::current().id());
                });

                with_rayon_thread_pool(4, {
                    let inner_threads = Arc::clone(&inner_threads);
                    move || {
                        (0..64usize).into_par_iter().for_each(|_| {
                            inner_threads.lock().insert(std::thread::current().id());
                        });
                    }
                })?;

                Ok(())
            }
        })??;

        let outer_threads = outer_threads.lock().clone();
        let inner_threads = inner_threads.lock().clone();

        assert!(
            outer_threads.len() > 1,
            "expected outer pool to use multiple workers"
        );
        assert!(
            inner_threads.len() > 1,
            "expected inner work to use multiple workers"
        );
        assert!(
            !inner_threads.is_subset(&outer_threads),
            "expected nested pool work to be able to use dedicated inner threads; outer={outer_threads:?}, inner={inner_threads:?}"
        );

        Ok(())
    }
}
