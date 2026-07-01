#![warn(missing_debug_implementations, missing_docs)]

//! In-memory Dataset

use std::fs::File;
use std::mem;
use std::ops::{Mul, Sub};
use std::path::Path;
#[cfg(test)]
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use memmap2::Mmap;
use rayon::prelude::*;
use tracing::info;

use crate::common::{AlignedBoxWithSlice, AnnError, AnnResult, Metric};
use crate::model::Vertex;
use crate::utils::copy_aligned_data_from_file;
use crate::utils::thread_pool::with_rayon_thread_pool;

/// Dataset of all in-memory FP points
#[derive(Debug)]
pub struct InmemDataset<T> {
    /// All in-memory points or memory-mapped file data
    pub data: AlignedBoxWithSlice<T>,
    /// Memory-mapped file for larger-than-memory datasets
    mmap: Option<Mmap>,
    /// File handle for mmap (needs to be kept open)
    _file: Option<File>,
    /// Number of points we anticipate to have
    pub num_points: usize,
    /// Number of active points i.e. existing in the graph
    pub num_active_pts: usize,
    /// Capacity of the dataset
    pub capacity: usize,
    /// The dimension of the dataset
    pub dim: usize,
    /// Whether this dataset uses memory mapping
    is_memory_mapped: bool,
}

#[cfg(test)]
pub(crate) trait MedoidThreadObserver: Send + Sync {
    fn on_nearest_point_worker(&self, worker: usize);
}

#[cfg(test)]
fn medoid_thread_observer() -> &'static StdMutex<Option<Arc<dyn MedoidThreadObserver>>> {
    static OBSERVER: OnceLock<StdMutex<Option<Arc<dyn MedoidThreadObserver>>>> = OnceLock::new();
    OBSERVER.get_or_init(|| StdMutex::new(None))
}

#[cfg(test)]
pub(crate) fn install_medoid_thread_observer(
    observer: Option<Arc<dyn MedoidThreadObserver>>,
) -> Option<Arc<dyn MedoidThreadObserver>> {
    let mut guard = medoid_thread_observer().lock().unwrap();
    std::mem::replace(&mut *guard, observer)
}

#[cfg(test)]
fn record_medoid_worker() {
    if let Some(worker) = rayon::current_thread_index() {
        if let Some(observer) = medoid_thread_observer().lock().unwrap().as_ref().cloned() {
            observer.on_nearest_point_worker(worker);
        }
    }
}

impl<'a, T> InmemDataset<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    /// Create the dataset with size num_points and growth factor.
    /// growth factor=1 means no growth (provision 100% space of num_points)
    /// growth factor=1.2 means provision 120% space of num_points (20% extra space)
    pub fn new(num_points: usize, index_growth_factor: f32, dim: usize) -> AnnResult<Self> {
        let capacity = (((num_points * dim) as f32) * index_growth_factor) as usize;

        Ok(Self {
            data: AlignedBoxWithSlice::new(capacity, mem::size_of::<T>() * 16)?,
            mmap: None,
            _file: None,
            num_points,
            num_active_pts: num_points,
            capacity,
            dim,
            is_memory_mapped: false,
        })
    }

    /// Create a dataset intended for memory-mapped access without allocating a full backing buffer.
    pub fn new_memory_mapped(num_points: usize, dim: usize) -> AnnResult<Self> {
        Ok(Self {
            data: AlignedBoxWithSlice::new_empty(mem::size_of::<T>() * 16)?,
            mmap: None,
            _file: None,
            num_points,
            num_active_pts: num_points,
            capacity: 0,
            dim,
            is_memory_mapped: false,
        })
    }

    /// Configure this dataset to use memory-mapped file access
    pub fn set_memory_mapped(&mut self, filename: &Path) -> AnnResult<()> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // Calculate number of points from file size and dimension
        let header_size = 8; // 4 bytes for num_points, 4 bytes for dim
        let data_size = file_size - header_size;
        let num_points = data_size / (self.dim * mem::size_of::<T>());

        // Memory map the file
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header to verify dimensions
        if mmap.len() < header_size {
            return Err(AnnError::log_index_error(
                "File too small for header".to_string(),
            ));
        }

        let file_num_points = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
        let file_dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;

        if file_dim != self.dim {
            return Err(AnnError::log_index_error(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, file_dim
            )));
        }

        if file_num_points != num_points {
            return Err(AnnError::log_index_error(format!(
                "Point count mismatch: expected {}, got {}",
                num_points, file_num_points
            )));
        }

        // Update dataset configuration for memory-mapped mode
        self.mmap = Some(mmap);
        self._file = Some(file);
        self.num_points = num_points;
        self.num_active_pts = num_points;
        self.capacity = num_points * self.dim;
        self.is_memory_mapped = true;

        Ok(())
    }

    /// get immutable data slice
    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    /// Build the dataset from file
    pub fn build_from_file(&mut self, filename: &Path, num_points_to_load: usize) -> AnnResult<()> {
        info!(
            "Loading {} vectors from file {:?} into dataset...",
            num_points_to_load, filename
        );

        if self.is_memory_mapped {
            // In memory-mapped mode, data should be loaded via set_memory_mapped()
            // We just need to set the number of active points
            info!("Using memory-mapped file access, data already loaded");
            self.num_active_pts = num_points_to_load;
        } else {
            // Traditional in-memory mode: copy data from file
            self.num_active_pts = num_points_to_load;
            copy_aligned_data_from_file(filename, self.into_dto(), 0)?;
        }

        info!("Dataset loaded.");
        Ok(())
    }

    /// Append the dataset from file
    pub fn append_from_file(
        &mut self,
        filename: &Path,
        num_points_to_append: usize,
    ) -> AnnResult<()> {
        info!(
            "Appending {} vectors from file {:?} into dataset...",
            num_points_to_append, filename
        );

        if self.is_memory_mapped {
            // Memory-mapped datasets don't support appending
            return Err(AnnError::log_index_error(
                "Cannot append to memory-mapped datasets".to_string(),
            ));
        }

        if self.num_points + num_points_to_append > self.capacity {
            return Err(AnnError::log_index_error(format!(
                "Cannot append {} points to dataset of capacity {}",
                num_points_to_append, self.capacity
            )));
        }

        let pts_offset = self.num_active_pts;
        copy_aligned_data_from_file(filename, self.into_dto(), pts_offset)?;

        self.num_active_pts += num_points_to_append;
        self.num_points += num_points_to_append;

        println!("Dataset appended.");
        Ok(())
    }

    /// Get vertex by id
    pub fn get_vertex(&'a self, id: u32) -> AnnResult<Vertex<'a, T>> {
        let start = id as usize * self.dim;
        let end = start + self.dim;

        if self.is_memory_mapped {
            // Memory-mapped mode: access data directly from mmap
            if let Some(mmap) = &self.mmap {
                let header_size = 8; // 4 bytes for num_points, 4 bytes for dim
                let data_start = header_size + start * mem::size_of::<T>();
                let data_end = header_size + end * mem::size_of::<T>();

                if data_end > mmap.len() {
                    return Err(AnnError::log_index_error(format!(
                        "Invalid vertex id {}: out of bounds",
                        id
                    )));
                }

                // Safe type conversion: we know the memory is properly aligned and sized
                let val = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr().add(data_start) as *const T, self.dim)
                };
                Ok(Vertex::new(val, id))
            } else {
                Err(AnnError::log_index_error(
                    "Memory mapping not available".to_string(),
                ))
            }
        } else {
            // Traditional in-memory mode
            if end <= self.data.len() {
                let val = &self.data[start..end];
                Ok(Vertex::new(val, id))
            } else {
                Err(AnnError::log_index_error(format!(
                    "Invalid vertex id {}.",
                    id
                )))
            }
        }
    }

    /// Get full precision distance between two nodes
    pub fn get_distance(&self, id1: u32, id2: u32, metric: Metric) -> AnnResult<f32> {
        let vertex1 = self.get_vertex(id1)?;
        let vertex2 = self.get_vertex(id2)?;

        Ok(vertex1.compare(&vertex2, metric))
    }

    /// find out the medoid, the vertex in the dataset that is closest to the centroid
    pub fn calculate_medoid_point_id(&self) -> AnnResult<u32> {
        self.calculate_medoid_point_id_with_threads(None)
    }

    /// find out the medoid, the vertex in the dataset that is closest to the centroid
    /// with an explicit hard thread cap for the nearest-point search.
    pub fn calculate_medoid_point_id_with_threads(
        &self,
        num_threads: Option<u32>,
    ) -> AnnResult<u32> {
        let center = self.calculate_centroid_point()?;
        if rayon::current_thread_index().is_some() {
            Ok(self.find_nearest_point_id(&center))
        } else {
            with_rayon_thread_pool(
                num_threads.unwrap_or_else(|| {
                    std::env::var("RAYON_NUM_THREADS")
                        .ok()
                        .and_then(|value| value.parse::<u32>().ok())
                        .filter(|&threads| threads > 0)
                        .unwrap_or_else(|| {
                            std::thread::available_parallelism()
                                .map(|parallelism| parallelism.get() as u32)
                                .unwrap_or(1)
                        })
                }),
                || self.find_nearest_point_id(&center),
            )
        }
    }

    /// calculate centroid, average of all vertices in the dataset
    fn calculate_centroid_point(&self) -> AnnResult<Vec<f32>> {
        // Allocate and initialize the centroid vector
        let mut center: Vec<f32> = vec![0.0; self.dim];

        if self.is_memory_mapped {
            // Memory-mapped mode: access data directly from mmap
            if let Some(mmap) = &self.mmap {
                let header_size = 8;
                for i in 0..self.num_active_pts {
                    let data_start = header_size + i * self.dim * mem::size_of::<T>();
                    let data_end = data_start + self.dim * mem::size_of::<T>();

                    if data_end <= mmap.len() {
                        let vertex_data = unsafe {
                            std::slice::from_raw_parts(
                                mmap.as_ptr().add(data_start) as *const T,
                                self.dim,
                            )
                        };
                        for j in 0..self.dim {
                            center[j] += vertex_data[j].into();
                        }
                    }
                }
            }
        } else {
            // Traditional in-memory mode
            for i in 0..self.num_active_pts {
                let vertex = self.get_vertex(i as u32)?;
                let vertex_slice = vertex.vector();
                for j in 0..self.dim {
                    center[j] += vertex_slice[j].into();
                }
            }
        }

        // Divide by the number of points to calculate the centroid
        let capacity = self.num_active_pts as f32;
        for item in center.iter_mut().take(self.dim) {
            *item /= capacity;
        }

        Ok(center)
    }

    /// find out the vertex closest to the given point
    fn find_nearest_point_id(&self, point: &[f32]) -> u32 {
        // compute all to one distance
        let mut distances = vec![0f32; self.num_active_pts];

        if self.is_memory_mapped {
            // Memory-mapped mode: access data directly from mmap
            if let Some(mmap) = &self.mmap {
                let header_size = 8;
                distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
                    #[cfg(test)]
                    if i % 2048 == 0 {
                        record_medoid_worker();
                    }
                    let start = header_size + i * self.dim * mem::size_of::<T>();
                    let end = start + self.dim * mem::size_of::<T>();

                    if end <= mmap.len() {
                        let vertex_data = unsafe {
                            std::slice::from_raw_parts(
                                mmap.as_ptr().add(start) as *const T,
                                self.dim,
                            )
                        };
                        for j in 0..self.dim {
                            let diff: f32 = (point[j] - vertex_data[j].into())
                                * (point[j] - vertex_data[j].into());
                            *dist += diff;
                        }
                    }
                });
            }
        } else {
            // Traditional in-memory mode
            let slice = &self.data[..];
            distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
                #[cfg(test)]
                if i % 2048 == 0 {
                    record_medoid_worker();
                }
                let start = i * self.dim;
                for j in 0..self.dim {
                    let diff: f32 =
                        (point[j] - slice[start + j].into()) * (point[j] - slice[start + j].into());
                    *dist += diff;
                }
            });
        }

        let mut min_idx = 0;
        let mut min_dist = f32::MAX;
        for (i, distance) in distances.iter().enumerate().take(self.num_active_pts) {
            if *distance < min_dist {
                min_idx = i;
                min_dist = *distance;
            }
        }
        min_idx as u32
    }

    /// Prefetch vertex data in the memory hierarchy
    /// NOTE: good efficiency when total_vec_size is integral multiple of 64
    #[inline]
    pub fn prefetch_vector(&self, id: u32) {
        let start = id as usize * self.dim;
        let end = start + self.dim;

        if self.is_memory_mapped {
            // For memory-mapped files, prefetching is handled by the OS
            // We can optionally add madvise calls for optimization
            if let Some(mmap) = &self.mmap {
                let header_size = 8;
                let data_start = header_size + start * mem::size_of::<T>();
                let data_end = header_size + end * mem::size_of::<T>();

                if data_end <= mmap.len() {
                    // Advise the kernel that we'll be accessing this range
                    #[cfg(target_os = "linux")]
                    unsafe {
                        use libc::{MADV_WILLNEED, madvise};
                        let ptr = mmap.as_ptr().add(data_start) as *mut libc::c_void;
                        let len = data_end - data_start;
                        madvise(ptr, len, MADV_WILLNEED);
                    }
                }
            }
        } else {
            // Traditional in-memory mode
            if end <= self.data.len() {
                let _vec = &self.data[start..end];
                // todo prefetch - could use __builtin_prefetch for in-memory data
            }
        }
    }

    /// Convert into dto object
    #[allow(clippy::wrong_self_convention)]
    pub fn into_dto(&'_ mut self) -> DatasetDto<'_, T> {
        if self.is_memory_mapped {
            // In memory-mapped mode, we cannot provide mutable access to data
            // Return an empty slice - this should only be used for compatibility
            // and not for actual data operations
            DatasetDto {
                data: &mut [],
                rounded_dim: self.dim,
            }
        } else {
            DatasetDto {
                data: &mut self.data,
                rounded_dim: self.dim,
            }
        }
    }

    /// Check if this dataset uses memory mapping
    pub fn is_memory_mapped(&self) -> bool {
        self.is_memory_mapped
    }

    /// Get the memory mapped file size if using memory mapping
    pub fn mmap_size(&self) -> Option<usize> {
        self.mmap.as_ref().map(|mmap| mmap.len())
    }
}

/// Dataset dto used for other layer, such as storage
/// N is the aligned dimension
#[derive(Debug)]
pub struct DatasetDto<'a, T> {
    /// data slice borrow from dataset
    pub data: &'a mut [T],

    /// rounded dimension
    pub rounded_dim: usize,
}

#[cfg(test)]
mod dataset_test {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::{env, fs};

    use super::*;
    use crate::model::vertex::DIM_128;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    #[derive(Default)]
    struct WorkerRecorder {
        workers: StdMutex<HashSet<usize>>,
    }

    impl MedoidThreadObserver for WorkerRecorder {
        fn on_nearest_point_worker(&self, worker: usize) {
            self.workers.lock().unwrap().insert(worker);
        }
    }

    #[test]
    fn get_vertex_within_range() {
        let num_points = 1_000_000;
        let id = 999_999;
        let dataset = InmemDataset::<f32>::new(num_points, 1f32, 128).unwrap();

        let vertex = dataset.get_vertex(999_999).unwrap();

        assert_eq!(vertex.vertex_id(), id);
        assert_eq!(vertex.vector().len(), DIM_128);
        // This assertion only works for in-memory mode
        if !dataset.is_memory_mapped() {
            assert_eq!(vertex.vector().as_ptr(), unsafe {
                dataset.data.as_ptr().add((id as usize) * DIM_128)
            });
        }
    }

    #[test]
    fn get_vertex_out_of_range() {
        let num_points = 1_000_000;
        let invalid_id = 1_000_000;
        let dataset = InmemDataset::<f32>::new(num_points, 1f32, 128).unwrap();

        if dataset.get_vertex(invalid_id).is_ok() {
            panic!("id ({}) should be out of range", invalid_id)
        };
    }

    #[test]
    fn load_data_test() {
        let file_name = "dataset_test_load_data_test.bin";
        // npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let mut dataset = InmemDataset::<f32>::new(2, 1f32, 8).unwrap();

        match copy_aligned_data_from_file(file_name, dataset.into_dto(), 0) {
            Ok((npts, dim)) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                assert!(npts == 2);
                assert!(dim == 8);
                assert!(dataset.data.len() == 16);

                let first_vertex = dataset.get_vertex(0).unwrap();
                let second_vertex = dataset.get_vertex(1).unwrap();

                assert!(*first_vertex.vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
                assert!(*second_vertex.vector() == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
            }
            Err(e) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                panic!("{}", e)
            }
        }
    }

    #[test]
    fn calculate_medoid_point_id_respects_rayon_num_threads_env_cap() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = env::var("RAYON_NUM_THREADS").ok();
        unsafe {
            env::set_var("RAYON_NUM_THREADS", "2");
        }

        let recorder = Arc::new(WorkerRecorder::default());
        let previous = install_medoid_thread_observer(Some(recorder.clone()));
        assert!(
            previous.is_none(),
            "unexpected pre-existing medoid observer"
        );

        let dataset = InmemDataset::<f32>::new(262_144, 1.0, 8).unwrap();
        let _ = dataset.calculate_medoid_point_id().unwrap();

        let _ = install_medoid_thread_observer(None);
        match original {
            Some(value) => unsafe {
                env::set_var("RAYON_NUM_THREADS", value);
            },
            None => unsafe {
                env::remove_var("RAYON_NUM_THREADS");
            },
        }

        let worker_count = recorder.workers.lock().unwrap().len();
        assert!(
            worker_count <= 2,
            "expected medoid search to respect RAYON_NUM_THREADS=2, observed {worker_count} workers"
        );
    }

    #[test]
    fn mmap_only_dataset_does_not_preallocate_full_data_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let file_name = dir.path().join("dataset.fbin");
        let rows = 4u32;
        let dim = 3u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.extend_from_slice(&dim.to_le_bytes());
        let values = [
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ];
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&file_name, bytes).unwrap();

        let mut dataset = InmemDataset::<f32>::new_memory_mapped(4, 3).unwrap();
        assert_eq!(dataset.data.len(), 0);
        assert!(!dataset.is_memory_mapped());

        dataset.set_memory_mapped(&file_name).unwrap();
        dataset.build_from_file(&file_name, 4).unwrap();

        assert!(dataset.is_memory_mapped());
        assert_eq!(dataset.data.len(), 0);
        let vertex = dataset.get_vertex(2).unwrap();
        assert_eq!(*vertex.vector(), [7.0, 8.0, 9.0]);
    }
}
