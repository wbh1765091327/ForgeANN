use std::alloc::LayoutError;
use std::array::TryFromSliceError;
use std::num::TryFromIntError;

/// Result
pub type AnnResult<T> = Result<T, AnnError>;

/// DiskANN Error
/// ANNError is `Send` (i.e., safe to send across threads)
#[derive(thiserror::Error, Debug)]
pub enum AnnError {
    /// Index construction and search error
    #[error("IndexError: {err}")]
    Index { err: String },

    /// Index configuration error
    #[error("IndexConfigError: {parameter} is invalid, err={err}")]
    IndexConfig { parameter: String, err: String },

    /// Integer conversion error
    #[error("TryFromIntError: {err}")]
    TryFromInt {
        #[from]
        err: TryFromIntError,
    },

    /// IO error
    #[error("IOError: {err}")]
    Io {
        #[from]
        err: std::io::Error,
    },

    /// Layout error in memory allocation
    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayout {
        #[from]
        err: LayoutError,
    },

    /// PoisonError which can be returned whenever a lock is acquired
    /// Both Mutexes and RwLocks are poisoned whenever a thread fails while the lock is held
    #[error("LockPoisonError: {err}")]
    LockPoison { err: String },

    /// DiskIOAlignmentError which can be returned when calling windows API CreateFileA for the disk
    /// index file fails.
    #[error("DiskIOAlignmentError: {err}")]
    DiskIoAlignmentError { err: String },

    /// IOQueueError which can be returned when we call windows API CreateIoQueue for the disk index
    /// file fails.
    #[error("IOQueueError: {err}")]
    IoQueue { err: String },

    /// Array conversion error
    #[error("Error try creating array from slice: {err}")]
    TryFromSlice {
        #[from]
        err: TryFromSliceError,
    },
}

impl AnnError {
    pub fn log_index_config_error(parameter: String, err: String) -> Self {
        AnnError::IndexConfig { parameter, err }
    }

    pub fn log_index_error(err: String) -> Self {
        AnnError::Index { err }
    }

    pub fn log_lock_poison_error(err: String) -> Self {
        AnnError::LockPoison { err }
    }

    pub fn log_io_error(err: std::io::Error) -> Self {
        AnnError::Io { err }
    }

    pub fn log_disk_io_alignment_error(err: String) -> Self {
        AnnError::DiskIoAlignmentError { err }
    }

    pub fn log_io_queue_error(err: String) -> Self {
        AnnError::IoQueue { err }
    }
}
