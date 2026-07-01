#[cfg(test)]
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use libc::O_DIRECT;

use crate::common::{AlignedBoxWithSlice, AnnError, AnnResult};

const DEFAULT_DIRECT_IO_ALIGNMENT: usize = 4096;

#[cfg(test)]
thread_local! {
    static DIRECT_WRITE_CALLS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_SYNC_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_direct_io_test_counters() {
    DIRECT_WRITE_CALLS.set(0);
    DIRECT_SYNC_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn direct_io_test_write_calls() -> usize {
    DIRECT_WRITE_CALLS.get()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectIoConfig {
    pub enabled: bool,
    pub alignment: usize,
}

impl Default for DirectIoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alignment: DEFAULT_DIRECT_IO_ALIGNMENT,
        }
    }
}

impl DirectIoConfig {
    #[inline]
    pub fn disabled() -> Self {
        Self::default()
    }

    #[inline]
    pub fn enabled_with_alignment(alignment: usize) -> Self {
        Self {
            enabled: true,
            alignment: alignment
                .max(DEFAULT_DIRECT_IO_ALIGNMENT)
                .next_power_of_two(),
        }
    }

    #[inline]
    pub fn effective_alignment(self) -> usize {
        self.alignment
            .max(DEFAULT_DIRECT_IO_ALIGNMENT)
            .next_power_of_two()
    }
}

#[derive(Debug)]
pub struct DirectIoFile {
    path: PathBuf,
    file: File,
    cfg: DirectIoConfig,
}

impl DirectIoFile {
    pub fn open_rw(path: &Path, cfg: DirectIoConfig, truncate: bool) -> AnnResult<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        if truncate {
            options.truncate(true);
        }
        if cfg.enabled {
            options.custom_flags(O_DIRECT);
        }
        let file = options.open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            cfg,
        })
    }

    pub fn open_read(path: &Path, cfg: DirectIoConfig) -> AnnResult<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        if cfg.enabled {
            options.custom_flags(O_DIRECT);
        }
        let file = options.open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            cfg,
        })
    }

    pub fn create_write(path: &Path, cfg: DirectIoConfig) -> AnnResult<Self> {
        Self::open_rw(path, cfg, true)
    }

    #[inline]
    pub fn config(&self) -> DirectIoConfig {
        self.cfg
    }

    #[inline]
    pub(crate) fn raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    fn alignment(&self) -> usize {
        self.cfg.effective_alignment()
    }

    pub fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> AnnResult<()> {
        if !self.cfg.enabled {
            read_exact_at_buffered(&self.file, &self.path, buf, offset)?;
            return Ok(());
        }

        let alignment = self.alignment() as u64;
        let aligned_offset = offset / alignment * alignment;
        let prefix = (offset - aligned_offset) as usize;
        let aligned_len = round_up(prefix.saturating_add(buf.len()), self.alignment());
        let mut scratch = AlignedBoxWithSlice::<u8>::new(aligned_len, self.alignment())?;
        let scratch_slice = scratch.as_mut_slice();
        read_exact_at_buffered_partial_ok(&self.file, scratch_slice, aligned_offset)?;
        let end = prefix + buf.len();
        buf.copy_from_slice(&scratch_slice[prefix..end]);
        Ok(())
    }

    pub fn write_all_at(&self, buf: &[u8], offset: u64) -> AnnResult<()> {
        if !self.cfg.enabled {
            write_all_at_buffered(&self.file, &self.path, buf, offset)?;
            return Ok(());
        }

        let alignment = self.alignment() as u64;
        let aligned_offset = offset / alignment * alignment;
        let prefix = (offset - aligned_offset) as usize;
        let aligned_len = round_up(prefix.saturating_add(buf.len()), self.alignment());
        let mut scratch = AlignedBoxWithSlice::<u8>::new(aligned_len, self.alignment())?;
        let scratch_slice = scratch.as_mut_slice();

        if prefix != 0 || aligned_len != buf.len() {
            read_exact_at_buffered_partial_ok(&self.file, scratch_slice, aligned_offset)?;
        }

        let end = prefix + buf.len();
        scratch_slice[prefix..end].copy_from_slice(buf);
        #[cfg(test)]
        DIRECT_WRITE_CALLS.set(DIRECT_WRITE_CALLS.get() + 1);
        write_all_at_buffered(&self.file, &self.path, scratch_slice, aligned_offset)?;
        Ok(())
    }

    pub fn sync_all(&self) -> AnnResult<()> {
        #[cfg(test)]
        if self.cfg.enabled {
            DIRECT_SYNC_CALLS.set(DIRECT_SYNC_CALLS.get() + 1);
        }
        self.file.sync_all()?;
        Ok(())
    }

    pub fn set_len(&self, len: u64) -> AnnResult<()> {
        self.file.set_len(len)?;
        Ok(())
    }
}

fn round_up(value: usize, alignment: usize) -> usize {
    if value == 0 {
        return 0;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn read_exact_at_buffered(
    file: &File,
    path: &Path,
    mut buf: &mut [u8],
    mut offset: u64,
) -> AnnResult<()> {
    while !buf.is_empty() {
        let read = file.read_at(buf, offset)?;
        if read == 0 {
            return Err(AnnError::log_index_error(format!(
                "Unexpected EOF while reading {} at offset {}",
                path.display(),
                offset
            )));
        }
        let (_, rest) = buf.split_at_mut(read);
        buf = rest;
        offset += read as u64;
    }
    Ok(())
}

fn read_exact_at_buffered_partial_ok(
    file: &File,
    buf: &mut [u8],
    mut offset: u64,
) -> AnnResult<()> {
    let mut filled = 0usize;
    while filled < buf.len() {
        let read = file.read_at(&mut buf[filled..], offset)?;
        if read == 0 {
            break;
        }
        filled += read;
        offset += read as u64;
    }
    if filled < buf.len() {
        buf[filled..].fill(0);
    }
    Ok(())
}

fn write_all_at_buffered(
    file: &File,
    path: &Path,
    mut buf: &[u8],
    mut offset: u64,
) -> AnnResult<()> {
    while !buf.is_empty() {
        let written = file.write_at(buf, offset)?;
        if written == 0 {
            return Err(AnnError::log_index_error(format!(
                "Short write while writing {} at offset {}",
                path.display(),
                offset
            )));
        }
        buf = &buf[written..];
        offset += written as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{DirectIoConfig, DirectIoFile};

    #[test]
    fn direct_io_file_round_trips_unaligned_subrange_without_mmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        let cfg = DirectIoConfig::enabled_with_alignment(4096);
        let file = DirectIoFile::create_write(&path, cfg).unwrap();

        let payload: Vec<u8> = (0..6000).map(|i| (i % 251) as u8).collect();
        file.write_all_at(&payload, 8).unwrap();
        file.sync_all().unwrap();

        let reader = DirectIoFile::open_read(&path, cfg).unwrap();
        let mut out = vec![0u8; payload.len()];
        reader.read_exact_at(&mut out, 8).unwrap();
        assert_eq!(out, payload);
    }
}
