use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::common::{AnnError, AnnResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedMemGraph {
    pub index_size: u64,
    pub max_degree: u32,
    pub start: u32,
    pub num_frozen_pts: u64,
    pub neighbors: Vec<Vec<u32>>,
}

#[derive(Debug)]
pub(crate) struct DirectMemGraphWriter {
    writer: BufWriter<File>,
    index_size: u64,
    max_degree: u32,
    start: u32,
    num_frozen_pts: u64,
}

#[derive(Debug)]
pub(crate) struct FixedDegreeMemGraphWriter {
    file: File,
    num_points: usize,
    max_degree: usize,
    record_bytes: usize,
}

impl FixedDegreeMemGraphWriter {
    pub(crate) fn create(path: &Path, num_points: usize, max_degree: usize) -> AnnResult<Self> {
        let file = File::create(path)?;
        let record_u32s = max_degree
            .checked_add(1)
            .ok_or_else(|| AnnError::log_index_error("fixed graph record overflow".to_string()))?;
        let record_bytes = record_u32s
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| AnnError::log_index_error("fixed graph record overflow".to_string()))?;
        let total_bytes = (num_points as u64)
            .checked_mul(record_bytes as u64)
            .ok_or_else(|| AnnError::log_index_error("fixed graph file overflow".to_string()))?;
        file.set_len(total_bytes)?;
        Ok(Self {
            file,
            num_points,
            max_degree,
            record_bytes,
        })
    }

    pub(crate) fn write_node(&self, uid: u32, neighbors: &[u32]) -> AnnResult<()> {
        let uid = uid as usize;
        if uid >= self.num_points {
            return Err(AnnError::log_index_error(format!(
                "fixed graph node id {uid} is out of range {}",
                self.num_points
            )));
        }
        let degree = neighbors.len().min(self.max_degree);
        let mut record = vec![0u8; self.record_bytes];
        record[..4].copy_from_slice(&(degree as u32).to_le_bytes());
        for (slot, neighbor) in neighbors.iter().take(degree).enumerate() {
            let offset = (slot + 1) * std::mem::size_of::<u32>();
            record[offset..offset + 4].copy_from_slice(&neighbor.to_le_bytes());
        }
        write_all_at(
            &self.file,
            &record,
            (uid as u64).saturating_mul(self.record_bytes as u64),
        )
    }

    pub(crate) fn finish(self) -> AnnResult<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

impl DirectMemGraphWriter {
    pub(crate) fn create(path: &Path, start: u32, num_frozen_pts: usize) -> AnnResult<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&24u64.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&start.to_le_bytes())?;
        writer.write_all(&(num_frozen_pts as u64).to_le_bytes())?;

        Ok(Self {
            writer,
            index_size: 24,
            max_degree: 0,
            start,
            num_frozen_pts: num_frozen_pts as u64,
        })
    }

    pub(crate) fn write_node(&mut self, _uid: u32, neighbors: &[u32]) -> AnnResult<()> {
        let degree = neighbors.len() as u32;
        self.writer.write_all(&degree.to_le_bytes())?;
        for neighbor in neighbors {
            self.writer.write_all(&neighbor.to_le_bytes())?;
        }
        self.index_size += ((neighbors.len() + 1) * std::mem::size_of::<u32>()) as u64;
        self.max_degree = self.max_degree.max(degree);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> AnnResult<()> {
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&self.index_size.to_le_bytes())?;
        self.writer.write_all(&self.max_degree.to_le_bytes())?;
        self.writer.write_all(&self.start.to_le_bytes())?;
        self.writer.write_all(&self.num_frozen_pts.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

pub(crate) fn load_fixed_degree_mem_graph(
    path: &Path,
    expected_num_points: usize,
    max_degree: usize,
) -> AnnResult<Vec<Vec<u32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let record_u32s = max_degree
        .checked_add(1)
        .ok_or_else(|| AnnError::log_index_error("fixed graph record overflow".to_string()))?;
    let record_bytes = record_u32s
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| AnnError::log_index_error("fixed graph record overflow".to_string()))?;
    let mut record = vec![0u8; record_bytes];
    let mut rows = Vec::with_capacity(expected_num_points);
    for vid in 0..expected_num_points {
        reader.read_exact(&mut record)?;
        let degree = u32::from_le_bytes(record[..4].try_into().unwrap()) as usize;
        if degree > max_degree {
            return Err(AnnError::log_index_error(format!(
                "fixed graph node {vid} degree {degree} exceeds max degree {max_degree}"
            )));
        }
        let mut row = Vec::with_capacity(degree);
        for slot in 0..degree {
            let offset = (slot + 1) * std::mem::size_of::<u32>();
            row.push(u32::from_le_bytes(
                record[offset..offset + 4].try_into().unwrap(),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

pub fn load_mem_graph(path: &Path, expected_num_points: usize) -> AnnResult<LoadedMemGraph> {
    let mut reader = BufReader::new(File::open(path)?);
    let index_size = read_u64(&mut reader)?;
    let max_degree = read_u32(&mut reader)?;
    let start = read_u32(&mut reader)?;
    let num_frozen_pts = read_u64(&mut reader)?;
    let payload_bytes = index_size
        .checked_sub(24)
        .ok_or_else(|| AnnError::log_index_error("invalid _mem.index header".to_string()))?
        as usize;
    if !payload_bytes.is_multiple_of(std::mem::size_of::<u32>()) {
        return Err(AnnError::log_index_error(format!(
            "graph payload size {} is not u32-aligned",
            payload_bytes
        )));
    }

    let mut bytes_read = 0usize;
    let mut neighbors = Vec::with_capacity(expected_num_points);
    while bytes_read < payload_bytes {
        if neighbors.len() >= expected_num_points {
            return Err(AnnError::log_index_error(format!(
                "graph has more than {expected_num_points} nodes"
            )));
        }
        let degree = read_u32(&mut reader)? as usize;
        bytes_read += std::mem::size_of::<u32>();
        if bytes_read + degree * std::mem::size_of::<u32>() > payload_bytes {
            return Err(AnnError::log_index_error(
                "graph record exceeds payload size".to_string(),
            ));
        }
        let mut row = Vec::with_capacity(degree);
        for _ in 0..degree {
            row.push(read_u32(&mut reader)?);
        }
        bytes_read += degree * std::mem::size_of::<u32>();
        neighbors.push(row);
    }

    if neighbors.len() != expected_num_points {
        return Err(AnnError::log_index_error(format!(
            "graph has {} nodes, expected {}",
            neighbors.len(),
            expected_num_points
        )));
    }

    Ok(LoadedMemGraph {
        index_size,
        max_degree,
        start,
        num_frozen_pts,
        neighbors,
    })
}

#[cfg(unix)]
fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> AnnResult<()> {
    while !buf.is_empty() {
        let written = file.write_at(buf, offset)?;
        if written == 0 {
            return Err(AnnError::log_index_error(
                "failed to write fixed graph record".to_string(),
            ));
        }
        offset += written as u64;
        buf = &buf[written..];
    }
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> AnnResult<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> AnnResult<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    #[test]
    fn direct_mem_graph_writer_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "forgeann-oom-writer-{}-{}.index",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut writer = super::DirectMemGraphWriter::create(&path, 4, 0).unwrap();
        writer.write_node(0, &[3, 4]).unwrap();
        writer.write_node(1, &[4]).unwrap();
        writer.write_node(2, &[1, 0, 4]).unwrap();
        writer.write_node(3, &[0]).unwrap();
        writer.write_node(4, &[2, 1]).unwrap();
        writer.finish().unwrap();

        let loaded = super::load_mem_graph(&path, 5).unwrap();

        assert_eq!(loaded.start, 4);
        assert_eq!(loaded.neighbors[0], &[3, 4]);
        assert_eq!(loaded.neighbors[1], &[4]);
        assert_eq!(loaded.neighbors[2], &[1, 0, 4]);
        assert_eq!(loaded.neighbors[3], &[0]);
        assert_eq!(loaded.neighbors[4], &[2, 1]);

        std::fs::remove_file(path).unwrap();
    }
}
