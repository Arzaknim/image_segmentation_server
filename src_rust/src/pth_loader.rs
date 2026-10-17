use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

const ZIP_LOCAL_FILE_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const ZIP_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0201_4b50;
const ZIP_END_OF_CENTRAL_DIR_SIGNATURE: u32 = 0x0605_4b50;
const ZIP_STORED_METHOD: u16 = 0;

#[derive(Debug, Clone)]
pub enum PthError {
    Io(String),
    InvalidArchive(String),
    InvalidPickle(String),
    MissingTensor(String),
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

impl fmt::Display for PthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::InvalidArchive(msg) => write!(f, "Invalid .pth archive: {msg}"),
            Self::InvalidPickle(msg) => write!(f, "Invalid pickle stream in .pth: {msg}"),
            Self::MissingTensor(name) => write!(f, "Missing tensor in state_dict: {name}"),
            Self::ShapeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "Shape mismatch for tensor '{tensor_name}': expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for PthError {}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct PthStateDict {
    tensors: HashMap<String, TensorData>,
}

impl PthStateDict {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, PthError> {
        let bytes = fs::read(path.as_ref())
            .map_err(|err| PthError::Io(format!("{}: {err}", path.as_ref().display())))?;
        Self::load_from_bytes(&bytes)
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, PthError> {
        let archive_entries = parse_stored_zip_entries(bytes)?;
        let pickle_bytes = find_data_pkl(&archive_entries)?;
        let descriptors = parse_tensor_descriptors(pickle_bytes)?;

        let mut tensors = HashMap::with_capacity(descriptors.len());
        for (tensor_name, descriptor) in descriptors {
            if descriptor.storage_type != "FloatStorage" {
                continue;
            }
            let raw_bytes = find_storage_bytes(&archive_entries, &descriptor.storage_key)?;
            let values = decode_f32_slice(raw_bytes, descriptor.offset, descriptor.element_count)?;
            tensors.insert(
                tensor_name,
                TensorData {
                    shape: descriptor.shape,
                    values,
                },
            );
        }

        Ok(Self { tensors })
    }

    pub fn get_tensor(&self, name: &str, expected_shape: &[usize]) -> Result<&[f32], PthError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| PthError::MissingTensor(name.to_string()))?;

        if tensor.shape != expected_shape {
            return Err(PthError::ShapeMismatch {
                tensor_name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: tensor.shape.clone(),
            });
        }

        Ok(&tensor.values)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Debug, Clone)]
struct TensorDescriptor {
    storage_type: String,
    storage_key: String,
    offset: usize,
    shape: Vec<usize>,
    element_count: usize,
}

fn find_data_pkl<'a>(entries: &'a HashMap<String, &'a [u8]>) -> Result<&'a [u8], PthError> {
    entries
        .iter()
        .find_map(|(name, data)| {
            if name.ends_with("/data.pkl") || name == "data.pkl" {
                Some(*data)
            } else {
                None
            }
        })
        .ok_or_else(|| PthError::InvalidArchive("Archive does not contain data.pkl".to_string()))
}

fn find_storage_bytes<'a>(
    entries: &'a HashMap<String, &'a [u8]>,
    storage_key: &str,
) -> Result<&'a [u8], PthError> {
    let suffix = format!("/data/{storage_key}");
    entries
        .iter()
        .find_map(|(name, data)| {
            if name.ends_with(&suffix) {
                Some(*data)
            } else {
                None
            }
        })
        .ok_or_else(|| {
            PthError::InvalidArchive(format!("Missing tensor data file for key '{storage_key}'"))
        })
}

fn decode_f32_slice(
    raw_bytes: &[u8],
    offset: usize,
    element_count: usize,
) -> Result<Vec<f32>, PthError> {
    let byte_offset = offset.checked_mul(4).ok_or_else(|| {
        PthError::InvalidArchive("Storage byte offset overflow".to_string())
    })?;
    let byte_len = element_count.checked_mul(4).ok_or_else(|| {
        PthError::InvalidArchive("Storage byte length overflow".to_string())
    })?;
    let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
        PthError::InvalidArchive("Storage slice range overflow".to_string())
    })?;

    if end > raw_bytes.len() {
        return Err(PthError::InvalidArchive(format!(
            "Storage buffer too small: need {end} bytes, got {}",
            raw_bytes.len()
        )));
    }

    let values = raw_bytes[byte_offset..end]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(values)
}

fn parse_stored_zip_entries(bytes: &[u8]) -> Result<HashMap<String, &[u8]>, PthError> {
    let eocd_offset = find_eocd_offset(bytes)?;
    let total_entries = read_u16_le(bytes, eocd_offset + 10)? as usize;
    let cd_offset = read_u32_le(bytes, eocd_offset + 16)? as usize;

    let mut entries = HashMap::with_capacity(total_entries);
    let mut cursor = cd_offset;

    for _ in 0..total_entries {
        let signature = read_u32_le(bytes, cursor)?;
        if signature != ZIP_CENTRAL_DIRECTORY_SIGNATURE {
            return Err(PthError::InvalidArchive(
                "Corrupted ZIP central directory header".to_string(),
            ));
        }

        let compression_method = read_u16_le(bytes, cursor + 10)?;
        if compression_method != ZIP_STORED_METHOD {
            return Err(PthError::InvalidArchive(format!(
                "Unsupported ZIP compression method {compression_method} (expected uncompressed store)"
            )));
        }

        let uncompressed_size = read_u32_le(bytes, cursor + 24)? as usize;
        let file_name_len = read_u16_le(bytes, cursor + 28)? as usize;
        let extra_len = read_u16_le(bytes, cursor + 30)? as usize;
        let comment_len = read_u16_le(bytes, cursor + 32)? as usize;
        let local_header_offset = read_u32_le(bytes, cursor + 42)? as usize;

        let name_start = cursor + 46;
        let name_end = name_start + file_name_len;
        let file_name = std::str::from_utf8(
            bytes
                .get(name_start..name_end)
                .ok_or_else(|| PthError::InvalidArchive("Truncated entry filename".to_string()))?,
        )
        .map_err(|err| PthError::InvalidArchive(err.to_string()))?
        .to_string();

        let data_slice = extract_local_entry_slice(bytes, local_header_offset, uncompressed_size)?;
        entries.insert(file_name, data_slice);

        cursor = name_end + extra_len + comment_len;
    }

    Ok(entries)
}

fn find_eocd_offset(bytes: &[u8]) -> Result<usize, PthError> {
    if bytes.len() < 22 {
        return Err(PthError::InvalidArchive("File too small to be a ZIP archive".to_string()));
    }
    let min_offset = bytes.len().saturating_sub(65_557);
    for offset in (min_offset..=bytes.len() - 22).rev() {
        if read_u32_le(bytes, offset)? == ZIP_END_OF_CENTRAL_DIR_SIGNATURE {
            return Ok(offset);
        }
    }
    Err(PthError::InvalidArchive(
        "ZIP End of Central Directory signature not found".to_string(),
    ))
}

fn extract_local_entry_slice(
    bytes: &[u8],
    local_header_offset: usize,
    uncompressed_size: usize,
) -> Result<&[u8], PthError> {
    if read_u32_le(bytes, local_header_offset)? != ZIP_LOCAL_FILE_HEADER_SIGNATURE {
        return Err(PthError::InvalidArchive(
            "Invalid local file header signature".to_string(),
        ));
    }
    let name_len = read_u16_le(bytes, local_header_offset + 26)? as usize;
    let extra_len = read_u16_le(bytes, local_header_offset + 28)? as usize;
    let data_start = local_header_offset + 30 + name_len + extra_len;
    let data_end = data_start + uncompressed_size;

    bytes
        .get(data_start..data_end)
        .ok_or_else(|| PthError::InvalidArchive("Truncated local entry data".to_string()))
}

#[derive(Debug, Clone)]
enum PickleValue {
    Mark,
    NoneVal,
    Bool(bool),
    Int(usize),
    Str(String),
    Global(String),
    Tuple(Vec<PickleValue>),
    PersId(Box<PickleValue>),
    Tensor(TensorDescriptor),
    Dict(Vec<(String, TensorDescriptor)>),
    Other,
}

fn parse_tensor_descriptors(buf: &[u8]) -> Result<HashMap<String, TensorDescriptor>, PthError> {
    let mut pos = 0;
    let mut stack: Vec<PickleValue> = Vec::new();
    let mut memo: HashMap<u32, PickleValue> = HashMap::new();
    let mut descriptors = HashMap::new();

    while pos < buf.len() {
        let opcode = buf[pos];
        pos += 1;

        match opcode {
            0x80 => {
                pos += 1; // PROTO
            }
            b'c' => {
                let module = read_line_utf8(buf, &mut pos)?;
                let name = read_line_utf8(buf, &mut pos)?;
                stack.push(PickleValue::Global(format!("{module}.{name}")));
            }
            b'q' => {
                let key = read_u8(buf, &mut pos)? as u32;
                if let Some(top) = stack.last().cloned() {
                    memo.insert(key, top);
                }
            }
            b'r' => {
                let key = read_u32_le(buf, pos)?;
                pos += 4;
                if let Some(top) = stack.last().cloned() {
                    memo.insert(key, top);
                }
            }
            b'h' => {
                let key = read_u8(buf, &mut pos)? as u32;
                let val = memo.get(&key).cloned().ok_or_else(|| {
                    PthError::InvalidPickle(format!("Missing memo key {key}"))
                })?;
                stack.push(val);
            }
            b'j' => {
                let key = read_u32_le(buf, pos)?;
                pos += 4;
                let val = memo.get(&key).cloned().ok_or_else(|| {
                    PthError::InvalidPickle(format!("Missing memo key {key}"))
                })?;
                stack.push(val);
            }
            b')' => stack.push(PickleValue::Tuple(Vec::new())),
            b'}' => stack.push(PickleValue::Dict(Vec::new())),
            b'(' => stack.push(PickleValue::Mark),
            b'N' => stack.push(PickleValue::NoneVal),
            0x88 => stack.push(PickleValue::Bool(true)),
            0x89 => stack.push(PickleValue::Bool(false)),
            b'K' => {
                let val = read_u8(buf, &mut pos)? as usize;
                stack.push(PickleValue::Int(val));
            }
            b'M' => {
                let val = read_u16_le(buf, pos)? as usize;
                pos += 2;
                stack.push(PickleValue::Int(val));
            }
            b'J' => {
                let val = read_i32_le(buf, pos)? as usize;
                pos += 4;
                stack.push(PickleValue::Int(val));
            }
            b'X' => {
                let len = read_u32_le(buf, pos)? as usize;
                pos += 4;
                let slice = buf.get(pos..pos + len).ok_or_else(|| {
                    PthError::InvalidPickle("Truncated BINUNICODE string".to_string())
                })?;
                let s = std::str::from_utf8(slice)
                    .map_err(|err| PthError::InvalidPickle(err.to_string()))?
                    .to_string();
                pos += len;
                stack.push(PickleValue::Str(s));
            }
            b't' => {
                let items = pop_until_mark(&mut stack)?;
                stack.push(PickleValue::Tuple(items));
            }
            0x85 => {
                let a = pop_stack(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop_stack(&mut stack)?;
                let a = pop_stack(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop_stack(&mut stack)?;
                let b = pop_stack(&mut stack)?;
                let a = pop_stack(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }
            b'Q' => {
                let pid = pop_stack(&mut stack)?;
                stack.push(PickleValue::PersId(Box::new(pid)));
            }
            b'R' => {
                let args = pop_stack(&mut stack)?;
                let func = pop_stack(&mut stack)?;
                let reduced = evaluate_reduce(func, args)?;
                stack.push(reduced);
            }
            b's' => {
                let val = pop_stack(&mut stack)?;
                let key = pop_stack(&mut stack)?;
                if let (PickleValue::Str(k), PickleValue::Tensor(t)) = (key, val) {
                    descriptors.insert(k, t);
                }
            }
            b'u' => {
                let items = pop_until_mark(&mut stack)?;
                for pair in items.chunks_exact(2) {
                    if let (PickleValue::Str(k), PickleValue::Tensor(t)) = (&pair[0], &pair[1]) {
                        descriptors.insert(k.clone(), t.clone());
                    }
                }
            }
            b'b' => {
                let _state = pop_stack(&mut stack)?;
            }
            b'.' => break,
            other => {
                return Err(PthError::InvalidPickle(format!(
                    "Unsupported pickle opcode 0x{other:02x}"
                )));
            }
        }
    }

    Ok(descriptors)
}

fn evaluate_reduce(func: PickleValue, args: PickleValue) -> Result<PickleValue, PthError> {
    if let PickleValue::Global(name) = func {
        if name == "collections.OrderedDict" {
            return Ok(PickleValue::Dict(Vec::new()));
        }
        if name == "torch._utils._rebuild_tensor_v2" {
            return build_tensor_descriptor(args);
        }
    }
    Ok(PickleValue::Other)
}

fn build_tensor_descriptor(args: PickleValue) -> Result<PickleValue, PthError> {
    let PickleValue::Tuple(items) = args else {
        return Err(PthError::InvalidPickle(
            "Expected tuple args for _rebuild_tensor_v2".to_string(),
        ));
    };
    if items.len() < 3 {
        return Err(PthError::InvalidPickle(
            "Insufficient args for _rebuild_tensor_v2".to_string(),
        ));
    }

    let PickleValue::PersId(persid) = &items[0] else {
        return Err(PthError::InvalidPickle("Expected PersId in tensor".to_string()));
    };
    let PickleValue::Tuple(storage_fields) = persid.as_ref() else {
        return Err(PthError::InvalidPickle("Expected storage tuple".to_string()));
    };
    if storage_fields.len() < 5 {
        return Err(PthError::InvalidPickle("Short storage tuple".to_string()));
    }

    let storage_type = match &storage_fields[1] {
        PickleValue::Global(g) => g.rsplit('.').next().unwrap_or(g).to_string(),
        _ => return Err(PthError::InvalidPickle("Invalid storage type".to_string())),
    };
    let storage_key = match &storage_fields[2] {
        PickleValue::Str(s) => s.clone(),
        _ => return Err(PthError::InvalidPickle("Invalid storage key".to_string())),
    };
    let element_count = match &storage_fields[4] {
        PickleValue::Int(n) => *n,
        _ => return Err(PthError::InvalidPickle("Invalid storage size".to_string())),
    };
    let offset = match &items[1] {
        PickleValue::Int(n) => *n,
        _ => return Err(PthError::InvalidPickle("Invalid tensor offset".to_string())),
    };
    let shape = match &items[2] {
        PickleValue::Tuple(dims) => dims
            .iter()
            .filter_map(|d| match d {
                PickleValue::Int(v) => Some(*v),
                _ => None,
            })
            .collect(),
        _ => return Err(PthError::InvalidPickle("Invalid tensor shape".to_string())),
    };

    Ok(PickleValue::Tensor(TensorDescriptor {
        storage_type,
        storage_key,
        offset,
        shape,
        element_count,
    }))
}

fn pop_stack(stack: &mut Vec<PickleValue>) -> Result<PickleValue, PthError> {
    stack
        .pop()
        .ok_or_else(|| PthError::InvalidPickle("Pickle stack underflow".to_string()))
}

fn pop_until_mark(stack: &mut Vec<PickleValue>) -> Result<Vec<PickleValue>, PthError> {
    let mut items = Vec::new();
    while let Some(val) = stack.pop() {
        if matches!(val, PickleValue::Mark) {
            items.reverse();
            return Ok(items);
        }
        items.push(val);
    }
    Err(PthError::InvalidPickle(
        "Pickle MARK not found on stack".to_string(),
    ))
}

fn read_line_utf8(buf: &[u8], pos: &mut usize) -> Result<String, PthError> {
    let slice = buf
        .get(*pos..)
        .ok_or_else(|| PthError::InvalidPickle("Unexpected EOF reading line".to_string()))?;
    let newline_rel = slice
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| PthError::InvalidPickle("Missing newline in GLOBAL opcode".to_string()))?;
    let line = std::str::from_utf8(&slice[..newline_rel])
        .map_err(|err| PthError::InvalidPickle(err.to_string()))?
        .to_string();
    *pos += newline_rel + 1;
    Ok(line)
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8, PthError> {
    let b = *buf
        .get(*pos)
        .ok_or_else(|| PthError::InvalidPickle("Unexpected EOF".to_string()))?;
    *pos += 1;
    Ok(b)
}

fn read_u16_le(buf: &[u8], offset: usize) -> Result<u16, PthError> {
    let slice = buf
        .get(offset..offset + 2)
        .ok_or_else(|| PthError::InvalidArchive("Unexpected EOF reading u16".to_string()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(buf: &[u8], offset: usize) -> Result<u32, PthError> {
    let slice = buf
        .get(offset..offset + 4)
        .ok_or_else(|| PthError::InvalidArchive("Unexpected EOF reading u32".to_string()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_i32_le(buf: &[u8], offset: usize) -> Result<i32, PthError> {
    let slice = buf
        .get(offset..offset + 4)
        .ok_or_else(|| PthError::InvalidArchive("Unexpected EOF reading i32".to_string()))?;
    Ok(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

