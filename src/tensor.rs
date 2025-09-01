use std::cmp::PartialEq;
use anyhow::anyhow;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float = 0,
    Half = 1,
    Int8 = 2,
    Int32 = 3,
    UINT8 = 5,
    FP8 = 6,
    BF16 = 7,
    INT64 = 8,
    INT4 = 9,
    FP4 = 10,
    E8M0 = 11,
}

pub fn data_type_from_u8(value: u8) -> Option<DataType> {
    match value {
        0 => Some(DataType::Float),
        1 => Some(DataType::Half),
        2 => Some(DataType::Int8),
        3 => Some(DataType::Int32),
        5 => Some(DataType::UINT8),
        6 => Some(DataType::FP8),
        7 => Some(DataType::BF16),
        8 => Some(DataType::INT64),
        9 => Some(DataType::INT4),
        10 => Some(DataType::FP4),
        11 => Some(DataType::E8M0),
        _ => None,
    }
}

pub fn get_element_size(dtype: &DataType) -> usize {
    match dtype {
        DataType::Float => 4,
        DataType::Half => 2,
        DataType::Int8 => 1,
        DataType::Int32 => 4,
        DataType::UINT8 => 1,
        DataType::FP8 => 1,
        DataType::BF16 => 2,
        DataType::INT64 => 8,
        DataType::INT4 => 1, // Assuming packed
        DataType::FP4 => 1, // Assuming packed
        DataType::E8M0 => 1, // Assuming packed
    }
}

pub struct Tensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

impl Tensor {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DataType) -> anyhow::Result<Self> {
        let element_size = get_element_size(&dtype);
        match {data.len() == shape.iter().product() * element_size} {
            true => {Ok(Self { data, shape, dtype })},
            false => {Err(anyhow!("Data length does not match shape and element size"))},
        }
    }

    pub fn element_size(&self) -> usize {
        get_element_size(&self.dtype)
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.element_size()
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.data.chunks_exact(self.element_size()).map(|chunk| {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            f32::from_le_bytes(arr)
        })
    }

    pub fn batch_size(&self) -> usize {
        self.shape[0]
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> anyhow::Result<Tensor> {
        if self.shape.is_empty() {
            return Err(anyhow!("Tensor has no dimensions"));
        }
        let first_dim = self.shape[0];
        if range.start >= first_dim || range.end > first_dim || range.start >= range.end {
            return Err(anyhow!("Invalid slice range"));
        }
        let slice_size = range.end - range.start;
        let element_count_per_slice: usize = self.shape.iter().skip(1).product();
        let byte_start = range.start * element_count_per_slice * self.element_size();
        let byte_end = range.end * element_count_per_slice * self.element_size();
        let sliced_data = self.data[byte_start..byte_end].to_vec();
        let mut new_shape = self.shape.clone();
        new_shape[0] = slice_size;
        Ok(Tensor {
            data: sliced_data,
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn merge(tensors: &[Tensor]) -> anyhow::Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow!("No tensors to merge"));
        }
        let first_shape = &tensors[0].shape;
        let dtype = tensors[0].dtype.clone();
        let element_size = tensors[0].element_size();
        for tensor in tensors {
            if tensor.shape.len() != first_shape.len() {
                return Err(anyhow!("Tensors have different number of dimensions"));
            }
            if tensor.dtype != dtype {
                return Err(anyhow!("Tensors have different data types"));
            }
            for (i, &dim) in tensor.shape.iter().enumerate().skip(1) {
                if dim != first_shape[i] {
                    return Err(anyhow!("Tensors have incompatible shapes"));
                }
            }
        }
        let mut merged_data = Vec::new();
        for tensor in tensors {
            merged_data.extend_from_slice(&tensor.data);
        }
        let mut new_shape = first_shape.clone();
        new_shape[0] = tensors.iter().map(|t| t.shape[0]).sum();
        Ok(Tensor {
            data: merged_data,
            shape: new_shape,
            dtype,
        })
    }
}
