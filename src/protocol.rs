use crate::tensor::{data_type_from_u8, Tensor};
use anyhow::anyhow;
use byteorder::{ByteOrder, LittleEndian};
use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

// 协议常量

pub(crate) struct InferenceProtocol {}

impl InferenceProtocol {
    pub(crate) const MAGIC_HEADER: u32 = 0x59722116;
    pub(crate) const ERROR_MAGIC: u32 = 0x87654321;

    pub(crate) async fn read_arrays(stream: &mut UnixStream) -> anyhow::Result<HashMap<String, Tensor>> {
        // 读取数组数量
        let mut count_buf = [0u8; 4];
        stream.read_exact(&mut count_buf).await?;
        let array_count = LittleEndian::read_u32(&count_buf) as usize;

        let mut arrays = HashMap::new();

        for _ in 0..array_count {
            // 读取数组名长度
            let mut name_len_buf = [0u8; 4];
            stream.read_exact(&mut name_len_buf).await?;
            let name_len = LittleEndian::read_u32(&name_len_buf) as usize;

            // 读取数组名
            let mut name_buf = vec![0u8; name_len];
            stream.read_exact(&mut name_buf).await?;
            let name = String::from_utf8(name_buf)?;

            // 读取数据类型
            let mut dtype_buf = [0u8; 1];
            stream.read_exact(&mut dtype_buf).await?;
            let dtype = data_type_from_u8(dtype_buf[0]);
            if dtype == None {
                return Err(anyhow!("Unsupported dtype, found dtype value={}", dtype_buf[0]));
            }

            // 读取维度数量
            let mut ndim_buf = [0u8; 4];
            stream.read_exact(&mut ndim_buf).await?;
            let ndim = LittleEndian::read_u32(&ndim_buf) as usize;

            // 读取各维度大小
            let mut shape = vec![0usize; ndim];
            for i in 0..ndim {
                let mut dim_buf = [0u8; 8];
                stream.read_exact(&mut dim_buf).await?;
                shape[i] = LittleEndian::read_u64(&dim_buf) as usize;
            }

            // 读取数据长度
            let mut data_len_buf = [0u8; 8];
            stream.read_exact(&mut data_len_buf).await?;
            let data_len = LittleEndian::read_u64(&data_len_buf) as usize;

            // 读取原始数据
            let mut data_buf = vec![0u8; data_len];
            stream.read_exact(&mut data_buf).await?;

            // 创建ArrayD
            // let array = ArrayD::from_shape_vec(shape, float_data.to_vec())?;
            arrays.insert(name, Tensor{data: data_buf, shape, dtype: dtype.unwrap()});
        }

        Ok(arrays)
    }

    pub(crate) async fn write_arrays(stream: &mut UnixStream, arrays: &HashMap<String, Tensor>) -> anyhow::Result<()> {
        // 写入magic header (成功标记)
        let magic_buf = InferenceProtocol::MAGIC_HEADER.to_le_bytes();
        stream.write_all(&magic_buf).await?;

        // 写入数组数量
        let count_buf = (arrays.len() as u32).to_le_bytes();
        stream.write_all(&count_buf).await?;

        for (name, array) in arrays {
            // 写入数组名长度和名称
            let name_bytes = name.as_bytes();
            let name_len_buf = (name_bytes.len() as u32).to_le_bytes();
            stream.write_all(&name_len_buf).await?;
            stream.write_all(name_bytes).await?;

            // 写入数据类型
            stream.write_all(&[array.dtype as u8]).await?;

            // 写入维度信息
            let shape = array.shape.clone();
            let ndim_buf = (shape.len() as u32).to_le_bytes();
            stream.write_all(&ndim_buf).await?;

            for dim in shape {
                let dim_buf = dim.to_le_bytes();
                stream.write_all(&dim_buf).await?;
            }

            // 获取原始数据并写入
            let raw_data = &array.data;
            let data_len_buf = (raw_data.len() as u64).to_le_bytes();
            stream.write_all(&data_len_buf).await?;
            stream.write_all(raw_data).await?;
        }

        Ok(())
    }

    pub(crate) async fn write_error(stream: &mut UnixStream, error: &str) -> anyhow::Result<()> {
        // 写入错误magic header
        let error_magic = InferenceProtocol::ERROR_MAGIC.to_le_bytes();
        stream.write_all(&error_magic).await?;

        // 写入错误消息
        let error_bytes = error.as_bytes();
        let error_len_buf = (error_bytes.len() as u32).to_le_bytes();
        stream.write_all(&error_len_buf).await?;
        stream.write_all(error_bytes).await?;

        Ok(())
    }
}
