use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = rust_model_inferencer_pyo3::stub_info()?;
    stub.generate()?;
    Ok(())
}