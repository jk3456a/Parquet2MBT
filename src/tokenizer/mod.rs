use anyhow::Result;
use tokenizers::{Tokenizer, EncodeInput, Encoding};

pub mod pool;
pub use pool::{TokenizerPool, TokenizedBatch};

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    pub fn from_path(path: &str) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("load tokenizer: {}: {}", path, e))?;
        Ok(Self { inner })
    }


    pub fn encode_batch_ids(&self, texts: &[String], add_special: bool) -> Result<Vec<Vec<u32>>> {
        let inputs: Vec<EncodeInput> = texts.iter().map(|t| t.as_str().into()).collect();
        let encs: Vec<Encoding> = self.inner.encode_batch(inputs, add_special).map_err(|e| anyhow::anyhow!("encode_batch: {}", e))?;
        Ok(encs.into_iter().map(|e| e.get_ids().to_vec()).collect())
    }

    pub fn vocab_size(&self, with_added: bool) -> usize {
        self.inner.get_vocab_size(with_added)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}


