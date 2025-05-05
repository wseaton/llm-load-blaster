pub mod data {
    use std::fs::File;

    use hf_hub::{
        api::tokio::{Api, ApiRepo},
        Repo,
    };
    use parquet::{
        file::reader::{FileReader, SerializedFileReader},
        record::RecordReader,
    };

    use tracing::{error, info};

    pub const READ_SIZE: usize = 100;

    #[derive(Debug)]
    pub struct UltraFeedbackItem {
        pub prompt: Option<String>,
    }

    impl ::parquet::record::RecordReader<UltraFeedbackItem> for Vec<UltraFeedbackItem> {
        fn read_from_row_group(
            &mut self,
            row_group_reader: &mut dyn ::parquet::file::reader::RowGroupReader,
            num_records: usize,
        ) -> Result<(), ::parquet::errors::ParquetError> {
            use ::parquet::column::reader::ColumnReader;
            // using existing param
            let mut name_to_index = std::collections::HashMap::new();
            for (idx, col) in row_group_reader
                .metadata()
                .schema_descr()
                .columns()
                .iter()
                .enumerate()
            {
                name_to_index.insert(col.name().to_string(), idx);
            }
            for _ in 0..num_records {
                self.push(UltraFeedbackItem {
                    prompt: Default::default(),
                })
            }
            let records = self;
            {
                let idx: usize = match name_to_index.get("prompt") {
                    Some(&col_idx) => col_idx,
                    None => {
                        return Err(::parquet::errors::ParquetError::General(
                            "Column 'prompt' not found".into(),
                        ));
                    }
                };
                if let Ok(column_reader) = row_group_reader.get_column_reader(idx) {
                    {
                        let mut vals = Vec::new();
                        if let ColumnReader::ByteArrayColumnReader(mut typed) = column_reader {
                            let mut definition_levels = Vec::new();
                            let (_total_num, valid_num, decoded_num) = typed.read_records(
                                num_records,
                                Some(&mut definition_levels),
                                None,
                                &mut vals,
                            )?;
                            if valid_num != decoded_num {
                                {
                                    return Err(::parquet::errors::ParquetError::General(
                                        "Invalid number of records".into(),
                                    ));
                                };
                            }
                        } else {
                            {
                                return Err(::parquet::errors::ParquetError::General(
                                    "Invalid column reader type".into(),
                                ));
                            };
                        }
                        for (i, r) in &mut records[..num_records].iter_mut().enumerate() {
                            r.prompt = Some(String::from(
                                std::str::from_utf8(vals[i].data())
                                    .expect("invalid UTF-8 sequence"),
                            ));
                        }
                    }
                } else {
                    return Err(::parquet::errors::ParquetError::General(
                        "Failed to get next column".into(),
                    ));
                }
            }
            Ok(())
        }
    }

    async fn sibling_to_parquet(
        rfilename: &str,
        repo: &ApiRepo,
    ) -> anyhow::Result<SerializedFileReader<File>> {
        let local = repo.get(rfilename).await?;
        let file = File::open(local)?;
        let reader = SerializedFileReader::new(file)?;
        Ok(reader)
    }

    pub async fn download_parquet_dataset<T>(
        dataset_name: &str,
        api: &Api,
        num_samples: usize,
    ) -> anyhow::Result<Vec<T>>
    where
        Vec<T>: RecordReader<T>,
        T: Send + 'static,
    {
        let repo = Repo::with_revision(
            dataset_name.to_string(),
            hf_hub::RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let repo = api.repo(repo);

        let read_size = if num_samples < READ_SIZE {
            num_samples
        } else {
            READ_SIZE
        };

        let info = match repo.info().await {
            Ok(info) => info,
            Err(e) => {
                error!("Failed to get dataset info: {}", e);
                return Err(anyhow::anyhow!("Failed to get dataset info"));
            }
        };

        info!("Dataset info: {:?}", info);

        let parquet_filenames = info
            .siblings
            .into_iter()
            .filter_map(|s| {
                if s.rfilename.ends_with(".parquet") {
                    Some(s.rfilename)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut files = Vec::new();
        for filename in parquet_filenames {
            let reader_result = sibling_to_parquet(&filename, &repo).await;
            files.push(reader_result);
        }

        let files: Result<Vec<_>, _> = files.into_iter().collect();
        let files = files?;
        let mut chunks: Vec<T> = Vec::new();
        for file in files {
            let mut row_group = file.get_row_group(0)?;
            // Call the trait method through Vec's implementation of RecordReader trait
            chunks.read_from_row_group(&mut *row_group, read_size)?;
        }

        Ok(chunks)
    }
}
