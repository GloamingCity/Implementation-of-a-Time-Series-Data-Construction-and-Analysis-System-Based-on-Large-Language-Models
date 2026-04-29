GitHub LFS rejects any single object larger than 2 GiB.

The following source files therefore cannot be stored at their original paths in this repository:

- Dataset/UEA&UCR_Multivariate_Time_Series_Classification_Archive/Multivariate/FruitFlies/FruitFlies.arff
- Dataset/UEA&UCR_Multivariate_Time_Series_Classification_Archive/Multivariate/UrbanSound/UrbanSound.arff
- Models/Qwen2.5-3B-Instruct/model-00001-of-00002.safetensors
- Models/Qwen2.5-3B-Instruct/model-00002-of-00002.safetensors
- Models/Qwen3-4B-Instruct-2507/model-00001-of-00003.safetensors
- Models/Qwen3-4B-Instruct-2507/model-00002-of-00003.safetensors

Their complete contents are preserved under SplitLargeFiles/ as sub-2-GiB parts tracked by Git LFS.

Use SplitLargeFiles/reassemble.ps1 to reconstruct the original files from those parts.