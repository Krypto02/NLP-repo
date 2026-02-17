# Data Directory

This directory contains the datasets used for the Misogyny Detection in Memes project.

## Files

### training.csv
- **Description**: Training dataset containing meme text content and classification labels
- **Format**: Tab-separated values (TSV)
- **Columns**:
  - `file_name`: Image file name of the meme
  - `misogynous`: Binary label (0 = non-misogynous, 1 = misogynous)
  - `shaming`, `stereotype`, `objectification`, `violence`: Additional classification categories
  - Text columns: Multiple columns containing the meme text content
- **Size**: 7,500 samples
- **Split**: 80% training (6,000), 20% test (1,500)

## Usage

Data files are loaded from notebooks using relative paths:

```python
import pandas as pd
df = pd.read_csv("../data/training.csv", sep="\t", header=0)
```

## Note

Due to privacy and copyright considerations, the actual data files are not included in the repository. 
Please refer to the main README.md for instructions on obtaining the dataset.
