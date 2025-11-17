import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = ""
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "emirhanakku/global-gaming-and-esports-growth-dataset-20102025",
  file_path,

)

print("First 5 records:", df.head())