import kagglehub

# Download latest version
path = kagglehub.dataset_download("sharangjindal/home-credit-dataset", output_dir='data/inputs/')

print("Path to dataset files:", path)
