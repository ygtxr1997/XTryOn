from tqdm import tqdm

from datasets.cp_datasets import CPDataset, CPDatasetLevel2  # is under XTryOn/

dataset = CPDatasetLevel2("/cfs/yuange/datasets/xss/standard/hoodie/", mode='train', image_size=256)
n = len(dataset)

for i in tqdm(range(1, n)):
    _ = dataset[-i]
