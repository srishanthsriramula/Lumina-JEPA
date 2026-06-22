from datasets import load_dataset
ds = load_dataset("pierreguillou/DocLayNet-small", split="validation", trust_remote_code=True)
print("Features:", ds.features)
print("bboxes_block:", ds[0]['bboxes_block'][:2])
print("categories:", ds[0]['categories'][:2])
