def generate_walk_forward_splits(index, train_years, val_years, test_years):
    years = sorted(index.year.unique())
    splits = []
    start_pos = 0
    while True:
        train_block = years[start_pos:start_pos + train_years]
        val_block = years[start_pos + train_years:start_pos + train_years + val_years]
        test_block = years[start_pos + train_years + val_years:start_pos + train_years + val_years + test_years]
        if len(train_block) < train_years or len(val_block) < val_years or len(test_block) < test_years:
            break
        splits.append({
            "train_mask": index.year.isin(train_block),
            "val_mask": index.year.isin(val_block),
            "test_mask": index.year.isin(test_block),
            "train_years": train_block,
            "val_years": val_block,
            "test_years": test_block,
        })
        start_pos += 1
    return splits
