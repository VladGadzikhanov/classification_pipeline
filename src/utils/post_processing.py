import pandas as pd


def aggregate_detailed_classes(pred_df, dict_class, map_only_preds=False):
    num_classes = len(dict_class.keys())
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    mapping_idx = {f"prob_{i}": dict_class[i] for i in range(num_classes)}
    inv_mapping = {v: f"prob_{v}" for k, v in dict_class.items()}

    t = pred_df[prob_cols].T
    t.index = t.index.map(mapping_idx)
    t = t.groupby(t.index).sum()
    t.index = t.index.map(inv_mapping)
    t = t.T

    t[["obj_id", "true"]] = pred_df[["obj_id", "true"]]
    prob_cols = [f"prob_{i}" for i in range(1 + max(dict_class.values()))]
    mapping_idx = {f"prob_{i}": i for i in range(num_classes)}
    t["pred"] = t[prob_cols].idxmax(axis=1).map(mapping_idx)
    if not map_only_preds:
        t["true"] = t["true"].map(dict_class)
    return t


def aggregate_obj_results(pred_df, num_classes, method="gmean"):
    from scipy.stats import gmean

    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    mapping_idx = {f"prob_{i}": i for i in range(num_classes)}

    # aggregate probs for each obj
    if method == "gmean":
        probs = pred_df.groupby("obj_id")[prob_cols].mean()
        for col in prob_cols:
            probs[col] = pred_df.groupby("obj_id")[col].apply(gmean)
    elif method == "mean":
        probs = pred_df.groupby("obj_id")[prob_cols].mean()
    else:
        raise (Exception("Not implemented method `%s`" % method))

    probs["pred"] = probs[prob_cols].idxmax(axis=1).map(mapping_idx)

    # take true label for obj
    true = pred_df.groupby("obj_id")["true"].first().reset_index()

    # merge
    result = pd.merge(probs, true, on="obj_id")

    return result
