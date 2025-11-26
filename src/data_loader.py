import numpy as np
import pandas as pd
import os

def load_data(data_dir):

    def p(fname):
        return os.path.join(data_dir, fname)

    df_expr = pd.read_csv(p("expr_mat.csv"))
    expr_mat = df_expr.values.astype(np.float32)

    df_scores = pd.read_csv(p("scores_3_z.csv"))
    scores_3 = df_scores.values.astype(np.float32)

    df_labels = pd.read_csv(p("labels_cls.csv"))
    labels_cls = df_labels.values.squeeze().astype(np.int64)

    df_meta = pd.read_csv(p("meta_id.csv"))
    meta_ids = df_meta.values.squeeze().astype(str)

    df_well = pd.read_csv(p("well_id.csv"))
    well_ids = df_well.values.squeeze().astype(str)

    try:
        df_old = pd.read_csv(p("activation_score_old.csv"))
        act_old = df_old.values.squeeze().astype(np.float32)
    except FileNotFoundError:
        act_old = None

    try:
        df_cluster = pd.read_csv(p("cluster_id.csv"))
        cluster_ids = df_cluster.values.squeeze().astype(str)
    except FileNotFoundError:
        cluster_ids = None

    return expr_mat, scores_3, labels_cls, meta_ids, well_ids, act_old, cluster_ids
