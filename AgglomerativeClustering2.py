import numpy as np
import time
from utils import extract, build_rankmat, similarity_mat, find_sessions_label, \
    make_seq_counts, sort_counts, make_seq_list, predict_next_command, \
    find_initial_commands_idx, sort_idx
from sklearn.cluster import AgglomerativeClustering


# only cluster previously unclustered
def my_agglomerativeclustering(df, alpha_cluster, cluster_threshold, alpha_label, max_depth=1, seq_len=12):
    """
    This function implements agglomerative clustering

    :param df: the dataframe, must contain Commands and Commands Length columns
    :param alpha: float, used for multinomial-dirichlet
    :param n_splits: number of splits for k-fold, default 10
    :param seq_len: length of sequences of commands for analysis, default 12
    :param init_len: length of initial commands we use for clustering, default 1

    :return memory: dictionary with information
    """
    # ensure the length of session is not too small
    new_df = df.loc[df['Commands Length'] > seq_len].reset_index(drop=True)
    # initialisation
    memory = dict()
    memory['clustering'] = []
    memory['init_commands_lists'] = []
    memory['posterior_dicts'] = []
    memory['expected_predictive_prob'] = []
    memory['A_list'] = []
    memory['E_list'] = []
    memory['cluster_num'] = 0
    start_time = time.time()

    sessions_list, _ = extract(new_df)
    # iterate over depth
    for depth in range(max_depth):
        print('Depth {} started.'.format(depth))
        # find rank matrix A and distance matrix E
        A, _, init_commands_list = build_rankmat(
            seq_len, sessions_list, depth+1)
        E = similarity_mat(A, np.ones(A.shape)*alpha_cluster)   # alpha cluster
        E[np.triu_indices(E.shape[0], 1)] -= np.amax(E)
        E = -E  # make distance matrix positive
        memory['A_list'].append(A)
        memory['E_list'].append(E)

        # start clustering, half max E for distance threshold!
        clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='precomputed',
                                             linkage='single',  # minimum
                                             distance_threshold=cluster_threshold  # threshold
                                             ).fit(E)
        clusters = clustering.labels_
        n_clusters = clustering.n_clusters_
        sessions_clusters = find_sessions_label(
            sessions_list, init_commands_list, clusters)

        # memory save
        memory['clustering'].append(clustering)
        memory['init_commands_lists'].append(init_commands_list)

        memory['posterior_dicts'].append(dict())
        memory['expected_predictive_prob'].append(dict())
        sessions_list_next = []
        # iterate over clusters
        for cluster in range(n_clusters):
            idx = (np.array(clusters) == cluster)
            init_commands_cluster = tuple(
                [init_commands_list[i] for i in range(len(idx)) if idx[i] == True])
            memory['posterior_dicts'][-1][init_commands_cluster] = \
                np.sum(A[idx], axis=0) + np.ones(A.shape[1])*alpha_label
            memory['expected_predictive_prob'][-1][init_commands_cluster] = \
                np.sum(((memory['posterior_dicts'][-1][init_commands_cluster]) /
                        np.sum(memory['posterior_dicts'][-1][init_commands_cluster]))**2)

            if sum(idx) == 1:
                sessions_list_next += [sessions_list[i] for i in range(
                    len(sessions_clusters)) if sessions_clusters[i] == cluster]
            else:
                memory['cluster_num'] += 1

        if len(sessions_list_next) == 0:
            print('Depth {} finished, no further depth required. Time spent: {:.2f}s.'.format(
                depth, time.time()-start_time))
            break
        else:
            sessions_list = sessions_list_next[:]
            print('Depth {} finished. Time spent: {:.2f}s.'.format(
                depth, time.time()-start_time))

    return memory


# test
if __name__ == '__main__':
    msdat_dir = '/home/hpms/Microsoft.IoT-Dump1.json'
    import codecs
    import json
    import pandas as pd
    with codecs.open(msdat_dir, 'r', 'utf-8-sig') as f:
        msdat = json.load(f)
    msdat = pd.DataFrame(msdat)
    msdat['Commands'] = [tuple(session) for session in msdat['Commands']]
    msdat = msdat.drop_duplicates(subset='Commands').reset_index(
        drop=True)  # drop duplicates
    msdat['Commands'] = [list(session) for session in msdat['Commands']]
    msdat['Commands Length'] = [len(session) for session in msdat['Commands']]
    my_agglomerativeclustering(msdat[:1000], 0.1, 0.7, 0.1, 2)
