import numpy as np
import time
from utils import extract, build_rankmat, similarity_mat, find_session_rank,\
    find_sessions_label
from sklearn.cluster import AgglomerativeClustering


class my_aggclustering():
    def __init__(
        self, df, alpha_cluster, cluster_threshold, alpha_label,
        init_commands_num=1, seq_len=12
    ):
        self.df = df
        self.alpha_cluster = alpha_cluster
        self.cluster_threshold = cluster_threshold
        self.alpha_label = alpha_label
        self.init_commands_num = init_commands_num
        self.seq_len = seq_len
        self.memory = None

    def fit(
        self, A=None, init_commands_list=None, seq_rank_list=None, E=None,
        full_memory=True, verbose=True
    ):
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
        new_df = (self.df).loc[self.df['Commands Length'] >
                               self.seq_len].reset_index(drop=True)
        # initialisation
        memory = dict()
        # memory['clustering'] = []
        # memory['init_commands_lists'] = []
        # memory['posterior_dicts'] = []
        # memory['expected_predictive_prob'] = []
        # memory['weights'] = []
        # memory['A_list'] = []
        # memory['E_list'] = []
        # memory['seq_rank_list'] = []
        start_time = time.time()

        sessions_list, _ = extract(new_df)
        if verbose:
            print('Clustering started.', end=' ')
        # find rank matrix A and distance matrix E
        if (A is None) or (init_commands_list is None) or (seq_rank_list is None):
            A, _, init_commands_list, seq_rank_list = build_rankmat(
                self.seq_len, sessions_list, self.init_commands_num, seq_rank_list=seq_rank_list)
        if (E is None):
            E = similarity_mat(A, np.ones(A.shape) *
                               self.alpha_cluster)   # alpha cluster
            # E[np.triu_indices(E.shape[0], 1)] -= np.amax(E)
            E = -E  # invert sign
            # E /= np.max(E)
        if full_memory:
            memory['A'] = A
            memory['init_commands_list'] = init_commands_list
            memory['seq_rank_list'] = seq_rank_list
        memory['E'] = E
        # start clustering, half max E for distance threshold!
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity='precomputed',
            linkage='single',  # minimum
            distance_threshold=self.cluster_threshold  # threshold
        ).fit(E)
        clusters = clustering.labels_
        n_clusters = clustering.n_clusters_
        if verbose:
            print('Total {} clusters.'.format(n_clusters), end=' ')

        # sessions_clusters = find_sessions_label(
        #     sessions_list, init_commands_list, clusters)

        # memory save
        memory['clustering'] = clustering
        memory['posterior_dict'] = dict()
        memory['expected_predictive_prob'] = dict()
        memory['weights'] = dict()
        # iterate over clusters
        for cluster in range(n_clusters):
            idx = (np.array(clusters) == cluster)
            init_commands_cluster = tuple(
                [init_commands_list[i] for i in range(len(idx)) if idx[i]])
            memory['weights'][init_commands_cluster] = np.sum(A[idx])
            memory['posterior_dict'][init_commands_cluster] = \
                np.sum(A[idx], axis=0) + np.ones(A.shape[1])*self.alpha_label
            memory['expected_predictive_prob'][init_commands_cluster] = \
                np.sum(((memory['posterior_dict'][init_commands_cluster]) /
                        np.sum(memory['posterior_dict'][init_commands_cluster]))**2) * np.sum(A[idx]) / np.sum(A)

        self.memory = memory
        if not full_memory:
            self.memory['A'] = A
            self.memory['init_commands_list'] = init_commands_list
            self.memory['seq_rank_list'] = seq_rank_list
        if verbose:
            print('Clustering finished. Time spent: {:.2f}s.'.format(
                time.time()-start_time))

        return memory

    def predict(self, df):
        if self.memory is None:
            raise ValueError('No training performed.')
        # ensure the length of session is not too small
        new_df = df.loc[df['Commands Length'] >
                        self.seq_len].reset_index(drop=True)
        sessions_list, _ = extract(new_df)
        score_list = [-1 for i in range(len(sessions_list))]
        session_rank_list = find_session_rank(
            sessions_list, self.memory['seq_rank_list'])

        for i, session in enumerate(sessions_list):
            rank = session_rank_list[i]
            for key, val in self.memory['posterior_dict'].items():
                if tuple(session[:self.init_commands_num]) in key:
                    val = np.append(val, self.alpha_label)
                    score_list[i] = val[rank]/sum(val)
                    break

        score_list = [score if score != -1 else 1 /
                      (self.memory['A'].shape[1]+1) for score in score_list]

        return score_list


# class my_aggclustering2():
#     def __init__(self, df, alpha_cluster, cluster_threshold, alpha_label, max_depth=1, seq_len=12):
#         self.df = df
#         self.alpha_cluster = alpha_cluster
#         self.cluster_threshold = cluster_threshold
#         self.alpha_label = alpha_label
#         self.max_depth = max_depth
#         self.seq_len = seq_len
#         self.memory = None

#     def fit(self):
#         """
#         This function implements agglomerative clustering

#         :param df: the dataframe, must contain Commands and Commands Length columns
#         :param alpha: float, used for multinomial-dirichlet
#         :param n_splits: number of splits for k-fold, default 10
#         :param seq_len: length of sequences of commands for analysis, default 12
#         :param init_len: length of initial commands we use for clustering, default 1

#         :return memory: dictionary with information
#         """
#         # ensure the length of session is not too small
#         new_df = (self.df).loc[self.df['Commands Length'] >
#                                self.seq_len].reset_index(drop=True)
#         # initialisation
#         memory = dict()
#         memory['clustering'] = []
#         memory['init_commands_lists'] = []
#         memory['posterior_dicts'] = []
#         memory['expected_predictive_prob'] = []
#         memory['A_list'] = []
#         memory['E_list'] = []
#         memory['seq_rank_list'] = []
#         memory['cluster_num'] = 0
#         start_time = time.time()

#         sessions_list, _ = extract(new_df)
#         sessions_rank_list = None
#         seq_rank_list = None
#         # iterate over depth
#         for depth in range(self.max_depth):
#             print('Depth {} started.'.format(depth))
#             # find rank matrix A and distance matrix E
#             A, sessions_rank_list, init_commands_list, seq_rank_list = build_rankmat(
#                 self.seq_len, sessions_list, depth+1, sessions_rank_list, seq_rank_list)
#             E = similarity_mat(A, np.ones(A.shape) *
#                                self.alpha_cluster)   # alpha cluster
#             E[np.triu_indices(E.shape[0], 1)] -= np.amax(E)
#             E = -E  # make distance matrix positive
#             E /= np.max(E)
#             memory['A_list'].append(A)
#             memory['E_list'].append(E)

#             # start clustering, half max E for distance threshold!
#             clustering = AgglomerativeClustering(n_clusters=None,
#                                                  affinity='precomputed',
#                                                  linkage='single',  # minimum
#                                                  distance_threshold=self.cluster_threshold  # threshold
#                                                  ).fit(E)
#             clusters = clustering.labels_
#             n_clusters = clustering.n_clusters_
#             sessions_clusters = find_sessions_label(
#                 sessions_list, init_commands_list, clusters)

#             # memory save
#             memory['clustering'].append(clustering)
#             memory['init_commands_lists'].append(init_commands_list)

#             memory['posterior_dicts'].append(dict())
#             memory['expected_predictive_prob'].append(dict())
#             sessions_list_next = []
#             sessions_rank_list_next = []
#             # iterate over clusters
#             for cluster in range(n_clusters):
#                 idx = (np.array(clusters) == cluster)
#                 init_commands_cluster = tuple(
#                     [init_commands_list[i] for i in range(len(idx)) if idx[i] == True])
#                 memory['posterior_dicts'][-1][init_commands_cluster] = \
#                     np.sum(A[idx], axis=0) + \
#                     np.ones(A.shape[1])*self.alpha_label
#                 memory['expected_predictive_prob'][-1][init_commands_cluster] = \
#                     np.sum(((memory['posterior_dicts'][-1][init_commands_cluster]) /
#                             np.sum(memory['posterior_dicts'][-1][init_commands_cluster]))**2)

#                 if sum(idx) == 1:
#                     sessions_list_next += [sessions_list[i] for i in range(
#                         len(sessions_clusters)) if sessions_clusters[i] == cluster]
#                     sessions_rank_list_next += [sessions_rank_list[i] for i in range(
#                         len(sessions_clusters)) if sessions_clusters[i] == cluster]
#                 if (sum(idx) != 1) or (depth == self.max_depth-1):
#                     memory['cluster_num'] += 1

#             if len(sessions_list_next) == 0:
#                 print('Depth {} finished, no further depth required. Time spent: {:.2f}s.'.format(
#                     depth, time.time()-start_time))
#                 break
#             else:
#                 sessions_list = sessions_list_next[:]
#                 sessions_rank_list = sessions_rank_list_next[:]
#                 print('Depth {} finished. Time spent: {:.2f}s.'.format(
#                     depth, time.time()-start_time))
#         memory['seq_rank_list'] = seq_rank_list
#         self.memory = memory

#         return memory

#     def predict(self, df):
#         if self.memory == None:
#             raise ValueError('No training performed.')
#         max_depth = len(self.memory['A_list'])
#         # ensure the length of session is not too small
#         new_df = df.loc[df['Commands Length'] >
#                         self.seq_len].reset_index(drop=True)
#         sessions_list, _ = extract(new_df)
#         score_list = [[] for i in range(len(sessions_list))]
#         session_rank_list = find_session_rank(
#             sessions_list, self.memory['seq_rank_list'])

#         for i, session in enumerate(sessions_list):
#             for depth in range(max_depth):
#                 rank = session_rank_list[i]
#                 for key, val in self.memory['posterior_dicts'][depth].items():
#                     if tuple(session[:depth+1]) in key:
#                         val = np.append(val, self.alpha_label)
#                         score_list[i].append(val[rank]/sum(val))
#                         break
#                 if len(score_list[i]) != depth+1:
#                     break

#         score_list = [score if score != [] else [
#             1/(self.memory['A_list'][0].shape[1]+1)] for score in score_list]

#         return score_list

class my_aggclustering3():
    def __init__(
        self, df, alpha_cluster, cluster_threshold, alpha_label, seq_len=12
    ):
        self.df = (df).loc[df['Commands Length'] >
                           seq_len].reset_index(drop=True)
        self.alpha_cluster = alpha_cluster
        self.cluster_threshold = cluster_threshold
        self.alpha_label = alpha_label
        self.seq_len = seq_len
        self.depth = 0
        self.sessions_clusters = []
        self.init_commands_clusters = []
        self.posteriors = []
        self.expected_predictive_probs = []
        self.weights = []
        self.sessions_rank_list = None
        self.seq_rank_list = None

    def fit(self, depth=None, verbose=True):
        """
        This function implements agglomerative clustering
        """
        # # ensure the length of session is not too small
        # new_df = (self.df).loc[self.df['Commands Length'] >
        #                        self.seq_len].reset_index(drop=True)
        # initialisation
        if depth is None:
            depth = self.depth+1
        if depth != self.depth+1:
            raise ValueError('Wrong depth!')
        if verbose:
            print('Depth {}:'.format(depth), end=' ')
            start_time = time.time()
        sessions_list, _ = extract(self.df)
        self.init_commands_clusters.append([])
        self.posteriors.append([])
        self.expected_predictive_probs.append([])
        self.weights.append([])

        if depth == 1:
            A, self.sessions_rank_list, init_commands_list, self.seq_rank_list = \
                build_rankmat(self.seq_len, sessions_list, depth)
        else:
            add = 0
            init_commands_list = []
            clusters_temp = []
            A = np.zeros((0, max(self.sessions_rank_list)+1), dtype='int')
            A_temp = np.zeros((0, max(self.sessions_rank_list)+1), dtype='int')
            for cluster in range(max(self.sessions_clusters[-1])+1):
                idx = (np.array(self.sessions_clusters[-1]) == cluster)
                sessions_list_clustering = \
                    [sessions_list[i]
                        for i in range(len(sessions_list)) if idx[i]]
                A_cluster_1, _, init_commands_list_cluster, _ =\
                    build_rankmat(
                        self.seq_len,
                        sessions_list_clustering,
                        depth,
                        [self.sessions_rank_list[i]
                            for i in range(len(sessions_list)) if idx[i]],
                        self.seq_rank_list
                    )
                A_cluster = np.zeros(
                    (A_cluster_1.shape[0], A.shape[1]), dtype='int')
                A_cluster[:A_cluster_1.shape[0],
                          :A_cluster_1.shape[1]] = 1*A_cluster_1
                E_cluster = - \
                    similarity_mat(A_cluster, np.ones(A_cluster.shape)
                                   * self.alpha_cluster)
                # start clustering
                if A_cluster.shape[0] <= 1:  # can be used for leaf size\
                    labels = [0 for i in range(1)]
                    n_labels = 1
                else:
                    clustering_cluster = AgglomerativeClustering(
                        n_clusters=None, affinity='precomputed',
                        linkage='single',  # minimum
                        distance_threshold=self.cluster_threshold  # threshold
                    ).fit(E_cluster)
                    labels = clustering_cluster.labels_
                    n_labels = clustering_cluster.n_clusters_
                clusters_temp += [i+add for i in labels]
                add += n_labels
                init_commands_list += init_commands_list_cluster
                A_temp = np.concatenate((A_temp, A_cluster))
            sessions_clusters_temp = find_sessions_label(
                sessions_list, init_commands_list, clusters_temp
            )
            init_commands_clusters_temp = []
            for cluster_temp in range(add):
                idx = (np.array(clusters_temp) == cluster_temp)
                A = np.concatenate(
                    (A, np.sum(A_temp[idx, :], axis=0).reshape((1, -1))))
                init_commands_clusters_temp.append(
                    [init_commands_list[i] for i in range(len(idx)) if idx[i]]
                )

        E = -similarity_mat(A, np.ones(A.shape) * self.alpha_cluster)
        # start clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity='precomputed',
            linkage='single',  # minimum
            distance_threshold=self.cluster_threshold  # threshold
        ).fit(E)
        clusters = clustering.labels_
        n_clusters = clustering.n_clusters_
        if verbose:
            print('{} clusters exist.'.format(n_clusters), end=' ')

        if depth == 1:
            self.sessions_clusters.append(
                find_sessions_label(
                    sessions_list, init_commands_list, clusters)
            )
        else:
            self.sessions_clusters.append(
                [clusters[cluster_temp]
                    for cluster_temp in sessions_clusters_temp]
            )

        # iterate over clusters
        for cluster in range(n_clusters):
            idx = (np.array(clusters) == cluster)
            if depth == 1:
                self.init_commands_clusters[-1].append(
                    tuple(
                        [init_commands_list[i]
                            for i in range(len(idx)) if idx[i]]
                    )
                )
            else:
                temp_cluster = [init_commands_clusters_temp[i]
                                for i in range(len(idx)) if idx[i]]
                self.init_commands_clusters[-1].append(
                    tuple(
                        [item for sublist in temp_cluster for item in sublist]
                    )
                )
            self.posteriors[-1].append(
                np.sum(A[idx], axis=0) +
                np.ones(A.shape[1])*self.alpha_label
            )
            self.expected_predictive_probs[-1].append(
                np.sum(
                    (self.posteriors[-1][-1] /
                        np.sum(self.posteriors[-1][-1]))**2
                ) * np.sum(A[idx]) / np.sum(A)
            )
            self.weights[-1].append(np.sum(A[idx]))

        self.depth += 1
        if verbose:
            print('Time spent {:.2f}s.'.format(time.time()-start_time))

    def predict(self, df, depth):
        if self.depth == 0:
            raise ValueError('No training performed.')
        # ensure the length of session is not too small
        new_df = df.loc[df['Commands Length'] >
                        self.seq_len].reset_index(drop=True)
        sessions_list, _ = extract(new_df)
        score_list = [-1 for i in range(len(sessions_list))]
        session_rank_list = find_session_rank(
            sessions_list, self.seq_rank_list)

        for i, session in enumerate(sessions_list):
            rank = session_rank_list[i]
            for key, val in zip(self.init_commands_clusters[depth-1], self.posteriors[depth-1]):
                if tuple(session[:depth]) in key:
                    val = np.append(val, self.alpha_label)
                    score_list[i] = val[rank]/sum(val)
                    break

        score_list = [score if score != -1 else 1 /
                      (max(self.sessions_rank_list)+1) for score in score_list]

        return score_list


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
    my_agg = my_aggclustering3(msdat[:1000], 0.1, -1, 0.1)
    my_agg.fit()
    my_agg.fit()
    my_agg.fit()
