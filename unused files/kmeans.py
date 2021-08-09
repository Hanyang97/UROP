import numpy as np
import time
from utils import extract, build_rankmat, similarity_mat, find_sessions_label, \
    make_seq_counts, sort_counts, make_seq_list, predict_next_command, \
    find_initial_commands_idx, sort_idx, \
    iterate_zhu
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from numpy.linalg import svd
from scipy.sparse.linalg import svds


def my_kmeans(df, alpha, n_clusters=21, n_splits=10, seq_len=12, init_len=1):
    """
    This function implements agglomerative clustering

    :param df: the dataframe, must contain Commands and Commands Length columns
    :param alpha: float, used for multinomial-dirichlet
    :param n_clusters: number of clusters for kmeans, default 21
    :param n_splits: number of splits for k-fold, default 10
    :param seq_len: length of sequences of commands for analysis, default 12
    :param init_len: length of initial commands we use for clustering, default 1

    :return memory: dictionary with information
    """
    # initialisation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rounds = 0
    new_df = df.loc[df['Commands Length'] > seq_len].reset_index(drop=True)

    memory = dict()
    memory['acc_train_list'] = [[] for i in range(n_splits)]
    memory['acc_test_list'] = [[] for i in range(n_splits)]
    memory['total_train_list'] = [[] for i in range(n_splits)]
    memory['total_test_list'] = [[] for i in range(n_splits)]
    memory['kmeans'] = []
    memory['init_commands_list'] = []
    memory['train_sessions_labels'] = []
    memory['test_sessions_labels'] = []

    for train_idx, test_idx in kf.split(new_df):
        start_time = time.time()

        # initialise train and test sets
        X_train, X_test = new_df.iloc[train_idx, :], new_df.iloc[test_idx, :]
        sessions_list_train, _ = extract(X_train)
        sessions_list_test, _ = extract(X_test)

        # find rank matrix A
        A, _, init_commands_list = build_rankmat(
            seq_len, sessions_list_train, init_len)

        # start clustering
        print('Round {}/{}, start clustering.'.format(rounds+1, n_splits), end=' ')

        # spectral decomposition
        # # Full SVD (might not be feasible for VERY large matrices) & takes only dense objects (np.array) as input
        # U, D, V = svd(A)
        # Approximate method (good for very large AND/OR sparse matrices)
        U, D, V = svds(A.astype('float'), k=20)
        if 0 in D:
            U, D, V = svds(A.astype('float'), k=list(D).index(0))
        # Use zhu on D to select dimension
        d = iterate_zhu(D[::-1], 2)[-1]
        # Dimensionality reduction
        X = np.matmul(U[:, ::-1][:, :d], np.diag(np.sqrt(D[::-1][:d])))
        # scale X
        X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
        print('Finished SVD.', end=' ')
        # kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        print('Finished K-means.')
        train_labels = kmeans.labels_

        total_train = 0
        accurate_train = 0
        total_test = 0
        accurate_test = 0
        # find labels
        train_sessions_labels = find_sessions_label(
            sessions_list_train, init_commands_list, train_labels)
        test_sessions_labels = find_sessions_label(
            sessions_list_test, init_commands_list, train_labels)
        # memory save
        memory['kmeans'].append(kmeans)
        memory['init_commands_list'].append(init_commands_list)
        memory['train_sessions_labels'].append(train_sessions_labels)
        memory['test_sessions_labels'].append(test_sessions_labels)

        # iterate over labels
        for label in range(n_clusters):
            ######### train ############
            selected_sessions_list_train = [sessions_list_train[i] for i in range(len(sessions_list_train))
                                            if train_sessions_labels[i] == label]
            # most common command in the selected training session
            # if unseen command given, predict next to be most common command
            most_common_command = sort_counts(
                make_seq_counts(1, selected_sessions_list_train))[0][0]
            # sequence X and y for training
            seq_train = make_seq_list(seq_len+1, selected_sessions_list_train)
            X_seq_train = np.array([i[:-1] for i in seq_train])
            y_seq_train = np.array([i[-1] for i in seq_train])
            # predict next command
            y_train_predict = predict_next_command(seq_len,
                                                   selected_sessions_list_train,
                                                   X_seq_train,
                                                   most_common_command)
            total_train += len(y_train_predict)
            accurate_train += sum(y_seq_train == y_train_predict)
            # memory save
            memory['acc_train_list'][rounds].append(
                sum(y_seq_train == y_train_predict))
            memory['total_train_list'][rounds].append(len(y_train_predict))
            ######### test ############
            selected_sessions_list_test = [sessions_list_test[i] for i in range(len(sessions_list_test))
                                           if test_sessions_labels[i] == label]
            # we want selected test sessions to be non-empty
            # selected test sessions could be empty for centain labels
            if len(selected_sessions_list_test) > 0:
                # X and y for testing
                seq_test = make_seq_list(
                    seq_len+1, selected_sessions_list_test)
                X_seq_test = np.array([i[:-1] for i in seq_test])
                y_seq_test = np.array([i[-1] for i in seq_test])

                y_test_predict = predict_next_command(seq_len,
                                                      selected_sessions_list_train,
                                                      X_seq_test,
                                                      most_common_command)
                total_test += len(y_test_predict)
                accurate_test += sum(y_seq_test == y_test_predict)
                # memory save
                memory['acc_test_list'][rounds].append(
                    sum(y_seq_test == y_test_predict))
                memory['total_test_list'][rounds].append(len(y_test_predict))
            else:
                # memory save
                memory['acc_test_list'][rounds].append(0)
                memory['total_test_list'][rounds].append(0)

            if label//10 == label/10 and total_test != 0 and total_train != 0:
                print('{} clusters completed, {:.2f} seconds spent, train accuracy {:.2f}, test accuracy {:.2f}'
                      .format(label+1, time.time()-start_time, accurate_train/total_train, accurate_test/total_test))

        # the last cluster in test but not in train, created by find_sessions_label()
        selected_sessions_list_test = [sessions_list_test[i] for i in range(
            len(sessions_list_test)) if test_sessions_labels[i] == n_clusters]

        if len(selected_sessions_list_test) > 0:
            # we select the most common command in the whole training dataset
            most_common_command = sort_counts(
                make_seq_counts(1, selected_sessions_list_train))[0][0]
            seq_test = make_seq_list(seq_len+1, selected_sessions_list_test)
            X_seq_test = np.array([i[:-1] for i in seq_test])
            y_seq_test = np.array([i[-1] for i in seq_test])

            y_test_predict = predict_next_command(seq_len,
                                                  selected_sessions_list_train,
                                                  X_seq_test,
                                                  most_common_command)
            total_test += len(y_test_predict)
            accurate_test += sum(y_seq_test == y_test_predict)

            # memory save
            memory['acc_test_list'][rounds].append(
                sum(y_seq_test == y_test_predict))
            memory['total_test_list'][rounds].append(len(y_test_predict))

        train_accuracy = accurate_train/total_train
        test_accuracy = accurate_test/total_test
        print('Round {}/{} completed, {:.2f} seconds spent, train accuracy {:.2f}, test accuracy {:.2f}'
              .format(rounds+1, n_splits, time.time()-start_time, train_accuracy, test_accuracy))
        rounds += 1

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
    msdat = msdat.iloc[:500, :]
    my_kmeans(msdat, 0.1, 2)
