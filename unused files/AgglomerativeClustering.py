import numpy as np
import time
from utils import extract, build_rankmat, similarity_mat, find_sessions_label, \
    make_seq_counts, sort_counts, make_seq_list, predict_next_command, \
    find_initial_commands_idx, sort_idx
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering

# predict directly based on max prob, like decision tree
def my_agglomerativeclustering(df, alpha, n_splits=10, seq_len=12, init_len=1):
    """
    This function implements agglomerative clustering

    :param df: the dataframe, must contain Commands and Commands Length columns
    :param alpha: float, used for multinomial-dirichlet
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
    memory['clustering'] = []
    memory['init_commands_list'] = []
    memory['train_sessions_labels'] = []
    memory['test_sessions_labels'] = []

    for train_idx, test_idx in kf.split(new_df):
        start_time = time.time()

        # initialise train and test sets
        X_train, X_test = new_df.iloc[train_idx, :], new_df.iloc[test_idx, :]
        sessions_list_train, _ = extract(X_train)
        sessions_list_test, _ = extract(X_test)

        # find rank matrix A and distance matrix E
        A, _, init_commands_list = build_rankmat(
            seq_len, sessions_list_train, init_len)
        E = similarity_mat(A, np.ones(A.shape)*alpha)   # alpha
        E[np.triu_indices(E.shape[0], 1)] -= np.amax(E)
        E = -E  # make distance matrix positive

        # start clustering, half max E for distance threshold!
        print('Round {}/{}, start clustering.'.format(rounds+1, n_splits), end=' ')
        clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='precomputed',
                                             linkage='average', ## minimum
                                             distance_threshold=np.amax(E)/2
                                             ).fit(E)
        train_labels = clustering.labels_
        n_clusters = clustering.n_clusters_
        print('{} clusters exist'.format(n_clusters))

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
        memory['clustering'].append(clustering)
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
                print('{} clusters completed, {:.2f} seconds spent, train accuracy{:.2f}, test accuracy{:.2f}'
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


def my_baseline(df, n_splits=10, seq_len=12, init_len=1):
    """
    This function does not implement any agglomerative clustering.
    baseline to see the performance of agglomerative clustering

    :param df: the dataframe, must contain Commands and Commands Length columns
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
    # memory['clustering'] = []
    memory['init_commands_list'] = []
    memory['train_sessions_labels'] = []
    memory['test_sessions_labels'] = []

    for train_idx, test_idx in kf.split(new_df):
        start_time = time.time()

        # initialise train and test sets
        X_train, X_test = new_df.iloc[train_idx, :], new_df.iloc[test_idx, :]
        sessions_list_train, _ = extract(X_train)
        sessions_list_test, _ = extract(X_test)

        # find rank matrix A and distance matrix E
        # A, _, init_commands_list = build_rankmat(
        #     seq_len, sessions_list_train, init_len)

        sorted_init_commands = sort_idx(find_initial_commands_idx(
            init_len, sessions_list_train))  # add
        init_commands_list = [commandsidx[0]
                              for commandsidx in sorted_init_commands]  # add

        # E = similarity_mat(A, np.ones(A.shape)*alpha)   # alpha
        # E[np.triu_indices(E.shape[0], 1)] -= np.amax(E)
        # E = -E  # make distance matrix positive

        # start clustering, half max E for distance threshold!
        # print('Round {}/{}, start clustering.'.format(rounds+1, n_splits), end=' ')
        # clustering = AgglomerativeClustering(n_clusters=None,
        #                                      affinity='precomputed',
        #                                      linkage='average',
        #                                      distance_threshold=np.amax(E)/2
        #                                      ).fit(E)
        # train_labels = clustering.labels_
        # n_clusters = clustering.n_clusters_
        # print('{} clusters exist'.format(n_clusters))
        print('Round {}/{}'.format(rounds+1, n_splits))
        train_labels = [i for i in range(
            len(init_commands_list)-1, -1, -1)]  # add
        n_clusters = len(init_commands_list)  # add

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
        # memory['clustering'].append(clustering)
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

            if label//400 == label/400 and total_test != 0 and total_train != 0:  # change
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
    my_baseline(msdat)
