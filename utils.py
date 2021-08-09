import numpy as np
import scipy.stats as ss
from scipy.stats import norm
# from sklearn.decomposition import TruncatedSVD
# from numpy.linalg import svd
# from scipy.sparse.linalg import svds
from scipy.special import gammaln


def extract(df):
    """
    This function extract list of sessions and commands
    """
    sessions_list = []  # list of sessions
    commands_list = []  # list of all commands
    for session in df['Commands']:
        sessions_list += [session]
        commands_list += session

    return sessions_list, commands_list


def make_seq_list(n, sessions_list):
    """
    create list of sequences of length n in a session
    """
    seq_list = []
    for seq in sessions_list:
        if len(seq) >= n:
            for m in range(len(seq)-n+1):
                seq_list += [tuple(seq[m:m+n])]

    return seq_list


def make_seq_counts(n, sessions_list):
    """
    create dictionary with records counts of sequences of length n
    """
    seq_counts = dict()  # dictionary, count of sequences
    seq_list = make_seq_list(n, sessions_list)
    for i in seq_list:
        seq_counts[i] = seq_counts.get(i, 0)+1

    return seq_counts


def sort_counts(counts_dict):
    """
    Sort dictionary of counts from high to low, return a list of tuples
    """
    return sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)


def find_initial_commands_idx(n, sessions_list):
    """
    find first n commands with respective index in the sessions_list
    """
    initial_commands_idx = dict()
    for i in range(len(sessions_list)):
        initial_commands_idx[tuple(sessions_list[i][:n])] = \
            initial_commands_idx.get(tuple(sessions_list[i][:n]), [])+[i]

    return initial_commands_idx


def sort_idx(idx_dic):
    """
    sort dictionary by length of index from high to low
    """

    return sorted(idx_dic.items(), key=lambda item: len(item[1]), reverse=True)


def rank_seq(n, sessions_list):
    sorted_seq_counts = sort_counts(make_seq_counts(n, sessions_list))
    seq_list = [i[0] for i in sorted_seq_counts]
    # dense?
    rank_list = ss.rankdata(
        [-i[1] for i in sorted_seq_counts], method='dense').astype('int')-1
    seq_rank_list = [(seq_list[i], rank_list[i]) for i in range(len(seq_list))]

    return seq_rank_list


def rank_session(n, sessions_list, seq_rank_list=None):
    """
    rank session, return list of rank of sessions
    """
    session_seq_list = [make_seq_list(n, [sessions_list[i]])
                        for i in range(len(sessions_list))]
    if seq_rank_list == None:
        seq_rank_list = rank_seq(n, sessions_list)
    session_rank_list = []
    for seqs in session_seq_list:
        for seq, rank in seq_rank_list:
            if seq in seqs:
                session_rank_list.append(rank)
                break

    return session_rank_list, seq_rank_list


def build_rankmat(seq_len, sessions_list, init_len, session_rank_list=None, seq_rank_list=None):
    """
    build rank matrix A
    init_commands_list: list of initial commands sorted by appearance, correspond to rows of A
    """
    if (session_rank_list == None) or (seq_rank_list == None):
        session_rank_list, seq_rank_list = rank_session(
            seq_len, sessions_list, seq_rank_list)
    if len(session_rank_list) != len(sessions_list):
        raise ValueError('Input session_rank_list wrong.')

    sorted_init_commands = sort_idx(
        find_initial_commands_idx(init_len, sessions_list))

    init_commands_list = []
    A = np.zeros((len(sorted_init_commands), max(
        session_rank_list)+1), dtype='int')
    for i in range(len(sorted_init_commands)):
        commands, idxs = sorted_init_commands[i]
        init_commands_list.append(commands)
        i_rank = [session_rank_list[idx] for idx in idxs]
        for j in i_rank:
            A[i, j] += 1

    return A, session_rank_list, init_commands_list, seq_rank_list


def find_session_rank(sessions_list, seq_rank_list):
    """
    find the ranks of sessions
    :param sessions_list: list of sessions
    :param seq_rank_list: list of rank of sequences

    :return session_rank_list: if unseen, rank it to unseen_rank
    """
    seq_len = len(seq_rank_list[0][0])
    unseen_rank = seq_rank_list[-1][1]+1
    session_seq_list = [make_seq_list(
        seq_len, [sessions_list[i]]) for i in range(len(sessions_list))]
    session_rank_list = [-1 for i in session_seq_list]
    for idx, seqs in enumerate(session_seq_list):
        for seq, rank in seq_rank_list:
            if seq in seqs:
                session_rank_list[idx] = rank
                break

    session_rank_list = [i if i != -
                         1 else unseen_rank for i in session_rank_list]

    return session_rank_list


#############################################################

# from zhu.py
def zhu(d):
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1, p-1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(
            ((q-1) * (np.std(d[:q]) ** 2) + (p-q-1) * (np.std(d[q:]) ** 2)) / (p-2))
        profile_likelihood[q] = norm.logpdf(d[:q], loc=mu1, scale=sd).sum(
        ) + norm.logpdf(d[q:], loc=mu2, scale=sd).sum()
    return profile_likelihood[1:p-1], np.argmax(profile_likelihood[1:p-1])+1


def iterate_zhu(d, x=3):
    results = np.zeros(x, dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x-1):
        results[i+1] = results[i] + zhu(d[results[i]:])[1]
    return results


# def spec_decomp(A, x=2):
#     """
#     spectual decomposition of A
#     """
#     # Full SVD (might not be feasible for VERY large matrices) & takes only dense objects (np.array) as input
#     U, D, V = svd(A)
#     # Use zhu on D to select dimension
#     d = iterate_zhu(D, x)[-1]
#     # Dimensionality reduction
#     X = np.matmul(U[:, :d], np.diag(np.sqrt(D[:d])))

    # ## Approximate method (good for very large AND/OR sparse matrices)
    # U, D, V = svds(A, k=200)
    # ## Use zhu on SORTED D to select dimension i.e. zhu(D[::-1])
    # ## Dimensionality reduction
    # X2 = np.matmul(U[:,::-1][:,:d], np.diag(np.sqrt(D[::-1][:d])))

    # ## Identical way for calculating X2 using the sklearn function
    # tsvd = TruncatedSVD(n_components=200, algorithm='arpack')
    # tsvd.fit(A)
    # D = tsvd.singular_values_
    # U = np.matmul(tsvd.transform(A), np.diag(1 / D))
    # X2_sklearn = np.matmul(U[:,:d], np.diag(np.sqrt(D[:d])))

#######################################################################


def multinomial_dirichlet_log_density(counts, alphas, alpha_dot=None):
    n_dot = sum(counts)
    if alpha_dot is None:
        alpha_dot = sum(alphas)
    lhd_terms = np.array([0.0 if c_k == 0 else gammaln(
        a_k+c_k)-gammaln(a_k) for (a_k, c_k) in zip(alphas, counts)])
    const = -n_dot*np.log(n_dot) if (alpha_dot ==
                                     0) else gammaln(alpha_dot)-gammaln(alpha_dot+n_dot)

    return(sum(lhd_terms)+const)


def multinomial_dirichlet_log_density2(counts, alphas, alpha_dot=None, return_alpha_star=False):
    # multi-dimensional version, alphas same dimension as counts
    n_dot = np.sum(counts, axis=-1)
    if alpha_dot is None:
        alpha_dot = np.sum(alphas, axis=-1)
    lhd_terms = np.zeros(counts.shape, dtype='float64')
    lhd_terms[counts != 0] = \
        gammaln(alphas[counts != 0]+counts[counts != 0]) - \
        gammaln(alphas[counts != 0])

    const = np.zeros(n_dot.shape)
    const[alpha_dot == 0] = -n_dot[alpha_dot == 0] * \
        np.log(n_dot[alpha_dot == 0])
    const[alpha_dot != 0] = gammaln(alpha_dot[alpha_dot != 0]) - \
        gammaln(alpha_dot[alpha_dot != 0]+n_dot[alpha_dot != 0])

    if return_alpha_star:
        return np.sum(lhd_terms, axis=-1) + const, alphas+counts

    return np.sum(lhd_terms, axis=-1) + const


def similarity_mat(A, alphas):
    """
    find the similarity matrix E
    """
    m = A.shape[0]
    E = np.zeros((m, m), dtype='float64')
    log_density_vec = multinomial_dirichlet_log_density2(A, alphas)
    for i in range(m):
        E[i, i+1:] = multinomial_dirichlet_log_density2(A[i]+A[i+1:], alphas[i+1:]) -\
            log_density_vec[i] -\
            log_density_vec[i+1:]
        E[i, i+1:] /= np.sum(A[i]+A[i+1:], axis=1)

    return E


# def similarity_mat(A, alphas):
#     m = A.shape[0]
#     E = np.zeros((m, m), dtype='float64')
#     iu = np.triu_indices(m, 1)
#     # inidividual
#     log_density_vec = multinomial_dirichlet_log_density2(A, alphas)
#     # combined
#     Acombine = A[:,np.newaxis,:]+A[np.newaxis,:,:]
#     log_density_mat = \
#     multinomial_dirichlet_log_density2(Acombine, (alphas[:,np.newaxis,:]+alphas[np.newaxis,:,:])/2)
#     E[iu] = (log_density_mat-log_density_vec.reshape((-1,1))-log_density_vec)[iu]
#     E[iu] /= np.sum(Acombine, axis=-1)[iu]

#     return E

def find_sessions_label(sessions_list, init_commands_list, labels):
    """
    This function can find the labels of sessions given the list of
    initial commands and corresponding labels

    :param sessions_list: list of sessions to find labels
    :param init_commands_list: list of initial commands (tuples)
    :param labels: list of initial commands corresponding labels

    :return sessions_labels: list of session labels
    """
    n = len(init_commands_list[0])
    n_clusters = len(set(labels))
    init_sessions_list = [tuple(session[:n]) for session in sessions_list]
    # if no cluster found, make a new cluster label
    sessions_labels = [labels[init_commands_list.index(commands)] if commands in init_commands_list
                       else n_clusters for commands in init_sessions_list]

    return sessions_labels


def predict_next_command(max_n, sessions_list, X, most_common_command):
    """
    :param max_n: max number of commands we look back
    :param sessions_list: list of sessions selected
    :param X: numpy array of sequence of commands (no session)
    :param most_common_command: tuple of commands

    :return y: numpy array of predicted commands

    """
    # let max_n be reasonable
    if max_n > len(X[0]):
        max_n = len(X[0])
    y = np.zeros(len(X), dtype='object')  # initialisation
    idx = np.arange(len(X))
    # X_temp = copy.deepcopy(X)
    X_temp = X

    # while loop, the smallest max_n can get is 1
    while max_n > 0:
        seq_counts = dict()  # dictionary, X and next command
        # create database
        seq_list = make_seq_list(max_n+1, sessions_list)
        for i in seq_list:
            seq_counts[i[:-1]] = seq_counts.get(i[:-1], dict())
            seq_counts[i[:-1]][i[-1]] = seq_counts[i[:-1]].get(i[-1], 0)+1
        # truncate X to fit number of commands we look back
        X_trunc = [x[-max_n:] for x in X_temp]
        # if find exact sequence of commands, record the most frequenct next command
        # if not, record 0
        y[idx] = [seq_counts.get(tuple(x), 0) for x in X_trunc]
        y[idx] = [sort_counts(dic)[0][0] if dic != 0 else 0 for dic in y[idx]]

        idx = np.arange(len(y))[y == 0]
        if len(idx) == 0:  # no need for further investigation
            break
        max_n = max_n//2    # halve
        X_temp = X[idx]

    # let y==0 predict most common command
    y[y == 0] = most_common_command

    return y
