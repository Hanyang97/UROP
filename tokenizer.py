from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary


def clean_commands(dat, no_below=15, no_above=1.1):
    """
    This function 
    1. splits multiple commands in the same line
    2. tokenize the commands
    3. replace rare commands by rarecommand

    :param dat: dataset
    :param no_below: Keep tokens which are contained in at least no_below documents.
    :param no_above: Keep tokens which are contained in no more than no_above documents 
    (fraction of total corpus size, not an absolute number).

    :return sessins_token_list: tokenized list of sessions of commands
    :return dictionary: dictionary generated
    """
    # for commands splitted by ;
    sessions = []
    for session in dat['Commands']:
        sessions.append([])
        for command in session:
            sessions[-1] += command.split('; ')
    # tokenizer
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_\.\-\*]+')
    sessions_list = []
    commands_list = []
    for session in sessions:
        sessions_list.append([])
        commands_list.append([])
        for command in session:
            command_token = tokenizer.tokenize(command)
            sessions_list[-1] += [command_token]
            commands_list[-1] += command_token
    dictionary = Dictionary(commands_list)
    dictionary.filter_extremes(no_below, no_above)
    # repleace rare commands by rarecommand
    dictionary.id2token[-1] = 'rarecommand'
    dictionary.token2id['rarecommand'] = -1
    sessions_token_list = []
    for session in sessions_list:
        sessions_token_list.append([])
        commands_token_list = []
        for command in session:
            idxs = dictionary.doc2idx(command)
            commands_token_list.append(
                ' '.join([dictionary[idx] for idx in idxs]))
        sessions_token_list[-1] += commands_token_list

    return sessions_token_list, dictionary
