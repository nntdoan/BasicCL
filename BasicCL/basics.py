import pandas as pd
import numpy as np
from itertools import product

'''
Based on some parts of the CL course, Tilburg University (2021) taught by Dr. Cassani.
The dynamic programming approach for the algorithms as follows is rather standard, 
minor differences in datatype might exist. 

These implementations are meant for "toy problems": 
for us to check our "pen-and-paper" solutions.
No guarantee entails.
'''


# =====================================================================
def get_min_edit_distance_table(strA, strB, count_substitution=1):
    """
    param str1, str2: the two string to calculate the edit distance.
    substitution can be counted once or twice (= deletion + insertion).
    return: 
    1. a table (np.array) filled with minimum edit distance up to each cell, 
    2. a row_trace_index (np.array): contain the row index of the previous cell 
    based on which the distance in the current cell is calculated.
    3. a col_trace_index (np.array): contain the col index of the previous cell 
    based on which the distance in the current cell is calculated.
    """
    
    # Initiate a table to store distance
    # each cell contains a tuple (a, b, c), a is the minimum distance, 
    # (b, c) is the index of the cell, based on which we calculate the current cell.
    D = np.zeros((len(strA)+1, len(strB)+1), dtype=int) 
    D[0, :] = [i for i in range(len(strB) + 1)]
    D[:, 0] = [i for i in range(len(strA) + 1)]
    
    row_trace_index = np.zeros((len(strA)+1, len(strB)+1), dtype=int)
    col_trace_index = np.zeros((len(strA)+1, len(strB)+1), dtype=int)
    
    for i in range(1, len(strA)+1):
        for j in range(1, len(strB)+1):
            # the character is different
            if strA[i-1] != strB[j-1]: 
                min_of_three_prev_cells = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
                
                if  min_of_three_prev_cells == D[i-1, j-1]:
                    D[i, j] = count_substitution + D[i-1, j-1] # subsitution
                    row_trace_index[i, j] = i-1
                    col_trace_index[i, j] = j-1
                    
                elif min_of_three_prev_cells == D[i, j-1]:
                    D[i, j] = 1 + D[i, j-1]
                    row_trace_index[i, j] = i
                    col_trace_index[i, j] = j-1
                    
                else:
                    D[i, j] = 1 + D[i-1, j]
                    row_trace_index[i, j] = i-1
                    col_trace_index[i, j] = j
                    
            # the character is the same
            else:
                D[i, j] = D[i-1, j-1]
                row_trace_index[i, j] = i-1
                col_trace_index[i, j] = j-1
                   
    return D, row_trace_index, col_trace_index



def get_min_edit_distance(strA, strB, count_substitution=1):
    """
    param: strA and strB are the two strings to calculate the minimum edit distance
    return: the minimum distance between the two string
    """
    D, _, _ = get_min_edit_distance_table(strA, strB, count_substitution)
    return D[len(strA), len(strB)] # bottom right-most cell in the minimum distance table



def trace_back_min_edit_distance(D, row_trace_index, col_trace_index):
    """
    Given minimum edit distance table D, and the trace indice.
    Return: the list of indices in the D table that leads to 
    the cell that holds the global minimum edit distance.
    """
    
    i, j = D.shape[0]-1, D.shape[1]-1
    
    # initiate the backward trace
    trace = [(i, j)]
    
    while i and j > 0:
        i = row_trace_index[i, j]
        j = col_trace_index[i, j]
        trace_index = (i, j)
        trace.append(trace_index)
    
    return [t for t in trace[::-1]]


def get_masked_distance_table(D, row_trace_i, col_trace_i, strA=None, strB=None):
    """
    return: pd.DataFrame. Highlight the trace that led to the cell 
    containing the minimum edit distance in the minimum distance table D. 
    """
    trace = trace_back_min_edit_distance(D, row_trace_i, col_trace_i)

    masked_D = np.zeros(D.shape).astype(str)
    masked_D[:] = "-"

    for i, j in trace:
        masked_D[i, j] = D[i, j]
    
    col_names = [i for i in "#"+strB if i!=" "] if strB else [i for i in range(len(strB))]
    row_names = [i for i in "#"+strA if i!=" "] if strA else [i for i in range(len(strA))]
    
    masked_D = pd.DataFrame(masked_D, columns=col_names, index=row_names)
    
    return masked_D


# =====================================================================
def get_cky_table(string, grammar):
    """
    string: the string to be parsed
    grammar: a list, where the value at index mod 3 starting from 0 is the LHS,
             the value at index mod 3 starting from 1, and 2 is the RHS.
    return: CKY parse table.
    """
    table = np.zeros((len(string)+1, len(string)+1)).astype(str)
    table[:] = "-"

    # First, loop through the string
    for j in range(1,len(string)+1):
        # Then, loop through all RHS rules that string[j] belong to the grammar
        for r in range(len(grammar)):
            if r%3!=0 and grammar[r]==string[j-1]:
                # then fill in the cell, otherwise leave it to be 0
                table[j-1, j] = table[j-1, j] + "," + grammar[r//3*3] if table[j-1, j]!="-" else grammar[r//3*3]
                
        # if there is empty cell above current cell?
        for i in reversed(range(0, j-1)): 
            
            for k in range(i+1, j): # fill the cells above current cell                
                
                if (table[i, k] != "-") and (table[k, j] != "-"):
                    
                    for combi in product(table[i, k].split(","), table[k, j].split(",")):
                        toCheck = "".join(combi)
                        # Then, loop through all RHS rules that string[j] belong to the grammar
                        for r2 in range(len(grammar)):
                            if r2%3!=0 and grammar[r2]==toCheck:
                                # then fill in the cell, otherwise leave it to be 0
                                table[i, j] = table[i, j] + "," + grammar[r2//3*3] if table[i, j]!="-" else grammar[r2//3*3]
    return table



def ckyParser(string, grammar):
    return get_cky_table(string, grammar)[0, len(string)]


def is_well_formed(string, grammar):
    return True if "S" in ckyParser(string, grammar) else False



# =====================================================================
def get_viberti_trellis(state_transition, emission, sentence):
    """
    param state_transition: states (np.array), 
    param emission: observations/events/words (pd.DataFrame).
    The columns keys in emission matrix must contain all the UNIQUE word TYPES appear in sentence

    return: a viberti trellis (np.array) of a Hiden Markov Model given the parameters and a trace (list).
    Each cell in the trellis contains the posterior probability of finding each tag given the current word.

    Probability in the viberti table depends on the emission probability and the local transition probability,
    ... the emission probability enforces the output dependence assumption of the model,
    ... the local transition enforces the Markov assumption.
    In other words, at each word wt for t in T, we compute the posteriors,
    ... considering all the emission probabilities and the local history,
    ... then take the cell with highest posterior and keep track of which transition hitherto.

    Example:

    # tags = ["Det", "Adj", "Noun", "Verb"]
    states = np.array([[0.0, 0.2, 0.8, 0.0, 0.0],
                       [0.0, 0.3, 0.6, 0.0, 0.1],
                       [0.0, 0.0, 0.0, 0.5, 0.5],
                       [0.5, 0.1, 0.2, 0.0, 0.2],
                       [0.5, 0.2, 0.3, 0.0, 0.0]])


    # unique_types = ["the", "dog", "chases", "cat", "fat"]
    # observations o
    o = pd.DataFrame(np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0],
                               [0.0, 0.5, 0.0, 0.4, 0.1],
                               [0.0, 0.1, 0.8, 0.1, 0.0]]), columns = unique_types)

    sent = "The dog chases the fat cat"
    trellis, trace = get_viberti_trellis(state_transition=q, emission=o, sentence= sent)

    """
    tokens = [w.lower() for w in sentence.split(" ")]
    # initiate matrix, +2 for bos and eos in the column
    vtrellis = np.zeros((emission.shape[0], len(tokens)+2))  
    # First, add the initial distribution for bos and eos (last row and last column in state_transition matrix)

    vtrellis[:, 0] = state_transition[-1, :-1] # bos
    vtrellis[:, -1] = state_transition[:-1, -1] # eos
    vtrellis[:, 1] = vtrellis[:, 0] * emission[tokens[0]].values # first word in the sentence

    # initivate variable to store the trace
    trace = []
    
    # Below we fill in the vtrellis, multiple the prior in the previous columns with 
    # the likelihood of the word in the emission matrix
    # for each column, recording the trace row index of the prior used to calculate this column
    for j in range(2, vtrellis.shape[1]-1): # excluding bos, first word and eos columns
        # Prior is probability of the tag for the current words given the probability of the previous tag
        prev_col = vtrellis[:, j-1].reshape(vtrellis[:, j-1].shape[0], 1)
        all_prior = prev_col * state_transition[:-1, :-1] # excluding prob of bos and eos
        most_probable_prior_index = np.unravel_index(np.argmax(all_prior, axis=None), all_prior.shape)
        prior = all_prior[most_probable_prior_index]
        trace.append(most_probable_prior_index[0])

        # Likelihood is probability of the word given the tag
        likelihood = emission[tokens[j-1]].values
        vtrellis[:, j] = prior * likelihood

    # Finally, the tags of the last words is the most probable tags give that word (likelihood)
    trace.append(np.argmax(emission[tokens[-1]].values))
    
    return vtrellis, trace


if __name__ == "__main___":
    # LHS = S, A, B, C.
    grammar_input = ["S", "AB", "BC", "A", "BA", "a", "B", "CC", "b", "C", "AB", "a"]
    string_input = "abba"

    # print(ckyParser(string_input, grammar_input))

