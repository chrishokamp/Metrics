import numpy as np

def weighted_apk(actual, predicted, weight_func=None, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items, where the score at each position is weighted by some function

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    weight_func: function
                A function which takes an item as input and returns a weight
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            i_score = num_hits / (i+1.0)
            if weight_func is not None:
                weight = weight_func(p)
                i_score = weight * i_score
            score += i_score

    if not actual:
        return 1.0

    return score / min(len(actual), k)

def weighted_mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([weighted_apk(a,p,k) for a,p in zip(actual, predicted)])
