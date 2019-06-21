
def select_score(type_id):

    if type_id == 'accuracy':
        return accuracy

    raise ValueError("Unknown score %s " % type_id)


def accuracy(pred, target):
    return 1 if pred.round() == target.round() else 0
