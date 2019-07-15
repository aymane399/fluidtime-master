class Model(object):
    def __init__(self):
        self.wordindex = {}
        pass

    def get_loss(self, targets, contexts, times, labels):
        raise NotImplementedError

