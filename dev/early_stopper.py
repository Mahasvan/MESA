class EarlyStopping:
    def __init__(self):
        self.history = []

    def record_loss(self, value):
        """
        Records the loss for one epoch. this is added to the internal history, to decide early stopping
        :param value:
        :return: None
        """
        self.history.append(value)

    def _get_derivative(self):
        d = []
        for i in range(1, len(self.history)):
            d.append(self.history[i] - self.history[i-1])
        return d

    def stop(self, window=3):
        """
        :param window: the number of past loss values to consider
        :return: True if early stopping, False otherwise
        """
        if len(self.history)  < window+1: return False
        derivative = self._get_derivative()[-window:]
        # print(derivative)
        # if for three continuous epochs, the loss keeps increasing, then stop the training.
        if all([x > 0 for x in derivative]): return True
        return False
