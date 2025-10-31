class EarlyStopping:
    """
    This class is designed to record loss values, exit the training loop if the validation loss does not reduce for N continuous epochs.
    """
    def __init__(self):
        self.history = []


    def _get_derivative(self, step=1):
        d = []
        for i in range(step, len(self.history)):
            d.append(self.history[i] - self.history[i-step])
        return d

    def record(self, value, window=3, step=1):
        """
        :param value: the loss vaue to be recorded
        :param window: the number of past loss values to consider
        :return: True if early stopping, False otherwise
        """
        self.history.append(value)
        if len(self.history)  < window+1: return False
        derivative = self._get_derivative(step)[-window:]
        # print(derivative)
        # if for three continuous epochs, the loss keeps increasing, then stop the training.
        if all([x > 0 for x in derivative]): return True
        return False
