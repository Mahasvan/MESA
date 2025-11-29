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
        This function records the latest loss value, and returns True if the training loop should terminate.

        If `step` is 1, `Δ[n] = history[n] - history[n-1]`.

        If step is 2, `Δ[n] = history[n] - history[n-2]`.

        `step` is set to 2 for dynamic batch training, to separately calculate loss derivatives for normal and dynamic epochs.

        :param value: the loss value to be recorded
        :param window: the number of past loss values to consider
        :param step: the distance from which delta is calculated.
        :return: True if early stopping, False otherwise
        """
        self.history.append(value)
        if len(self.history)  < window+1: return False
        derivative = self._get_derivative(step)[-window:]
        # print(derivative)
        # if for {window} continuous epochs, the loss keeps increasing, then stop the training.
        if all([x > 0 for x in derivative]): return True
        return False
