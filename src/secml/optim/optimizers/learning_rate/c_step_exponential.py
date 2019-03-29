
from secml.optim.optimizers.learning_rate import CStep


class CStepExponential(CStep):
    """Exponential step.

    For every iteration, the step is compute as:
            step^it
    in this case we have as constraint abs(step) < 1.

    Attributes
    ----------
    class_type : 'exp'

    """
    __class_type = 'exp'

    def __init__(self, initial_step_value):
        '''
        Sets the initial value of step
        '''
        if abs(initial_step_value) >= 1:
            raise ValueError(
                'Step size too big. Its abs must be smaller than 1')
        super(CStepExponential, self).__init__(
            initial_step_value=initial_step_value)

    def get_actual_step(self, iter):
        return self.initial_step ** iter + 1
