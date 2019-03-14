
from secml.optim.learning_rate import CStep


class CStepConstant(CStep):
    """Constant step.

    Attributes
    ----------
    class_type : 'constant'

    """
    __class_type = 'constant'

    def __init__(self, initial_step_value):
        '''
        Sets the initial value of step
        '''
        super(CStepConstant, self).__init__(
            initial_step_value=initial_step_value)

    def get_actual_step(self, iter):
        return self._initial_step
