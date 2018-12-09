
from secml.optimization.learning_rate import CStep


class CStepInvscaling(CStep):

    '''
    Implements invscaling step.
    For every iteration, the step is compute as
            init_step/(t^power_t)
    init_step=initial value of the step
    t: t-th iteration
    '''

    __class_type = 'CStepInvscaling'

    def __init__(self, initial_step_value, power_t=1):
        '''
        Sets the initial value of step
        '''
        super(CStepInvscaling, self).__init__(
            initial_step_value=initial_step_value)
        self.power_t = power_t

    def get_actual_step(self, iter):
        return float(self.initial_step) / (iter + 1 ** self.power_t)
