
from secml.core import CCreator
from abc import ABCMeta, abstractmethod, abstractproperty


class CStep(CCreator):

    '''
    Abstract class to implement type of step used in optimization problems
    '''
    __metaclass__ = ABCMeta
    __super__ = 'CLearningRate'

    def __init__(self, initial_step_value, **kwargs):
        '''
        Sets the initial value of step
        '''
        self._initial_step = initial_step_value
    @property
    def initial_step(self):
        return self._initial_step
    @initial_step.setter
    def initial_step(self,value):
        self._initial_step=value
    
    
    @abstractmethod
    def get_actual_step(self, iter):
        '''
        Returns the step value for i-th iteration
        '''
        raise NotImplementedError()
