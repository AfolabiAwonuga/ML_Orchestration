from typing import Any


class Preprocess(object):
    '''
    Preprocesses input data for making predictions.

    Attributes:
        None
    '''

    def __init__(self):
        pass

    def preprocess(
            self, 
            body: dict, 
            # state: dict, 
            # collect_custom_statistics_fn=None
    ) -> list:
        '''
        Preprocesses input data and returns a list of lists containing feature values.

        Parameters:
            body (dict): A dictionary containing input data.
            state (dict): A dictionary containing state information.
            collect_custom_statistics_fn (function, optional): A function for collecting custom statistics (default is None).

        Returns:
            list: A list of lists containing feature values. Each inner list represents a sample.

        '''
        return [[
            body.get('months_as_member', None), 
            body.get('weight', None), 
            body.get('days_before', None), 
            body.get('day_of_week', None), 
            body.get('time', None)
        ],]
    
    def postprocess(
            self,
            data: Any, 
            # state: dict, 
            # collect_custom_statistics_fn=None
    ) -> dict:
        '''
        Postprocesses the prediction result and returns a descriptive message.

        Parameters:
            data (Any): The prediction result.
            state (dict): A dictionary containing state information.
            collect_custom_statistics_fn (function, optional): A function for collecting custom statistics (default is None).

        Returns:
            str: A descriptive message based on the prediction result.
        '''

        if data >= 0.5:
            return 'Member will attend class'
        else:
            return 'Member will not attend class'

