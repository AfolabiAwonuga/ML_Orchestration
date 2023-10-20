from typing import Any

import numpy as np
import xgboost as xgb


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
    ) -> Any:
        '''
        Preprocesses input data and converts it into an xgb.DMatrix for making predictions.

        Parameters:
            body (dict): A dictionary containing input data.
            state (dict): A dictionary containing state information.
            collect_custom_statistics_fn (function, optional): A function for collecting custom statistics (default is None).

        Returns:
            xgb.DMatrix: The preprocessed data in xgb.DMatrix format.
        '''

        return xgb.DMatrix(
            [
                [
                    body.get('months_as_member', None), 
                    body.get('weight', None), 
                    body.get('days_before', None), 
                    body.get('day_of_week', None), 
                    body.get('time', None)
                ]
            ],
            feature_names=list(body.keys())
        )

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
