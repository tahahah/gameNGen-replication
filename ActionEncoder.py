from typing import Union

import numpy as np


class ActionEncoder():
    """
    A class used to encode and decode actions into one-hot representations.

    Attributes
    ----------
    dir2mv : dict
        A dictionary mapping action strings to their corresponding one-hot indices.

    Methods
    -------
    encode(action: str)
        Encodes an action string into a one-hot numpy array.
    
    decode(one_hot_encoding)
        Decodes a one-hot numpy array back into an action string.
    
    validate_action(action: str)
        Validates if a given action string is valid.
    
    get_possible_actions()
        Returns a list of all possible action strings.
    """
    def __init__(self):
        """
        Initializes the ActionEncoder with a predefined mapping of actions to one-hot indices.
        """
        self.__dir2mv = {
            "left"  :   (0, 0),
            "up"    :   (1, 1),
            "right" :   (2, 2),
            "down"  :   (3, 3),
        }

    def __call__(self, action: str) -> np.ndarray:
        """
        Allows the object to be called as a function to encode an action string into a one-hot numpy array.

        Parameters
        ----------
        action : str
            The action string to encode.

        Returns
        -------
        numpy.ndarray
            A one-hot encoded numpy array representing the action.
        """
        return self.encode(action)

    def encode(self, action: str) -> np.ndarray:
        """
        Encodes an action string into a one-hot numpy array.

        Parameters
        ----------
        action : str
            The action string to encode.

        Returns
        -------
        numpy.ndarray
            A one-hot encoded numpy array representing the action.
        """
        one_hot_encoding = np.zeros((4, 4))
        one_hot_encoding[self.__dir2mv[action.lower()]] = 1
        return one_hot_encoding

    def decode(self, one_hot_encoding: Union[np.ndarray, list]) -> str:
        """
        Decodes a one-hot numpy array back into an action string.

        Parameters
        ----------
        one_hot_encoding : numpy.ndarray
            The one-hot encoded numpy array to decode.

        Returns
        -------
        str or None
            The decoded action string, or None if the encoding is invalid.
        """
        one_hot_encoding = np.array(one_hot_encoding)
        index = np.argwhere(one_hot_encoding == 1)
        for action, idx in self.__dir2mv.items():
            if tuple(index[0]) == idx:
                return action
        return None

    def validate_action(self, action: str) -> bool:
        """
        Validates if a given action string is valid.

        Parameters
        ----------
        action : str
            The action string to validate.

        Returns
        -------
        bool
            True if the action is valid, False otherwise.
        """
        return action.lower() in self.__dir2mv

    def get_possible_actions(self) -> list:
        """
        Returns a list of all possible action strings.

        Returns
        -------
        list
            A list of all possible action strings.
        """
        return list(self.__dir2mv.keys())