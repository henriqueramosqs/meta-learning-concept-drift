from abc import ABC,abstractmethod

class MfeExtractor(ABC):

    """
    Abstract Class for Meta-Feature Extractors.
    
    Any concrete class derived from MfeExtractor must implement the 'fit' and 'evaluate' methods.
    """
    
    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def evaluate()->dict:
        pass