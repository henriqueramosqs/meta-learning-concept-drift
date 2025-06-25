from abc import ABC,abstractmethod

class MfeExtractor(ABC):
    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def evaluate()->dict:
        pass