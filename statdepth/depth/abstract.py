from abc import ABC, abstractmethod

class FunctionalDepth(ABC):
    '''Abstract class for all Functional Depth classes. This should be used as the superclass in any future work.'''

    # Abstract methods
    # These must be implemented by subclasses
    @abstractmethod
    def ordered(self, ascending=False):
        raise NotImplementedError

    @abstractmethod
    def deepest(self, n=1):
        raise NotImplementedError

    @abstractmethod
    def outlying(self, n=1):
        raise NotImplementedError

    # Aliases for some abstract methods above
    def sorted(self, ascending=False):
        return self.ordered(ascending=ascending)

    def median(self):
        return self.deepest(n=1)

