import inspect

__all__ = ['BaseComponent']


class BaseComponent:
	"""Base class for all paltas models (e.g. SourceBase) """

	@classmethod
	def init_from_sample(cls, sample):
		return cls(**cls._sample_to_kwargs(sample))

	def update_from_sample(self, sample):
		self.update_parameters(**self._sample_to_kwargs(sample))

	@classmethod
	def _sample_to_kwargs(cls, sample):
		if not hasattr(cls, '__init_kwargs'):
			# Find out which arguments __init__ takes
			sig = inspect.signature(cls.__init__)
			cls.__init_kwargs = [p for p in sig.parameters.keys() if p != 'self']
		return {p: sample[p] for p in cls.__init_kwargs}
