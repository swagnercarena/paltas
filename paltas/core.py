
class BaseComponent:
	"""Base class for all paltas models (e.g. SourceBase) """

	init_kwargs = tuple()
	required_parameters = tuple()

	@property
	def main_param_dict_name(self):
		return self.init_kwargs[0]

	def __init__(self, *args, **kwargs):
		# Fold in any positional arguments
		kwargs.update(dict(zip(self.init_kwargs, args)))

		for dict_name in self.init_kwargs:
			setattr(self, dict_name, dict())
		self.update_parameters(**kwargs, _first_time=True)

		# Check that all the required parameters are present
		self.check_parameterization(self.required_parameters)

	def update_parameters(self, *args, _first_time=False, **kwargs):
		"""Updated the parameters

		Args:
			Dictionary for each argument named in _init_kwargs

		Notes:
			Use this function to update parameter values instead of
			starting a new class.
		"""
		# Fold in any positional arguments
		kwargs.update(dict(zip(self.init_kwargs, args)))

		import paltas
		for dict_name in self.init_kwargs:
			if kwargs.get(dict_name) is None:
				if _first_time:
					raise ValueError(f"Must provide {dict_name} to init {self}")
				continue
			if dict_name == 'cosmology_parameters':
				self.cosmo = paltas.Utils.cosmology_utils.get_cosmology(
					kwargs['cosmology_parameters'])
			else:
				getattr(self, dict_name).update(kwargs[dict_name])

	def check_parameterization(self,required_params):
		""" Check that all the required parameters are present
		"""
		present_params = set(getattr(self, self.main_param_dict_name).keys())
		required_params = set(required_params)
		missing_params = required_params - present_params
		if missing_params:
			raise ValueError(
				f"Missing parameters {missing_params}"
			 	f" in {self.main_param_dict_name}.")

	@classmethod
	def init_from_sample(cls, sample):
		return cls(**cls._sample_to_kwargs(sample))

	def update_from_sample(self, sample):
		self.update_parameters(**self._sample_to_kwargs(sample))

	@classmethod
	def _sample_to_kwargs(cls, sample):
		return {p: sample[p] for p in cls.init_kwargs}
