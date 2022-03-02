Config Files
------------

config_train.py
	Configuration file used to make the training set for the paper.
config_val.py
	Configuration file used to make the validation set for the paper. All parameters are the same as the training set but the held-out sources are being used.
config_test_1.py -- config_test_20.py
	Configuration files used to generate test sets 1 through 20 for the paper. The SHMF normalization parameter and the main deflector parameters are drawn from tight, shifted distributions compared to the training and validation sets. The held out COSMOS images are used.
config_test_high_m_min.py
	Configuration file with same distributions as validation set, but with the minimum subhalo mass shifted to :math:`5 \times 10^{7}`. Used in Appendix.
config_test_low_m_min.py
	Configuration file with same distributions as validation set, but with the minimum subhalo mass shifted to :math:`5 \times 10^{6}`. Used in Appendix.
config_test_source_smoothing.py
	Configuration file with same distributions as validation set, but with a smoothing applied to the sources. Used in Appendix.
config_test_large_los_dz.py
	Configuration file with same distributions as validation set, but with a larger step in redshift space used to realize line-of-sight halos. Used in Appendix.
config_test_large_cone_angle.py
	Configuration file with same distributions as validation set, but with a larger opening angle than the training set used to realize line-of-sight haos. Used in Appendix.
