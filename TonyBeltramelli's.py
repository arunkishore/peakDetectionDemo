__author__ = 'Tony Beltramelli - 07/11/2015'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt

def detect_peaks(signal, threshold=0.5):
	""" Performs peak detection on three steps: root mean square, peak to
	average ratios and first order logic.
	threshold used to discard peaks too small """
	# compute root mean square
	root_mean_square = sqrt(np.sum(np.square(signal) / len(signal)))
	# compute peak to average ratios
	ratios = np.array([pow(x / root_mean_square, 2) for x in signal])
	# apply first order logic
	peaks = (ratios > np.roll(ratios, 1)) & (ratios > np.roll(ratios, -1)) & (ratios > threshold)
	# optional: return peak indices
	peak_indexes = []
	for i in range(0, len(peaks)):
		if peaks[i]:
			peak_indexes.append(i)
	return peak_indexes

#### sample code to use the above function

import numpy as np
vector = [
    0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
    1, 20, 7, 3, 0 ]
print('Detect peaks with height threshold.')
indexes = detect_peaks(vector, 1.5)
print('Peaks are: %s' % (indexes))

