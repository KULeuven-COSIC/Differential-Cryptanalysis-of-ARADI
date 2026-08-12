# Differential-Cryptanalysis-of-ARADI

## Preliminary files
transition.py: Calculations of quasidifferential transition matrix. <br>
aradi.py: Specifications of the ARADI cipher. <br>
printing.py: Different printing functions. <br>
masterkey.py: Specifications of the key schedule, to write subkey bits as linear combinations of the master key bits. <br>

## 3 Probability estimates for differentials in ARADI
probability_calculations.ipynb: <br>
Given a differential we search all characteristics with a given key-averaged probability. For these characteristics we can search all independent deterministic quasidifferential trails. Lastly, the probability of the differential is calculated by combining the probability of multiple characteristics.

## 5.1 Search method
keyrecovery.ipynb: Code used to determine table 5 and 6. <br>
differentialcharsearch.ipynb: Search method for new characteristics. <br>
