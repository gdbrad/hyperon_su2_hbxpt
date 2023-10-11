## Remaining tasks ##

- [ ] Fix error budget tabulation 
	- [ ] total errors do not add up to 100%
- [ ] logGBF comparison of data ingestion methods (should be the same!)
	- [ ] data converted before 
	- [ ] data converted after 
	- [ ] decorrelate scale data 
- [ ] Inclusion of chiral logs in xpt expressions 
- [ ] 


### turning off scale setting correlation ###

We can ingest the scale factors data from the input file, which contains the pseudoscalar data.: 
```
{'a06m310L': {'a_fm': 0.05717(51)}, 'a09m135': {'a_fm': 0.08789(71)}, 'a09m220': {'a_fm': 0.08789(71)}, 'a09m310': {'a_fm': 0.08789(71)}, 'a09m350': {'a_fm': 0.08789(71)}, 'a09m400': {'a_fm': 0.08789(71)}, 'a12m130': {'a_fm': 0.12126(87)}, 'a12m220L': {'a_fm': 0.12126(87)}, 'a12m310': {'a_fm': 0.12126(87)}, 'a12m350': {'a_fm': 0.12126(87)}, 'a12m400': {'a_fm': 0.12126(87)}, 'a15m135XL': {'a_fm': 0.1492(10)}, 'a15m220': {'a_fm': 0.1492(10)}, 'a15m310': {'a_fm': 0.1492(10)}, 'a15m310L': {'a_fm': 0.1492(10)}, 'a15m350': {'a_fm': 0.1492(10)}, 'a15m400': {'a_fm': 0.1492(10)}}
```

Alternatively, we can use the a_fm_results.p file to obtain these factors : 
```
{'a15': 0.15036(98), 'a12': 0.12055(83), 'a09': 0.08722(67), 'a06': 0.05686(49)}
```

{$C_H : H \in {\Lambda,\Sigma,\Sigma^*,\Xi,\Xi^*}$: raw correlator data on each ensemble (n=17)
1. convert $C_H$  data at time of ingestion to either
	1. $f_{\pi}$ units or 
	2. physical (mev) units
	Extrapolation results:
	- $C_H : H \in {\Lambda,\Sigma,\Sigma^*}*$ 
	- scale setting correlation: on 
	- 
	
2. Convert $M_H$ extrapolated value(will still be in lattice units) to either 
	1. $f_{\pi}$ units or 
	2. physical (mev) units 
	3. 

