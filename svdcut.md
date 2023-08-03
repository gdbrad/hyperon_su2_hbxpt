## svdcut study ##

When loading the lattice spacing data ``a_fm`` to convert bootstrapped correlator data to MeV units, we have two options:

1. Retain values without creating a new gvar object
2. Create a new gvar object via 
``  gv.gvar(arr[0], arr[1])``

When loading in the bootstrapped hyperon correlator data, we also have two options:
1. manually reconstruct the gvar object to decorrelate x and y data via

`` self.data_subset = {part : self.data['m_'+part] for part in y_particles} ``

``self.y = gv.gvar(dict(gv.mean(self.data_subset)),dict(gv.evalcov(self.data_subset)))``

2. Discard the covariance matrix 

``self.y = gv.gvar(dict(gv.mean(self.data_subset)),dict(gv.sdev(self.data_subset)))``

Model to investigate:
```yaml 
    lam:sigma:sigma_st:l_lo:d_n2lo:s_lo:
      particles : [lambda,sigma,sigma_st]
      eps2a_defn : w0_org
      order_chiral  : null
      order_disc : n2lo
      order_strange : lo
      order_light: lo
      xpt : False
      fv : True
      units: phys
```
1. 
```
---
Extrapolation:
Particle: lambda
mass: 1136(21)
---
Particle: sigma
mass: 1211(20)
---
Particle: sigma_st
mass: 1386(35)
---

---

Error Budget:
lambda
  stat    99.6%
  disc     6.1%
  pp       0.4%
  chiral   0.2%
sigma
  stat    94.7%
  pp       5.3%
  disc     4.7%
  chiral   0.0%
sigma_st
  stat    98.5%
  disc     9.6%
  pp       1.5%
  chiral   0.3%
Least Square Fit:
  chi2/dof [dof] = 0.46 [51]    Q = 1    logGBF = -249.35
```
2. 

```
---
Extrapolation:
Particle: lambda
mass: 1142(39)
---
Particle: sigma
mass: 1245(60)
---
Particle: sigma_st
mass: 1433(79)
---

---

Error Budget:
lambda
  stat    96.2%
  disc    19.4%
  pp       0.5%
  chiral   0.0%
sigma
  stat    87.5%
  disc    28.3%
  pp       2.9%
  chiral   0.1%
sigma_st
  stat    91.6%
  disc    31.4%
  pp       2.4%
  chiral   0.1%
Least Square Fit:
  chi2/dof [dof] = 0.13 [51]    Q = 1    logGBF = -272.23
```


3. 


4. 
```
Extrapolation:
Particle: lambda
mass: 1138(32)
---
Particle: sigma
mass: 1238(52)
---
Particle: sigma_st
mass: 1421(70)
---

---

Error Budget:
lambda
  stat    96.3%
  disc    13.9%
  pp       0.6%
  chiral   0.0%
sigma
  stat    85.7%
  disc    24.9%
  pp       4.1%
  chiral   0.0%
sigma_st
  stat    90.2%
  disc    28.3%
  pp       3.6%
  chiral   0.1%
Least Square Fit:
  chi2/dof [dof] = 0.18 [51]    Q = 1    logGBF = -265.49
```