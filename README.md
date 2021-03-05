# Neutral patterns in age structured populations

This repository contains matlab code used to generate cultural change in age-structured populations and some output files. 

## Project aims

* Understanding the impact of age structure on 
  * the cultural composition of a population at a single point in time (as measured by various population-level and sample-level statistics) and the reliability of standard neutrality test such as the Ewens-Watterson test,
  * the temporal dynamic of cultural change (as measured by the progeny distribution)
* Development of a general understanding of the effects of demographic properties on cultural change and our ability to detect underlying processes of cultural transmission  

## File structure

The file

`main_ageSim.m`

defines all parameter used and executes the simulation. Depending on the chosen settings it provides the frequencies of all variant types over the time interval _[0,tMAx]_ (saved as a .csv file), the composition of the population at the last time step (saved as a .mat file) and the progeny distribution generated over _[0,tMAx]_ (sabed as a .csv file) as well as the estimation of its power law behaviour. The burn-in phase is carried out by the function

`get_burnIn.m`.

It is not straightforward to determine when the cultural system has reached equilibrium and here we use a relatively conservative approach: we assume that the system has reached equilibrium when none of the initial variant types are present in the population anymore, i.e. every excisting type has undergone neutral dynamics. However, this approach can be very time-intensive, especially for low innovation rates or positive frequency-dependent biases. An alternative approach may be to evaluate when the heterogeneity inidices of two populations initialised with maximum and minimum heterogeneity, respectively intersect. This approach is less time-intensive but we still need to confirm that the later approach indeed reaches the equilibrium state. The function 

`get_dynamics.m`

describes the cultural and demographic dynamics of the considered cultural system. 
