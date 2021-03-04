# Neutral patterns in age structured populations

This repository contains matlab code used to generate cultural change in age-structured populations. 

## Project aims

* Understanding the impact of age structure on 
  * the cultural composition of a population at a single point in time (as measured by various population-level and sample-level statistics) and the reliability of standard neutrality test such as the Ewens-Watterson test
  * the temporal dynamic of cultural change (as measured by the progeny distribution)
* Development of a general understanding of the impact of demographic properties on cultural change and our ability to detect underlying processes of cultural transmission  

## File structure

The file

`<main_ageSim>`

defines all parameter used and executes the simulation. Depending on the chosen settings it provides the frequencies of all variant types over the time interval _[0,tMAx]_ (saved as a .csv file), the composition of the population at the last time step (saved as a .mat file) and the progeny distribution generated over _[0,tMAx]_ (sabed as a .csv file) as well as the estimation of its power law behaviour.

