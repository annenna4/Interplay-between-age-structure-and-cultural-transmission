# The interplay between age structure and cultural transmission

## Paper
This repository contains all code used for the manuscript:

'The interplay between age structure and cultural transmission' by Anne Kandler, Laurel
Fogarty, and Folgert Karsdorp

> *Abstract*: Empirical work has shown that human cultural transmission can be
> heavily influenced by population age structure. We aim to explore the role of
> such age structure in shaping the cultural composition of a population when
> cultural transmission occurs in an unbiased way. In particular, we are
> interested in understanding the effect induced by the interplay between age
> structure and the cultural transmission process by allowing cultural
> transmission from individuals within a limited age range only. To this end we
> develop an age-structured cultural transmission model and find that
> age-structured and non age-structured population evolving through unbiased
> transmission possess very similar cultural compositions (at a single point in
> time) at the population and sample level if the copy pool consists of a good
> fraction of the population. If, however, a recency bias --- a bias favouring
> the transmission of recently transmitted cultural variants, i.e. the
> transmission from young individuals in the population --- exist the cultural
> compositions age structured and non age-structured population show stark
> differences. This may have drastic consequences for our ability to correctly
> analyse cultural data sets. Tests of neutrality blind to age structure and,
> importantly, the interaction between age structure and cultural transmission
> are only applicable to datasets where it is known *a priori* that no or only a
> weak recency bias is operating. As this knowledge is rarely available for
> specific empirical case studies we develop a generative inference approach
> based on our age-structured cultural transmission model and novel machine
> learning techniques. We show that in some circumstances it is possible to
> simultaneously infer the characteristics of the age structure, the nature of
> the transmission process, and the interplay between them from observed samples
> of cultural variants. Our results also point to hard limits on inference from
> population-level data at a single point in time, regardless of the approach
> used.

## Code

All code is implemented in Python. The script `synthetic.py` can be used to
generate populations using the simulation model described in the paper. The
actual implementation of the simulation model can be found in the script
`simulation.py`. The actual experiments, analyses and output are described in
the Jupyter notebook
[notebooks/scenario-analysis.ipynb](https://github.com/annenna4/neutral_patterns_in_age_structured_populations/blob/main/notebooks/scenario-analysis.ipynb).

Analyses were done with Python 3.8. All dependencies can be installed from the
`requirements.txt` file from the top-level directory in the repository:

```bash
>>> python -m pip install -r requirements.txt
```


## License
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
