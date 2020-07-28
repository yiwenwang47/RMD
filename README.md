# Revised molecular descriptors
Commonly used descriptors in cheminformatics revised. They are revised as in being defined with customized specifications such as scope, distance, elemental or other properties, etc.

## Revised autocorrelation descriptors

Molecular descriptors that are based on a molecule's topology and elemental properties. Popular properties used include but are not limited to: atomic number/atomic mass, covalent/van der Waals radius, electronegativity, polarizability, coordination number.

However, properties provided by other calculations such as NBO can also be included. Currently working on NAO/NPA features.

### Moreau-Broto styled autocorrelation
Inspired by https://doi.org/10.1021/acs.jpca.7b08750. Currently only working on complexes with CN/NN type ligands. Could accommodate more. 

Details in calculation are slightly different from their version. All algorithms are rewritten. No important dependency.

### Moran styled and Geary styled autocorrelation
Ideas drawn from conventional 2D autocorrelation descriptors.

## Radial distribution functions
Work in progress.