# Variables in this file are directly copied from https://github.com/hjkgrp/molSimplify/blob/31ae03eec6eaa1d3d7946f6413d756051f8feea5/molSimplify/Classes/globalvars.py
# Slight modifications might be added in the future.

# Dictionary containing atomic mass, atomic number, covalent radius, number of valence electrons
# Data from http://www.webelements.com/ (last accessed May 13th 2015)
elementdict = {'X': (1.0, 0, 0.77, 0),     'H': (1.0079, 1, 0.37, 1),     'He': (4.002602, 2, 0.46, 2),
             'Li': (6.94, 3, 1.33, 1),   'Be': (9.0121831, 4, 1.02, 2), 'B': (10.83, 5, 0.85, 3),
             'C': (12.0107, 6, 0.77, 4), 'N': (14.0067, 7, 0.75, 5),    'O': (15.9994, 8, 0.73, 6), 
             'F': (18.9984, 9, 0.71, 7), 'Ne': (20.1797, 10, 0.67, 8),  'Na': (22.99, 11, 1.55, 1), 
             'Mg': (24.30, 12, 1.39, 2), 'Al': (26.98, 13, 1.26, 3),    'Si': (28.08, 14, 1.16, 4),
             'P': (30.9738, 15, 1.06, 5),'S': (32.065, 16, 1.02, 6),    'Cl': (35.453, 17, 0.99, 7),
             'Ar': (39.948, 18, 0.96, 8),'K': (39.10, 19, 1.96, 1),     'Ca': (40.08, 20, 1.71, 2),
             'Sc': (44.96, 21, 1.7, 3),  'Ti': (47.867, 22, 1.36, 4),   'V': (50.94, 23, 1.22, 5),
             'Cr': (51.9961, 24, 1.27, 6),'Mn': (54.938, 25, 1.39, 7),  'Fe': (55.84526, 26, 1.25, 8), 
             'Co': (58.9332, 27, 1.26, 9), 'Ni': (58.4934, 28, 1.21, 10),'Cu': (63.546, 29, 1.38, 11), 
             'Zn': (65.39, 30, 1.31, 12), 'Ga': (69.72, 31, 1.24, 3), 'Ge': (72.63, 32, 1.21, 4), 
             'As': (74.92, 33, 1.21, 5),'Se': (78.96, 34, 1.16, 6),'Br': (79.904, 35, 1.14, 7), 
             'Kr': (83.798, 36, 1.17, 8),'Rb': (85.47, 37, 2.10, 1), 'Sr': (87.62, 38, 1.85, 2),
             'Y': (88.91, 39, 1.63, 3), 'Zr': (91.22, 40, 1.54, 4), 'Nb': (92.91, 41, 1.47, 5),
             'Mo': (95.96, 42, 1.38, 6),'Tc': (98.9, 43, 1.56, 7), 'Ru': (101.1, 44, 1.25, 8), 
             'Rh': (102.9, 45, 1.25, 9),'Pd': (106.4, 46, 1.20, 10),'Ag': (107.9, 47, 1.28, 11), 
             'Cd': (112.4, 48, 1.48, 12),'In': (111.818, 49, 1.42, 3), 'Sn': (118.710, 50, 1.40, 4), 
             'Sb': (121.760, 51, 1.40, 5),'Te': (127.60, 52, 1.99, 6),'I': (126.90447, 53, 1.40, 7), 
             'Xe': (131.293, 54, 1.31, 8),'Cs': (132.9055, 55, 2.32, 1), 'Ba': (137.327, 56, 1.96, 2),
             'La': (138.9, 57, 1.69, 3), 'Ce': (140.116, 58, 1.63, 4), 'Pr': (140.90766, 59, 1.76, 5),
             'Nd': (144.242, 60, 1.74, 6),'Pm': (145, 61, 1.73, 7), 'Sm': (150.36, 62, 1.72, 8), 
             'Eu': (151.964, 63, 1.68, 9),'Gd': (157.25, 64, 1.69, 10),'Tb': (158.92535, 65, 1.68, 11), 
             'Dy': (162.500, 66, 1.67, 12), 'Ho': (164.93033, 67, 1.66, 13),'Er': (167.259, 68, 1.65, 14), 
             'Tm': (168.93422, 69, 1.64, 15), 'Yb': (173.045, 70, 1.70, 16),'Lu': (174.9668, 71, 1.62, 3),
             'Hf': (178.5, 72, 1.50, 8), 'Ta': (180.9, 73, 1.38, 5), 'W': (183.8, 74, 1.46, 6),
             'Re': (186.2, 75, 1.59, 7),'Os': (190.2, 76, 1.28, 8), 'Ir': (192.2, 77, 1.37, 9), 
             'Pt': (195.1, 78, 1.23, 10),'Au': (197.0, 79, 1.24, 11),'Hg': (200.6, 80, 1.49, 2),
             'Tl': (204.38, 81, 1.44, 3), 'Pb': (207.2, 82, 1.44, 4), 'Bi': (208.9804, 83, 1.51, 5),
             'Po': (208.98, 84, 1.90, 6), 'At': (209.99, 85, 2.00, 7), 'Rn': (222.6, 86, 142, 4),
             'Fr': (223.02, 87, 3.48, 8),'Ra': (226.03, 88, 2.01, 2), 'Ac': (277, 89, 1.86, 3), 
             'Th': (232.0377, 90, 1.75, 4),'Pa': (231.04, 91,2.00, 5),'U': (238.02891, 92, 1.70, 6),
             'Np': (237.05, 93, 1.90, 7), 'Pu': (244.06, 94, 1.75, 8),'Am': (243.06, 95,1.80, 9),
             'Cm': (247.07, 96, 1.69, 10), 'Bk': (247.07, 97, 1.68, 11),'Cf': (251.08, 98, 1.68, 12)}

# Electronegativity (Pauling)
endict = {"H": 2.20, "He": 4.16,
          "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
          "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
          "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
          "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81,
          "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
          "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 2.10, "Ru": 2.20, "Rh": 2.28,
          "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "I": 2.66,
          "Cs": 0.79, "Ba": 0.89, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
          "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
          "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
          "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27,
          "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
          "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
          "Yb": 1.1, "Eu": 1.2, "Tb": 1.1, "Te": 2.10}

# Polarizability (alpha) 
# From https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1535143
# Last accessed 4/28/20
poldict = {"H": 4.50711, "He": 1.38375, 
           "Li": 164.1125, "Be": 37.74, "B": 20.5, "C": 11.3, "N": 7.4,
           "O":5.3, "F": 3.74, "Ne": 2.66, "Na": 162.7, "Mg":71.2, "Al": 57.8, "Si": 37.3, "P": 25, 
           "S": 19.4, "Cl": 14.6, "Ar": 11.083, "K": 289.7, "Ca": 160.8, "Sc": 97, "Ti": 100, 
           "V": 87, "Cr": 83, "Mn": 68, "Fe": 62, "Co": 55, "Ni": 49, "Cu": 46.5, "Zn": 38.67,
           "Ga": 50, "Ge": 40, "As": 30, "Se": 28.9, "Br": 21, "Kr": 16.78, "Rb": 319.8, "Sr": 197.2,
           "Y": 162, "Zr": 112, "Nb": 98, "Mo": 87, "Tc": 79, "Ru": 72, "Rh": 66, "Pd": 26.14,
           "Ag": 55, "Cd": 46, "In": 65, "Sn": 53, "Sb": 43, "Te": 38, "I": 32.9, "Xe": 27.32,
           "Cs": 400.9, "Ba": 272, "La": 215, "Ce": 205, "Pr": 216, "Nd": 208, "Pm": 200, "Sm": 192,
           "Eu": 184, "Gd": 158, "Tb": 170, "Dy": 163, "Ho": 156, "Er": 150, "Tm": 144, 
           "Yb": 139, "Lu": 137, "Hf": 103, "Ta": 74, "W": 68, "Re": 62, "Os": 57, "Ir": 54,
           "Pt": 48, "Au": 36, "Hg": 33.91, "Tl": 50, "Pb": 47, "Bi": 48, "Po": 44, "At": 42, 
           "Rn": 35, "Fr": 317.8, "Ra": 246, "Ac": 203, "Pa": 154, "U": 129, "Np": 151, "Pu": 132,
           "Am": 131, "Cm": 144, "Bk": 125, "Cf": 122, "Es": 118, "Fm": 113, "Md": 109, "No": 110,
           "Lr": 320, "Rf": 112, "Db": 42, "Sg": 40, "Bh": 38, "Hs": 36, "Mt": 34, "Ds": 32, 
           "Rg": 32, "Cn": 28, "Nh": 29, "Fl": 31, "Mc": 71, "Ts": 76, "Og": 58}

# Metals (includes alkali, alkaline earth, and transition metals)
metalslist = ['Li', 'li', 'LI', 'lithium', 'Be', 'be', 'BE', 'beryllium',
    'Na', 'na', 'NA', 'sodium', 'Mg', 'mg', 'MG', 'magnesium',
    'Al', 'al', 'AL', 'aluminum', 'aluminium',
    'K', 'k', 'potassium', 'Ca', 'ca', 'CA', 'calcium',
    'Rb', 'rb', 'RB', 'rubidium', 'Sr', 'sr', 'SR', 'strontium',
    'Cs', 'cs', 'CS', 'cesium', 'Ba', 'ba', 'BA', 'barium',
    'Fr', 'fr', 'FR', 'francium', 'Ra', 'ra', 'RA', 'radium',
    'Sc', 'sc', 'SC', 'scandium', 'Ti', 'ti', 'TI', 'titanium',
    'V', 'v', 'vanadium', 'Cr', 'cr', 'CR', 'chromium',
    'Mn', 'mn', 'MN', 'manganese', 'Fe', 'fe', 'FE', 'iron',
    'Co', 'co', 'CO', 'cobalt', 'Ni', 'ni', 'NI', 'nickel',
    'Cu', 'cu', 'CU', 'copper', 'Zn', 'zn', 'ZN', 'zinc',
    'Ga', 'ga', 'GA', 'gallium',
    'Y', 'y', 'yttrium', 'Zr', 'zr', 'ZR', 'zirconium',
    'Nb', 'nb', 'NB', 'niobium', 'Mo', 'mo', 'MO', 'molybdenum',
    'Tc', 'tc', 'TC', 'technetium', 'Ru', 'ru', 'RU', 'ruthenium',
    'Rh', 'rh', 'RH', 'rhodium', 'Pd', 'pd', 'PD', 'palladium',
    'Ag', 'ag', 'AG', 'silver', 'Cd', 'cd', 'CD', 'cadmium',
    'In', 'in', 'IN', 'indium', 'Sn', 'sn', 'SN', 'tin',
    'Hf', 'hf', 'HF', 'hafnium', 'Ta', 'ta', 'TA', 'tantalum',
    'W', 'w', 'tungsten', 'Re', 're', 'RE', 'rhenium',
    'Os', 'os', 'OS', 'osmium', 'Ir', 'ir', 'IR', 'iridium',
    'Pt', 'pt', 'PT', 'platinum', 'Au', 'au', 'AU', 'gold',
    'Hg', 'hg', 'HG', 'mercury', 'X',
    'Tl', 'tl', 'TL', 'thallium', 'Pb', 'pb', 'PB', 'lead',
    'Bi', 'bi', 'BI', 'bismuth', 'Po', 'po', 'PO', 'polonium',
    'La', 'la', 'LA', 'lanthanum',
    'Ce', 'ce', 'CE', 'cerium', 'Pr', 'pr', 'PR', 'praseodymium',
    'Nd', 'nd', 'ND', 'neodymium', 'Pm', 'pm', 'PM', 'promethium',
    'Sm', 'sm', 'SM', 'samarium', 'Eu', 'eu', 'EU', 'europium',
    'Gd', 'gd', 'GD', 'gadolinium', 'Tb', 'tb', 'TB', 'terbium',
    'Dy', 'dy', 'DY', 'dysprosium', 'Ho', 'ho', 'HO', 'holmium',
    'Er', 'er', 'ER', 'erbium', 'Tm', 'tm', 'TM', 'thulium',
    'Yb', 'yb', 'YB', 'ytterbium', 'Lu', 'lu', 'LU', 'lutetium',
    'Ac', 'ac', 'AC', 'actinium', 'Th', 'th', 'TH', 'thorium',
    'Pa', 'pa', 'PA', 'proactinium', 'U', 'u', 'uranium',
    'Np', 'np', 'NP', 'neptunium', 'Pu', 'pu', 'PU', 'plutonium',
    'Am', 'am', 'AM', 'americium', 'Cu', 'cu', 'CU', 'curium',
    'Bk', 'bk', 'BK', 'berkelium', 'Cf', 'cf', 'CF', 'californium',
    'Es', 'es', 'ES', 'einsteinium', 'Fm', 'fm', 'FM', 'fermium',
    'Md', 'md', 'MD', 'mendelevium', 'No', 'no', 'NO', 'nobelium',
    'Lr', 'lr', 'LR', 'lawrencium'
]

def ismetal(atom: str) -> bool:
    return atom in metalslist

# The following functions and dictionaries are intended for autocorrelation.

covalent_radius = lambda atom: elementdict[atom][2]

properties = {
    'electronegativity': lambda atom: endict[atom],
    'atomic number': lambda atom: elementdict[atom][1],
    'identity': lambda atom: 1,
    'covalent radius': covalent_radius,
    'polarizability': lambda atom :poldict[atom]
}

property_notation = {
    'electronegativity': 'chi',
    'atomic number': 'Z',
    'identity': "I",
    'covalent radius': 'S',
    'topology': "T",
    'polarizability': "alpha"
}