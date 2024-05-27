import random
from rdkit.Chem import MolToSmiles, MolFromSmiles

double_symbols = ['Uue', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al',
                  'Si', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                  'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Kr', 'Rb', 'Sr', 'Zr',
                  'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                  'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                  'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                  'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                  'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np',
                  'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
                  'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Cl', 'Br']
symbols = ['N', 'O', 'F', "I", 'P', 'S', 'B', 'C', 'K', 'V', 'Y', 'W', 'U']
symbol2Id = {'H': 1,
             'He': 2,
             'Li': 3,
             'Be': 4,
             'B': 5,
             'C': 6,
             'N': 7,
             'O': 8,
             'F': 9,
             'Ne': 10,
             'Na': 11,
             'Mg': 12,
             'Al': 13,
             'Si': 14,
             'P': 15,
             'S': 16,
             'Cl': 17,
             'Ar': 18,
             'K': 19,
             'Ca': 20,
             'Sc': 21,
             'Ti': 22,
             'V': 23,
             'Cr': 24,
             'Mn': 25,
             'Fe': 26,
             'Co': 27,
             'Ni': 28,
             'Cu': 29,
             'Zn': 30,
             'Ga': 31,
             'Ge': 32,
             'As': 33,
             'Se': 34,
             'Br': 35,
             'Kr': 36,
             'Rb': 37,
             'Sr': 38,
             'Y': 39,
             'Zr': 40,
             'Nb': 41,
             'Mo': 42,
             'Tc': 43,
             'Ru': 44,
             'Rh': 45,
             'Pd': 46,
             'Ag': 47,
             'Cd': 48,
             'In': 49,
             'Sn': 50,
             'Sb': 51,
             'Te': 52,
             'I': 53,
             'Xe': 54,
             'Cs': 55,
             'Ba': 56,
             'La': 57,
             'Ce': 58,
             'Pr': 59,
             'Nd': 60,
             'Pm': 61,
             'Sm': 62,
             'Eu': 63,
             'Gd': 64,
             'Tb': 65,
             'Dy': 66,
             'Ho': 67,
             'Er': 68,
             'Tm': 69,
             'Yb': 70,
             'Lu': 71,
             'Hf': 72,
             'Ta': 73,
             'W': 74,
             'Re': 75,
             'Os': 76,
             'Ir': 77,
             'Pt': 78,
             'Au': 79,
             'Hg': 80,
             'Tl': 81,
             'Pb': 82,
             'Bi': 83,
             'Po': 84,
             'At': 85,
             'Rn': 86,
             'Fr': 87,
             'Ra': 88,
             'Ac': 89,
             'Th': 90,
             'Pa': 91,
             'U': 92,
             'Np': 93,
             'Pu': 94,
             'Am': 95,
             'Cm': 96,
             'Bk': 97,
             'Cf': 98,
             'Es': 99,
             'Fm': 100,
             'Md': 101,
             'No': 102,
             'Lr': 103,
             'Rf': 104,
             'Db': 105,
             'Sg': 106,
             'Bh': 107,
             'Hs': 108,
             'Mt': 109,
             'Ds': 110,
             'Rg': 111,
             'Cn': 112,
             'Nh': 113,
             'Fl': 114,
             'Mc': 115,
             'Lv': 116,
             'Ts': 117,
             'Og': 118,
             'Uue': 119}


def rdkitCanonical(smiles, mol=None):
    if not mol:
        mol = MolFromSmiles(smiles)
    return MolToSmiles(mol)


def addExplicitHs(smiles, mol=None):
    if not mol:
        mol = MolFromSmiles(smiles)
    return MolToSmiles(mol, allHsExplicit=True)


def kekulizeSmiles(smiles, mol=None):
    if not mol:
        mol = MolFromSmiles(smiles)
    return MolToSmiles(mol, kekuleSmiles=True)


def varySmiles(smiles, mol=None, atomId=-1, doRandom=False):
    if not mol:
        mol = MolFromSmiles(smiles)
    return MolToSmiles(mol, rootedAtAtom=atomId, doRandom=doRandom)


def replaceAtomsWNumbers(smiles, mol=None):
    smiles = addExplicitHs(smiles, mol)
    for x in double_symbols:
        if x in smiles:
            smiles = smiles.replace(x, f"#{symbol2Id[x]}")
    for x in symbols:
        smiles = smiles.replace(x, f"#{symbol2Id[x]}")
    return smiles


def cycleRenumering(smiles, mol=None, method="dummy"):
    symbols_seq = [""]
    cycle_ids_seq = []

    is_in_square_bracket = False
    for symbol in smiles:
        if symbol == "[":
            is_in_square_bracket = True
            symbols_seq[-1] += symbol
        elif symbol == "]":
            is_in_square_bracket = False
            symbols_seq[-1] += symbol
        elif is_in_square_bracket:
            symbols_seq[-1] += symbol
        elif not symbol.isnumeric():
            symbols_seq[-1] += symbol
        else:
            symbols_seq.append("")
            cycle_ids_seq.append(symbol)
    possible_cycle_ids = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    open_cycles = set()
    new_cycle_ids_seq = []

    if method == "dummy":
        for seq_id, cycle_id in enumerate(cycle_ids_seq):
            new_cycle_ids_seq.append(cycle_id)
            if cycle_id not in open_cycles:
                open_cycles.add(cycle_id)
            else:
                if cycle_ids_seq[seq_id - 1] == cycle_id:
                    new_cycle_id = random.choice(list(possible_cycle_ids - open_cycles))
                    new_cycle_ids_seq[-2] = new_cycle_ids_seq[-1] = new_cycle_id
                    open_cycles.remove(cycle_id)
    else:
        raise NotImplemented

    smiles_tokens = [symbols_seq[i] + new_cycle_ids_seq[i] for i in range(len(new_cycle_ids_seq))]
    smiles_tokens.append(symbols_seq[-1])

    return "".join(smiles_tokens)

def get_rdkitCanonical(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        augs_list.append(rdkitCanonical(smile, mol))  # Add rdkit canonical smiles
    return set(augs_list)

def get_ExplicitHs(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        augs_list.append(addExplicitHs(smile, mol))  # Add explicit hydrogens
    return set(augs_list)

def get_kekulizeSmiles(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        augs_list.append(kekulizeSmiles(smile, mol))  # Add SMILES in Kekule form
    return set(augs_list)

def get_NumAtoms(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        for atomId in range(mol.GetNumAtoms()):  # Start traversal from different atoms
            augs_list.append(varySmiles(smile, mol, atomId))
    return set(augs_list)

def get_replaceAtomsWNumbers(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        augs_list.append(
        replaceAtomsWNumbers(smile, mol))  # Add explicit hydrogens and replace atoms symbols with atomic numbers
    return set(augs_list)

def get_cycleRenumering(smiles):
    augs_list = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        augs_list.append(cycleRenumering(smile, mol, method="dummy"))  # Rename consecutive cycle ids
    return set(augs_list)


def valid_augmentations_set(smiles):
    augs_list = []
    mol = MolFromSmiles(smiles)
    augs_list.append(rdkitCanonical(smiles, mol))  # Add rdkit canonical smiles
    augs_list.append(addExplicitHs(smiles, mol))  # Add explicit hydrogens
    augs_list.append(kekulizeSmiles(smiles, mol))  # Add SMILES in Kekule form

    for atomId in range(mol.GetNumAtoms()):  # Start traversal from different atoms
        augs_list.append(varySmiles(smiles, mol, atomId))

    augs_list.append(
        replaceAtomsWNumbers(smiles, mol))  # Add explicit hydrogens and replace atoms symbols with atomic numbers

    augs_list.append(cycleRenumering(smiles, mol, method="dummy"))  # Rename consecutive cycle ids
    return set(augs_list)
