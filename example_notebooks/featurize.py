# from jarvis.db.figshare import data
# from jarvis.core.atoms import Atoms
import pandas as pd
from modnet.featurizers.presets import DeBreuck2020Featurizer


# To launch this script. modify 
# python featurize.py /path/to/data

lst_properties = ['avg_elec_mass','avg_hole_mass','bulk_modulus_kv','dfpt_piezo_max_dielectric',
                  'density','dfpt_piezo_max_dij','dfpt_piezo_max_eij','dfpt_piezo_max_dielectric_electronic','encut',
                  'epsx','epsy','epsz','et_c11','dfpt_piezo_max_dielectric_ionic','ehull','et_c12','et_c13','et_c22',
                  'et_c33','et_c66','et_c44','et_c55','exfoliation_energy','formation_energy_peratom',
                  'kpoint_length_unit','magmom_oszicar','magmom_outcar','max_efg','max_ir_mode','max_mode',
                  'mbj_bandgap','mepsx','mepsy','min_mode','n-Seebeck','min_ir_mode','n-powerfact','n_em300k',
                  'mepsz','optb88vdw_bandgap','optb88vdw_total_energy','p-Seebeck','p-powerfact','poisson',
                  'shear_modulus_gv','p_em300k','slme','spillage']

class BasicFeaturizer(DeBreuck2020Featurizer):
    from pymatgen.analysis.local_env import VoronoiNN

    from matminer.featurizers.composition import (
        AtomicOrbitals,
        ElementFraction,
        ElementProperty,
        Stoichiometry,
        TMetalFraction,
        ValenceOrbital,
    )

    from matminer.featurizers.structure import (
        BondFractions,
        ChemicalOrdering,
        CoulombMatrix,
        DensityFeatures,
        EwaldEnergy,
        GlobalSymmetryFeatures,
        MaximumPackingEfficiency,
        RadialDistributionFunction,
        SineCoulombMatrix,
        StructuralHeterogeneity,
        XRDPowderPattern,
    )

    from matminer.featurizers.site import (
        AGNIFingerprints,
        AverageBondAngle,
        AverageBondLength,
        BondOrientationalParameter,
        ChemEnvSiteFingerprint,
        CoordinationNumber,
        CrystalNNFingerprint,
        GaussianSymmFunc,
        GeneralizedRadialDistributionFunction,
        LocalPropertyDifference,
        OPSiteFingerprint,
        VoronoiFingerprint,
    )

    oxid_composition_featurizers = ()

    composition_featurizers = (
        AtomicOrbitals(),
        ElementFraction(),
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        TMetalFraction(),
        ValenceOrbital(),
    )

    site_featurizers = (
        AGNIFingerprints(),
        AverageBondAngle(VoronoiNN()),
        AverageBondLength(VoronoiNN()),
        BondOrientationalParameter(),
        ChemEnvSiteFingerprint.from_preset("simple"),
        CoordinationNumber(),
        CrystalNNFingerprint.from_preset("ops"),
        GaussianSymmFunc(),
        GeneralizedRadialDistributionFunction.from_preset("gaussian"),
        LocalPropertyDifference(),
        OPSiteFingerprint(),
        VoronoiFingerprint(),
    )

basic_featurizer = BasicFeaturizer()
basic_featurizer.set_n_jobs(20)
# basic_featurizer._n_jobs = None

import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from pymatgen.core import Composition
import warnings
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import sys
warnings.filterwarnings('ignore')

def iterate_dataset(folder_path):
    dataset = []
    for root, subfolders, files in os.walk(folder_path):
        dataset.append(subfolders)
    return dataset[0]


# LOOP 
target_name = "target"
mae_dic = {}

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[1:], 'GPU')

file_path = sys.argv[1]
def iterate_dataset(folder_path):
    dataset = []
    for root, subfolders, files in os.walk(folder_path):
        dataset.append(subfolders)
    return dataset[0]

datasets = iterate_dataset(file_path)
# with tf.device('/device:GPU:1'):

df_train_lst = []
df_val_lst = []
df_test_lst = []
for target_property in datasets[:]:
    if target_property in lst_properties:
        df_train = pd.read_csv(os.path.join(file_path, target_property, "train.csv"))
        df_test = pd.read_csv(os.path.join(file_path, target_property, "test.csv"))
        df_val = pd.read_csv(os.path.join(file_path, target_property, "val.csv"))
        df_train_lst.append(df_train)
        df_val_lst.append(df_val)
        df_test_lst.append(df_test)
df_train_all = pd.concat(df_train_lst, ignore_index=True).drop_duplicates(subset = 'formula', keep = 'first')
df_val_all = pd.concat(df_val_lst, ignore_index=True).drop_duplicates(subset = 'formula', keep = 'first')
df_test_all = pd.concat(df_test_lst, ignore_index=True).drop_duplicates(subset = 'formula', keep = 'first')



        
df_train_all["composition"] = df_train_all["formula"].map(Composition) # maps composition to a pymatgen composition object
df_val_all["composition"] = df_val_all["formula"].map(Composition) # maps composition to a pymatgen composition object
df_test_all["composition"] = df_test_all["formula"].map(Composition) # maps composition to a pymatgen composition object

# Creating MODData
data_train = MODData(materials = df_train_all["composition"],
                featurizer=basic_featurizer,
                structure_ids=df_train_all.formula)

data_train.featurize()
data_train.df_featurized.to_csv(os.path.join(file_path, f"df_train_all_featurized.csv"))

data_val = MODData(materials = df_val_all["composition"],
                featurizer=basic_featurizer,
                structure_ids=df_val_all.formula)

data_val.featurize()
data_val.df_featurized.to_csv(os.path.join(file_path, f"df_val_all_featurized.csv"))
# # Predicting on unlabeled data
data_to_predict = MODData(materials = df_test_all["composition"],
                featurizer=basic_featurizer,
                structure_ids=df_test_all.formula)
data_to_predict.featurize()
data_to_predict.df_featurized.to_csv(os.path.join(file_path, f"df_test_all_featurized.csv"))