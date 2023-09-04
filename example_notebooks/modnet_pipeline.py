from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import pandas as pd
from modnet.featurizers.presets import DeBreuck2020Featurizer


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
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1:], 'GPU')

file_path = sys.argv[1]
target_property = sys.argv[2]

with tf.device('/device:GPU:1'):
    df_train = pd.read_csv(os.path.join(file_path, target_property, "train.csv"))
    df_test = pd.read_csv(os.path.join(file_path, target_property, "test.csv"))
    df_val = pd.read_csv(os.path.join(file_path, target_property, "val.csv"))
                          
    df_train["composition"] = df_train["formula"].map(Composition) # maps composition to a pymatgen composition object

    # Creating MODData
    data_train = MODData(materials = df_train["composition"],
                   targets = df_train[target_name],
                   target_names=[target_name],
                   featurizer=basic_featurizer,
                   structure_ids=df_train.index, )

    data_train.featurize()
    data_train.feature_selection(n=200)
    df_val["composition"] = df_val["formula"].map(Composition) # maps composition to a pymatgen composition object
    data_val = MODData(materials = df_val["composition"],
                   targets = df_val[target_name],
                   target_names=[target_name],
                   featurizer=basic_featurizer,
                   structure_ids=df_val.index, )

    data_val.featurize()
    data_val.feature_selection(n=200)
    
    # Creating MODNetModel
    model = MODNetModel([[[target_name]]],
                        weights={target_name:1},
                        num_neurons=[[256],[64],[64],[32]],
                       )
    
    model.fit(data_train,
              val_data = data_val,
              epochs = 250,
              verbose = 1
             )
    
    # # Predicting on unlabeled data
    df_test["composition"] = df_test["formula"].map(Composition)
    data_to_predict = MODData(materials = df_test["composition"],
                   targets = df_test[target_name],
                   target_names=[target_name],
                   featurizer=basic_featurizer,
                   structure_ids=df_test.index, )
    data_to_predict.featurize()
    data_to_predict.feature_selection(n=200)
    df_predictions = model.predict(data_to_predict)
    df_test_pred = df_test.merge(df_predictions, how = 'left', left_index = True, right_index = True, suffixes=('_true', '_pred'))
    mae = mean_absolute_error(df_test_pred[target_name+'_true'].values,df_test_pred[target_name+'_pred'].values)
    print("-" * 40)
    print(f"{target_property}: {mae}")
    mae_dic[target_property] = mae
    df_test_pred.to_csv(os.path.join(file_path, target_property, "test_pred.csv"))   

    df_mae_all = pd.read_csv(os.path.join(file_path, "mae_all.csv"), index_col = 0)
    new_entry = {'target': target_property, 'mae': mae}
    df_mae_all.loc[len(df_mae_all)] = new_entry
    df_mae_all.to_csv(os.path.join(file_path, "mae_all.csv"))
    