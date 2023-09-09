# from jarvis.db.figshare import data
# from jarvis.core.atoms import Atoms
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
epochs = 600
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[1:], 'GPU')

file_path = sys.argv[1]
target_property = sys.argv[2]

# with tf.device('/device:GPU:1'):
df_train = pd.read_csv(os.path.join(file_path, target_property, "train.csv"))
df_test = pd.read_csv(os.path.join(file_path, target_property, "test.csv"))
df_val = pd.read_csv(os.path.join(file_path, target_property, "val.csv"))

df_train_featurized = pd.read_csv(os.path.join(file_path, f"df_train_all_featurized.csv"), index_col = 0).drop(columns = 'target', errors='ignore') 
df_val_featurized = pd.read_csv(os.path.join(file_path, f"df_val_all_featurized.csv"), index_col = 0).drop(columns = 'target', errors='ignore')
df_test_featurized = pd.read_csv(os.path.join(file_path, f"df_test_all_featurized.csv"), index_col = 0).drop(columns = 'target', errors='ignore')
# print(df_train_featurized)                        
# df_train["composition"] = df_train["formula"].map(Composition) # maps composition to a pymatgen composition object

df_train_featurized = df_train.merge(df_train_featurized, how='left', left_on = 'formula', right_index = True)
df_val_featurized = df_val.merge(df_val_featurized, how='left', left_on = 'formula', right_index = True)
df_test_featurized = df_test.merge(df_test_featurized, how='left', left_on = 'formula', right_index = True)
# Creating MODData
df_train_featurized["composition"] = df_train_featurized["formula"].map(Composition)
print(df_train_featurized[df_train_featurized.isna().any(axis=1)])
data_train = MODData(materials = df_train_featurized["composition"],
                targets = df_train_featurized[target_name],
                target_names=[target_name],
                df_featurized = df_train_featurized.drop(columns = ['formula', target_name, 'composition']),
                featurizer=basic_featurizer,
                structure_ids=df_train_featurized.formula)
# print(list(df_train_featurized.drop(columns = ['target', 'formula']).columns))
# exit(1)

# data_train.featurize()
data_train.feature_selection(n=200)

df_val_featurized["composition"] = df_val_featurized["formula"].map(Composition)
data_val = MODData(materials = df_val_featurized["composition"],
                targets = df_val_featurized[target_name],
                target_names=[target_name],
                df_featurized = df_val_featurized.drop(columns = ['formula', target_name, 'composition']),
                featurizer=basic_featurizer,
                structure_ids=df_val_featurized.formula)


# data_val.featurize()

# Creating MODNetModel
model = MODNetModel([[[target_name]]],
                    weights={target_name:1},
                    num_neurons=[[256],[128],[64],[8]],
                    )

model.fit(data_train,
            val_data = data_val,
            epochs = epochs,
            batch_size = 256,
            verbose = 1
            )

# # Predicting on unlabeled data
df_test_featurized["composition"] = df_test_featurized["formula"].map(Composition)
data_to_predict = MODData(materials = df_test_featurized["composition"],
                featurizer=basic_featurizer,
                df_featurized = df_test_featurized.drop(columns = ['formula', target_name, 'composition']),
                structure_ids=df_test_featurized.formula)
data_to_predict.featurize()
df_predictions = model.predict(data_to_predict)
df_test_pred = df_test.merge(df_predictions, how = 'left', left_on = "formula", right_index = True, suffixes=('_true', '_pred'))
mae = mean_absolute_error(df_test_pred[target_name+'_true'].values,df_test_pred[target_name+'_pred'].values)
print("-" * 40)
print(f"{target_property}: {mae}")
if os.path.exists(os.path.join(file_path, f"mae_all_epch_{epochs}.csv")):
    df_mae_all = pd.read_csv(os.path.join(file_path, f"mae_all_epch_{epochs}.csv"), index_col = 0)
    new_entry = {'target': target_property, 'mae': mae}
    df_mae_all.loc[len(df_mae_all)] = new_entry
else:
    mae_dic = {'target': [target_property], 'mae': [mae]}
    df_mae_all = pd.DataFrame.from_dict(mae_dic)
mae_dic[target_property] = mae
df_test_pred.to_csv(os.path.join(file_path, target_property, f"test_pred_epch_{epochs}.csv"))   

