from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import pandas as pd
import json
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
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1:], 'GPU')

file_path = sys.argv[1]
target_property = sys.argv[2]

target_name = "target"
with open(os.path.join(file_path, "feature_grp.json"), 'r') as json_file:
    feature_grp = json.load(json_file)

df_train = pd.read_csv(os.path.join(file_path, target_property, "train.csv")).fillna("NNa")
df_test = pd.read_csv(os.path.join(file_path, target_property, "test.csv")).fillna("NNa")
df_val = pd.read_csv(os.path.join(file_path, target_property, "val.csv")).fillna("NNa")
key = str(df_train.shape[0])
featurize = key not in feature_grp.keys()


df_train["composition"] = df_train["formula"].map(Composition) # maps composition to a pymatgen composition object
df_test["composition"] = df_test["formula"].map(Composition)
df_val["composition"] = df_val["formula"].map(Composition) # maps composition to a pymatgen composition object
with tf.device('/device:GPU:1'):
    if featurize:
        feature_grp[key]= [target_property]
        data_train = MODData(materials = df_train["composition"],
                       targets = df_train[target_name],
                       target_names=[target_name],
                       featurizer=basic_featurizer,
                       structure_ids=df_train.formula)
    
        data_train.featurize()
        data_train.feature_selection(n=200)
        data_train.df_featurized.to_csv(os.path.join(file_path, f"df_train_featurized_{key}.csv"))
    
        
        data_val = MODData(materials = df_val["composition"],
                       targets = df_val[target_name],
                       target_names=[target_name],
                       featurizer=basic_featurizer,
                       structure_ids=df_val.formula)
        data_val.featurize()
        data_val.df_featurized.to_csv(os.path.join(file_path, f"df_val_featurized_{key}.csv"))
        
    else:
        feature_grp[key].append(target_property)
    
        df_train_featurized = pd.read_csv(os.path.join(file_path, f"df_train_featurized_{key}.csv"), index_col = 0)
        data_train = MODData(materials = df_train["composition"],
                             targets = df_train[target_name],
                             target_names=[target_name],
                             df_featurized = df_train_featurized,
                             structure_ids=df_train_featurized.index)
        data_train.feature_selection(n=200)
        
        df_val_featurized = pd.read_csv(os.path.join(file_path, f"df_val_featurized_{key}.csv"), index_col = 0)
        data_val = MODData(materials = df_val["composition"],
                           targets = df_val[target_name],
                           target_names=[target_name],
                           df_featurized = df_val_featurized,
                           structure_ids=df_val_featurized.index)
    
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
    if featurize:
        data_to_predict = MODData(materials = df_test["composition"],
                       featurizer=basic_featurizer,
                       structure_ids=df_test.formula)
        data_to_predict.featurize()
        data_to_predict.df_featurized.to_csv(os.path.join(file_path, f"df_pred_featurized_{key}.csv"))

    else:
        df_pred_featurized = pd.read_csv(os.path.join(file_path, f"df_pred_featurized_{key}.csv"), index_col = 0)
        data_to_predict = MODData(materials = df_test["composition"],
                                  df_featurized = df_pred_featurized,
                                  structure_ids=df_pred_featurized.index)
    df_predictions = model.predict(data_to_predict)
df_test_pred = df_test.merge(df_predictions, how = 'left', left_on = "formula", right_index = True, suffixes=('_true', '_pred'))
df_test_pred.to_csv(os.path.join(file_path, target_property, f"test_pred_epch_{epochs}.csv"))   
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
df_mae_all.to_csv(os.path.join(file_path, f"mae_all_epch_{epochs}.csv"))
feature_grp_str = json.dumps(feature_grp)
with open(os.path.join(file_path, "feature_grp.json"), 'w') as json_file:
    json_file.write(feature_grp_str)
print(feature_grp)
    