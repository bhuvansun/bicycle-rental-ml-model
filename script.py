import numpy
import numpy as np
import pandas as pd
import pickle
import argparse

from azureml.training.tabular._diagnostics import logging_utilities


def setup_instrumentation(automl_run_id):
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Add logging to STDOUT
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        # Add telemetry logging with formatter to strip identifying info
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)

        # Attach run IDs to logging info for correlation if running inside AzureML
        try:
            run = Run.get_context()
            return logging.LoggerAdapter(logger, extra={
                "properties": {
                    "codegen_run_id": run.id,
                    "automl_run_id": automl_run_id
                }
            })
        except Exception:
            pass
    except Exception:
        pass

    return logger


automl_run_id = 'AutoML_27220109-27e7-4f3b-882c-3b786c7f6b08_0'
logger = setup_instrumentation(automl_run_id)


def split_dataset(X, y, weights, split_ratio, should_stratify):
    '''
    Splits the dataset into a training and testing set.

    Splits the dataset using the given split ratio. The default ratio given is 0.25 but can be
    changed in the main function. If should_stratify is true the data will be split in a stratified
    way, meaning that each new set will have the same distribution of the target value as the
    original dataset. should_stratify is true for a classification run, false otherwise.
    '''
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_id):
    '''
    Loads the previously used dataset.

    It assumes that the script is run in an AzureML command job under the same workspace as the original experiment.
    '''

    from azureml.core.dataset import Dataset
    from azureml.core.run import Run

    logger.info("Running get_training_dataset")
    ws = Run.get_context().experiment.workspace
    dataset = Dataset.get_by_id(workspace=ws, id=dataset_id)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    '''
    Prepares data for training.

    Cleans the data, splits out the feature and sample weight columns and prepares the data for use in training.
    This function can vary depending on the type of dataset and the experiment task type: classification,
    regression, or time-series forecasting.
    '''

    from azureml.training.tabular.preprocessing import data_cleaning

    logger.info("Running prepare_data")
    label_column_name = 'rentals'

    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
                                                                 is_timeseries=False, target_column=label_column_name)

    return X, y, sample_weights


def get_mapper_0(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from azureml.training.tabular.featurization.utilities import wrap_in_list
    from numpy import uint8
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features

    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': CountVectorizer,
                'analyzer': 'word',
                'binary': True,
                'decode_error': 'strict',
                'dtype': numpy.uint8,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 1.0,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (1, 1),
                'preprocessor': None,
                'stop_words': None,
                'strip_accents': None,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': wrap_in_list,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)

    return mapper


def get_mapper_1(column_names):
    from azureml.training.tabular.featurization.categorical.cat_imputer import CatImputer
    from azureml.training.tabular.featurization.categorical.labelencoder_transformer import LabelEncoderTransformer
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features

    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': CatImputer,
                'copy': True,
            },
            {
                'class': StringCastTransformer,
            },
            {
                'class': LabelEncoderTransformer,
                'hashing_seed_val': 314489979,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)

    return mapper


def get_mapper_2(column_names):
    from sklearn.impute import SimpleImputer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features

    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': SimpleImputer,
                'add_indicator': False,
                'copy': True,
                'fill_value': None,
                'missing_values': numpy.nan,
                'strategy': 'mean',
                'verbose': 0,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)

    return mapper


def generate_data_transformation_config():
    '''
    Specifies the featurization step in the final scikit-learn pipeline.

    If you have many columns that need to have the same featurization/transformation applied (for example,
    50 columns in several column groups), these columns are handled by grouping based on type. Each column
    group then has a unique mapper applied to all columns in the group.
    '''
    from sklearn.pipeline import FeatureUnion

    column_group_1 = ['year', 'holiday', 'workingday']

    column_group_2 = [['temp'], ['atemp'], ['hum'], ['windspeed']]

    column_group_0 = ['day', 'mnth', 'season', 'weekday', 'weathersit']

    feature_union = FeatureUnion([
        ('mapper_0', get_mapper_0(column_group_0)),
        ('mapper_1', get_mapper_1(column_group_1)),
        ('mapper_2', get_mapper_2(column_group_2)),
    ])
    return feature_union


def generate_preprocessor_config():
    '''
    Specifies a preprocessing step to be done after featurization in the final scikit-learn pipeline.

    Normally, this preprocessing step only consists of data standardization/normalization that is
    accomplished with sklearn.preprocessing. Automated ML only specifies a preprocessing step for
    non-ensemble classification and regression models.
    '''
    from sklearn.preprocessing import MaxAbsScaler

    preproc = MaxAbsScaler(
        copy=True
    )

    return preproc


def generate_algorithm_config():
    '''
    Specifies the actual algorithm and hyperparameters for training the model.

    It is the last stage of the final scikit-learn pipeline. For ensemble models, generate_preprocessor_config_N()
    (if needed) and generate_algorithm_config_N() are defined for each learner in the ensemble model,
    where N represents the placement of each learner in the ensemble model's list. For stack ensemble
    models, the meta learner generate_algorithm_config_meta() is defined.
    '''
    from lightgbm.sklearn import LGBMRegressor

    algorithm = LGBMRegressor(
        boosting_type='gbdt',
        class_weight=None,
        colsample_bytree=1.0,
        importance_type='split',
        learning_rate=0.1,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=0.001,
        min_split_gain=0.0,
        n_estimators=100,
        n_jobs=-1,
        num_leaves=31,
        objective=None,
        random_state=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        silent=True,
        subsample=1.0,
        subsample_for_bin=200000,
        subsample_freq=0,
        verbose=-1
    )

    return algorithm


def build_model_pipeline():
    '''
    Defines the scikit-learn pipeline steps.
    '''
    from sklearn.pipeline import Pipeline

    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config()),
            ('preproc', generate_preprocessor_config()),
            ('model', generate_algorithm_config()),
        ]
    )

    return pipeline


def train_model(X, y, sample_weights=None, transformer=None):
    '''
    Calls the fit() method to train the model.

    The return value is the model fitted/trained on the input data.
    '''

    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()

    model = model_pipeline.fit(X, y)
    return model


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    '''
    Calculates the metrics that can be used to evaluate the model's performance.

    Metrics calculated vary depending on the experiment type. Classification, regression and time-series
    forecasting jobs each have their own set of metrics that are calculated.'''

    from azureml.training.tabular.preprocessing._dataset_binning import make_dataset_bins
    from azureml.training.tabular.score.scoring import score_regression

    y_pred = model.predict(X_test)
    y_min = np.min(y)
    y_max = np.max(y)
    y_std = np.std(y)

    bin_info = make_dataset_bins(X_test.shape[0], y_test)
    metrics = score_regression(
        y_test, y_pred, get_metrics_names(), y_max, y_min, y_std, sample_weights, bin_info)
    return metrics


def get_metrics_names():
    metrics_names = [
        'normalized_root_mean_squared_log_error',
        'normalized_median_absolute_error',
        'residuals',
        'root_mean_squared_error',
        'mean_absolute_error',
        'normalized_root_mean_squared_error',
        'predicted_true',
        'mean_absolute_percentage_error',
        'explained_variance',
        'median_absolute_error',
        'spearman_correlation',
        'root_mean_squared_log_error',
        'r2_score',
        'normalized_mean_absolute_error',
    ]
    return metrics_names


def get_metrics_log_methods():
    metrics_log_methods = {
        'normalized_root_mean_squared_log_error': 'log',
        'normalized_median_absolute_error': 'log',
        'residuals': 'log_residuals',
        'root_mean_squared_error': 'log',
        'mean_absolute_error': 'log',
        'normalized_root_mean_squared_error': 'log',
        'predicted_true': 'log_predictions',
        'mean_absolute_percentage_error': 'log',
        'explained_variance': 'log',
        'median_absolute_error': 'log',
        'spearman_correlation': 'log',
        'root_mean_squared_log_error': 'log',
        'r2_score': 'log',
        'normalized_mean_absolute_error': 'log',
    }
    return metrics_log_methods


def main(training_dataset_id=None):
    '''
    Runs all functions defined above.
    '''

    from azureml.automl.core.inference import inference
    from azureml.core.run import Run

    import mlflow

    # The following code is for when running this code as part of an AzureML script run.
    run = Run.get_context()

    df = get_training_dataset(training_dataset_id)
    X, y, sample_weights = prepare_data(df)
    split_ratio = 0.25
    (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y,
                                                                                                       sample_weights,
                                                                                                       split_ratio,
                                                                                                       should_stratify=False)
    model = train_model(X_train, y_train, sample_weights_train)

    metrics = calculate_metrics(model, X, y, sample_weights, X_test=X_valid, y_test=y_valid)
    metrics_log_methods = get_metrics_log_methods()
    print(metrics)
    for metric in metrics:
        if metrics_log_methods[metric] == 'None':
            logger.warning("Unsupported non-scalar metric {}. Will not log.".format(metric))
        elif metrics_log_methods[metric] == 'Skip':
            pass  # Forecasting non-scalar metrics and unsupported classification metrics are not logged
        else:
            getattr(run, metrics_log_methods[metric])(metric, metrics[metric])
    cd = inference.get_conda_deps_as_dict(True)

    # Saving ML model to outputs/.
    signature = mlflow.models.signature.infer_signature(X, y)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='outputs/',
        conda_env=cd,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

    run.upload_folder('outputs/', 'outputs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_id', type=str, default='37bbc6d9-9f05-4528-99cd-efc508cc2aa9',
                        help='Default training dataset id is populated from the parent run')
    args = parser.parse_args()

    try:
        main(args.training_dataset_id)
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise
