import json
import os

from azureml.core.workspace import Workspace

print("Starting the automl_driver setup...")
script_directory = None

import sys
import traceback

try:
    from azureml.train.automl._remote_script import setup_wrapper
    from azureml.train.automl._remote_script import driver_wrapper
except Exception as e:
    print("v2 driver import failed with exception: {}. Falling back to v1 driver.".format(e))
    traceback.print_exc()
    import importlib
    import inspect
    import logging
    import os
    import sys
    import time

    from automl.client.core.common import utilities
    from azureml.core.experiment import Experiment
    from azureml.core.run import Run
    from azureml.train.automl import automl
    from azureml.train.automl import fit_pipeline
    from azureml.train.automl._automl_settings import _AutoMLSettings

    try:
        from azureml.train.automl._cachestore import _CacheStore
        from azureml.train.automl._preprocessorcontexts import (RawDataContext,
                                                            TransformedDataContext)
        from azureml.train.automl._transform_data import _transform_data
        sdk_has_cache_capability = True
    except ImportError:
        sdk_has_cache_capability = False

    try:
        from azureml.train.automl.utilities import _validate_data_splits
        sdk_has_validate_data_splits = True
    except ImportError:
        sdk_has_validate_data_splits = False

    try:
        # Works only for azureml-train-automl>1.0.10
        from automl.client.core.common.training_utilities import validate_training_data, check_x_y
    except:
        from azureml.train.automl.utilities import _validate_training_data as validate_training_data, _check_x_y as check_x_y

    try:
        from automl.client.core.common.training_utilities import validate_training_data_dict
        sdk_has_validate_data_dict = True
    except:
        sdk_has_validate_data_dict = False


    try:
        from automl.client.core.common.logging_utilities import log_traceback
    except ImportError:
        def log_traceback(exception, logger, **kwargs):
            """Do nothing if not imported."""
            pass

    # holding these strings here to identify the exception that created by this script
    try:
        from  automl.client.core.common.exceptions import ErrorTypes
    except ImportError:
        class ErrorTypes:
            """Possible types of errors."""

            User = 'User'
            Service = 'Service'
            Client = 'Client'
            Unclassified = 'Unclassified'
            All = {User, Service, Client, Unclassified}

    def _get_auto_cv(X, y, X_valid, y_valid, cv_splits_indices, automl_settings_obj, logger):
        if hasattr(automl_settings_obj, "rule_based_validation"):
            return automl_settings_obj.rule_based_validation(
                X, y, X_valid, y_valid, cv_splits_indices,
                logger=logger)
        else:
            logger.info("SDK has no auto cv capability.")
            return X, y, X_valid, y_valid

    def _get_auto_cv_dict(input_dict, automl_settings_obj, logger):
        input_dict['X'], input_dict['y'], input_dict['X_valid'], input_dict['y_valid'] = _get_auto_cv(
            input_dict.get('X'),
            input_dict.get('y'),
            input_dict.get('X_valid'),
            input_dict.get('y_valid'),
            input_dict.get('cv_splits_indices'),
            automl_settings_obj,
            logger=logger)
        return input_dict

    def _get_cv_from_transformed_data_context(transformed_data_context, logger):
        n_cv = None
        if transformed_data_context._on_demand_pickle_keys is None:
            n_cv = None
        else:
            n_cv = sum([1 if "cv" in key else 0 for key in transformed_data_context._on_demand_pickle_keys])
        logger.info("The cv got from transformed_data_context is {}.".format(n_cv))
        return n_cv

    def _get_data_from_dataprep(dataprep_json, automl_settings_obj, logger):
        current_run = Run.get_submitted_run()
        parent_run_id = _get_parent_run_id(current_run._run_id)
        print("[ParentRunId:{}]: Start getting data using dataprep.".format(parent_run_id))
        logger.info("[ParentRunId:{}]: Start getting data using dataprep.".format(parent_run_id))
        try:
            import azureml.train.automl._dataprep_utilities as dataprep_utilities
        except Exception as e:
            e.error_type = ErrorTypes.Unclassified
            log_traceback(e, logger)
            logger.error(e)
            raise e

        fit_iteration_parameters_dict = dict()

        class RetrieveNumpyArrayError(Exception):
            def __init__(self):
                super().__init__()

        try:
            print("Resolving Dataflows...")
            logger.info("Resolving Dataflows...")
            dataprep_json_obj = json.loads(dataprep_json)
            if 'activities' in dataprep_json_obj: # json is serialized dataflows
                dataflow_dict = dataprep_utilities.load_dataflows_from_json(
                    dataprep_json)
                for k in ['X', 'X_valid', 'sample_weight', 'sample_weight_valid']:
                    fit_iteration_parameters_dict[k] = dataprep_utilities.try_retrieve_pandas_dataframe(dataflow_dict.get(k))
                for k in ['y', 'y_valid']:
                    try:
                        fit_iteration_parameters_dict[k] = dataprep_utilities.try_retrieve_numpy_array(dataflow_dict.get(k))
                    except IndexError:
                        raise RetrieveNumpyArrayError()

                cv_splits_dataflows = []
                i = 0
                while 'cv_splits_indices_{0}'.format(i) in dataflow_dict:
                    cv_splits_dataflows.append(
                        dataflow_dict['cv_splits_indices_{0}'.format(i)])
                    i = i + 1
                fit_iteration_parameters_dict['cv_splits_indices'] = None if len(cv_splits_dataflows) == 0 \
                    else dataprep_utilities.try_resolve_cv_splits_indices(cv_splits_dataflows)
            else: # json is dataprep options
                print('Creating Dataflow from options...\r\nOptions:')
                logger.info('Creating Dataflow from options...')
                print(dataprep_json_obj)
                datastore_name = dataprep_json_obj['datastoreName'] # mandatory
                data_path = dataprep_json_obj['dataPath'] # mandatory
                label_column = dataprep_json_obj['label'] # mandatory
                separator = dataprep_json_obj.get('columnSeparator', ',')
                header = dataprep_json_obj.get('promoteHeader', True)
                encoding = dataprep_json_obj.get('encoding', None)
                quoting = dataprep_json_obj.get('ignoreNewlineInQuotes', False)
                skip_rows = dataprep_json_obj.get('skipRows', 0)
                feature_columns = dataprep_json_obj.get('features', [])

                from azureml.core import Datastore
                import azureml.dataprep as dprep
                if header:
                    header = dprep.PromoteHeadersMode.CONSTANTGROUPED
                else:
                    header = dprep.PromoteHeadersMode.NONE
                try:
                    encoding = dprep.FileEncoding[encoding]
                except:
                    encoding = dprep.FileEncoding.UTF8

                ws = Run.get_context().experiment.workspace
                datastore = Datastore(ws, datastore_name)
                dflow = dprep.read_csv(path=datastore.path(data_path),
                                        separator=separator,
                                        header=header,
                                        encoding=encoding,
                                        quoting=quoting,
                                        skip_rows=skip_rows)

                if len(feature_columns) == 0:
                    X = dflow.drop_columns(label_column)
                else:
                    X = dflow.keep_columns(feature_columns)

                print('Inferring types for feature columns...')
                logger.info('Inferring types for feature columns...')
                sct = X.builders.set_column_types()
                sct.learn()
                sct.ambiguous_date_conversions_drop()
                X = sct.to_dataflow()

                y = dflow.keep_columns(label_column)
                if automl_settings_obj.task_type.lower() == 'regression':
                    y = y.to_number(label_column)

                print('X:')
                print(X)
                logger.info('X:')
                logger.info(X)

                print('y:')
                print(y)
                logger.info('y:')
                logger.info(y)

                try:
                    from azureml.train.automl._dataprep_utilities import try_retrieve_pandas_dataframe_adb
                    _X = try_retrieve_pandas_dataframe_adb(X)
                    fit_iteration_parameters_dict['X'] = _X.values
                    fit_iteration_parameters_dict['x_raw_column_names'] = _X.columns.values
                except ImportError:
                    logger.info("SDK version does not support column names extraction, fallback to old path")
                    fit_iteration_parameters_dict['X'] = dataprep_utilities.try_retrieve_pandas_dataframe(X)

                try:
                    fit_iteration_parameters_dict['y'] = dataprep_utilities.try_retrieve_numpy_array(y)
                except IndexError:
                    raise RetrieveNumpyArrayError()

            logger.info("Finish getting data using dataprep.")
            return fit_iteration_parameters_dict
        except Exception as e:
            print("[ParentRunId:{0}]: Error from resolving Dataflows: {1} {2}".format(parent_run_id, e.__class__, e))
            logger.error("[ParentRunId:{0}]: Error from resolving Dataflows: {1} {2}".format(parent_run_id, e.__class__, e))
            if isinstance(e, RetrieveNumpyArrayError):
                logger.debug("Label column (y) does not exist in user's data.")
                e.error_type = ErrorTypes.User
            elif "The provided path is not valid." in str(e):
                logger.debug("User's data is not accessible from remote run.")
                e.error_type = ErrorTypes.User
            elif "Required secrets are missing. Please call use_secrets to register the missing secrets." in str(e):
                logger.debug("User should use Datastore to data that requires secrets.")
                e.error_type = ErrorTypes.User
            else:
                e.error_type = ErrorTypes.Client
            log_traceback(e, logger)
            raise RuntimeError("Error during extracting Dataflows")

    def _init_logger(automl_settings_obj=None):
        sdk_has_custom_dimension_logger = False
        try:
            from azureml.telemetry import set_diagnostics_collection
            if automl_settings_obj is not None:
                set_diagnostics_collection(send_diagnostics=automl_settings_obj.send_telemetry,
                                           verbosity=automl_settings_obj.telemetry_verbosity)
        except:
            print("set_diagnostics_collection failed.")

        try:
            from azureml.train.automl._logging import get_logger
            if "automl_settings" in inspect.getcallargs(get_logger, log_file_name="AutoML_remote.log"):
                logger = get_logger(log_file_name="AutoML_remote.log", automl_settings=automl_settings_obj)
                sdk_has_custom_dimension_logger = True
            else:
                logger = get_logger(log_file_name="AutoML_remote.log")
                sdk_has_custom_dimension_logger = False
            logger.info("sdk_has_custom_dimension_logger {}.".format(sdk_has_custom_dimension_logger))
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        logger.info("Init logger successfully with automl_settings {}.".format(automl_settings_obj))
        try:
            from automl.client.core.common.utilities import get_sdk_dependencies
            logger.info(get_sdk_dependencies())
        except Exception as e:
            pass
        return logger, sdk_has_custom_dimension_logger

    def _init_directory(directory, logger):
        logger.info("Start init directory.")
        if(directory == None):
            directory = os.path.dirname(__file__)

        logger.info("Adding directory to system path.")
        sys.path.append(directory)

        # create the outputs folder
        logger.info("Creating output folder.")
        os.makedirs('./outputs', exist_ok=True)
        print("create output folder")
        logger.info("Finished init directory.")
        return directory

    def _get_parent_run_id(run_id):
        split = run_id.split("_")
        if len(split) > 2:
            split.pop()
        else:
            return run_id

        parent_run_id = '_'.join(str(e) for e in split)
        return parent_run_id

    def _load_data_from_user_script(script_directory, entry_point, logger):
        #  Load user script to get access to GetData function
        logger.info("Loading data using user script.")
        try:
            from azureml.train.automl import extract_user_data
        except Exception as e:
            logger.warning(e)

        module_name = None
        if (entry_point.endswith('.py')):
            module_name = entry_point[:-3]

        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(script_directory, entry_point))
        module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_obj)
        # print("Extracting user Data from {0}".format(module_name))

        fit_iteration_parameters_dict = dict()
        try:
            output_dict = extract_user_data(module_obj)
            for k, v in output_dict.items():
                fit_iteration_parameters_dict[k] = v
        except Exception as e:
            logger.warning("Meeting exceptions using user script {}.".format(e))
            fit_iteration_parameters_dict['X'], fit_iteration_parameters_dict['y'] = module_obj.get_data()

        return fit_iteration_parameters_dict

    def _prepare_data(dataprep_json, automl_settings_obj, script_directory, entry_point, logger):
        if dataprep_json:
            return _get_data_from_dataprep(dataprep_json, automl_settings_obj, logger)
        else:
            return _load_data_from_user_script(script_directory, entry_point, logger)

    def _get_transformed_data_context(X, y, X_valid, y_valid,
                                     sample_weight, sample_weight_valid,
                                     x_raw_column_names, cv_splits_indices,
                                     data_store, run_target,
                                     automl_settings_obj, parent_run_id, logger,
                                     raw_data_context=None):
        logger.info("Getting transformed data context.")
        if raw_data_context is None:
            logger.info("raw_data_context is None, creating a new one.")
            raw_data_context = RawDataContext(task_type=automl_settings_obj.task_type,
                                              X=X,
                                              y=y,
                                              X_valid=X_valid,
                                              y_valid=y_valid,
                                              sample_weight=sample_weight,
                                              sample_weight_valid=sample_weight_valid,
                                              x_raw_column_names=x_raw_column_names,
                                              lag_length=automl_settings_obj.lag_length,
                                              cv_splits_indices=cv_splits_indices,
                                              automl_settings_obj=automl_settings_obj,
                                              enable_cache=automl_settings_obj.enable_cache,
                                              data_store=data_store,
                                              run_target='remote',
                                              timeseries=automl_settings_obj.is_timeseries,
                                              timeseries_param_dict=utilities._get_ts_params_dict(automl_settings_obj)
                                              )

        transformed_data_context = _transform_data(raw_data_context=raw_data_context,
                                                   preprocess=automl_settings_obj.preprocess,
                                                   logger=logger,
                                                   run_id=parent_run_id)
        logger.info("Finished getting transformed data context.")

        return transformed_data_context

    def _set_problem_info_for_setup(fit_iteration_parameters_dict,
                                   automl_settings_obj, task_type, preprocess,
                                   enable_subsampling, num_iterations,
                                   logger):
        current_run = Run.get_submitted_run()
        logger.info("Start to set problem info for the setup for run id {}.".format(current_run._run_id))
        logger.info("Setup experiment.")
        try:
            experiment = current_run.experiment
            parent_run_id = _get_parent_run_id(current_run._run_id)
            data_store = experiment.workspace.get_default_datastore()
            found_data_store = True
            logger.info("Using data store.")
        except Exception as e:
            logger.warning("Getting data store, fallback to default {}".format(e))
            found_data_store = False

        logger.info("Caching supported {}.".format(sdk_has_cache_capability and found_data_store))
        print("caching supported {}".format(sdk_has_cache_capability and found_data_store))
        if sdk_has_validate_data_dict:
            # The newest version of validate_training_data_dict should contains check_x_y
            logger.info("Using validate_training_data_dict now.")
            validate_training_data_dict(data_dict=fit_iteration_parameters_dict, automl_settings=automl_settings_obj)
        else:
            logger.info("Using validate_training_data now.")
            validate_training_data(X=fit_iteration_parameters_dict.get('X'),
                                      y=fit_iteration_parameters_dict.get('y'),
                                      X_valid=fit_iteration_parameters_dict.get('X_valid'),
                                      y_valid=fit_iteration_parameters_dict.get('y_valid'),
                                      sample_weight=fit_iteration_parameters_dict.get('sample_weight'),
                                      sample_weight_valid=fit_iteration_parameters_dict.get('sample_weight_valid'),
                                      cv_splits_indices=fit_iteration_parameters_dict.get('cv_splits_indices'),
                                      automl_settings=automl_settings_obj)
            check_x_y(fit_iteration_parameters_dict.get('X'), fit_iteration_parameters_dict.get('y'), automl_settings_obj)
        if sdk_has_cache_capability and found_data_store:
            data_splits_validated = True
            try:
                start = time.time()
                transformed_data_context = _get_transformed_data_context(
                    X=fit_iteration_parameters_dict.get('X'),
                    y=fit_iteration_parameters_dict.get('y'),
                    X_valid=fit_iteration_parameters_dict.get('X_valid'),
                    y_valid=fit_iteration_parameters_dict.get('y_valid'),
                    sample_weight=fit_iteration_parameters_dict.get('sample_weight'),
                    sample_weight_valid=fit_iteration_parameters_dict.get('sample_weight_valid'),
                    x_raw_column_names=fit_iteration_parameters_dict.get('x_raw_column_names'),
                    cv_splits_indices=fit_iteration_parameters_dict.get('cv_splits_indices'),
                    automl_settings_obj=automl_settings_obj,
                    data_store=data_store,
                    run_target='remote',
                    parent_run_id=parent_run_id,
                    logger=logger
                )
                end = time.time()
                print("time taken for transform {}".format(end-start))
                logger.info("time taken for transform {}".format(end-start))
                if sdk_has_validate_data_splits:
                    try:
                        logger.info("Validating data splits now.")
                        _validate_data_splits(X=transformed_data_context.X,
                                              y=transformed_data_context.y,
                                              X_valid=transformed_data_context.X_valid,
                                              y_valid=transformed_data_context.y_valid,
                                              cv_splits=transformed_data_context.cv_splits,
                                              automl_settings=automl_settings_obj)
                        data_splits_validated = True
                    except Exception as data_split_exception:
                        data_splits_validated = False
                        logger.error("Meeting validation errors {}.".format(data_split_exception))
                        log_traceback(data_split_exception, logger)
                        raise data_split_exception
                logger.info("Start setting problem info.")
                automl.set_problem_info(transformed_data_context.X, transformed_data_context.y,
                                        automl_settings_obj.task_type,
                                        current_run=current_run,
                                        preprocess=automl_settings_obj.preprocess,
                                        lag_length=automl_settings_obj.lag_length,
                                        transformed_data_context=transformed_data_context,
                                        enable_cache=automl_settings_obj.enable_cache,
                                        subsampling=enable_subsampling)
            except Exception as e:
                if sdk_has_validate_data_splits and not data_splits_validated:
                    logger.error("sdk_has_validate_data_splits is True and data_splits_validated is False {}.".format(e))
                    log_traceback(e, logger)
                    raise e
                else:
                    logger.warning("Setup failed, fall back to old model {}".format(e))
                    print("Setup failed, fall back to old model {}".format(e))
                    automl.set_problem_info(
                        X=fit_iteration_parameters_dict.get('X'),
                        y=fit_iteration_parameters_dict.get('y'),
                        task_type=task_type, current_run=current_run,
                        preprocess=preprocess, subsampling=enable_subsampling
                    )
        else:
            logger.info("Start setting problem info using old model.")
            if sdk_has_validate_data_splits:
                _validate_data_splits(X=fit_iteration_parameters_dict.get('X'),
                                      y=fit_iteration_parameters_dict.get('y'),
                                      X_valid=fit_iteration_parameters_dict.get('X_valid'),
                                      y_valid=fit_iteration_parameters_dict.get('y_valid'),
                                      cv_splits=fit_iteration_parameters_dict.get('cv_splits_indices'),
                                      automl_settings=automl_settings_obj)
            automl.set_problem_info(
                X=fit_iteration_parameters_dict.get('X'),
                y=fit_iteration_parameters_dict.get('y'),
                task_type=task_type, current_run=current_run,
                preprocess=preprocess, subsampling=enable_subsampling
            )

    def _post_setup(logger):
        logger.info("Setup run completed successfully!")
        print("Setup run completed successfully!")

    def _get_automl_settings(automl_settings, logger):
        automl_settings_obj = None
        current_run = Run.get_submitted_run()
        found_data_store = False
        data_store = None

        start = time.time()

        try:
            experiment = current_run.experiment

            parent_run_id = _get_parent_run_id(current_run._run_id)
            print("parent run id {}".format(parent_run_id))

            automl_settings_obj = _AutoMLSettings.from_string_or_dict(automl_settings)
            data_store = experiment.workspace.get_default_datastore()
            found_data_store = True
        except Exception as e:
            logger.warning("getting data store, fallback to default {}".format(e))
            print("failed to get default data store  {}".format(e))
            found_data_store = False

        end = time.time()
        print("Caching supported {}, time taken for get default DS {}".format(sdk_has_cache_capability and found_data_store, (end - start)))

        return automl_settings_obj, found_data_store, data_store

    def _load_transformed_data_context_from_cache(automl_settings_obj, parent_run_id,
                                                 found_data_store, data_store,
                                                 logger):
        logger.info("Loading the data from datastore.")
        transformed_data_context = None
        if sdk_has_cache_capability and automl_settings_obj is not None and automl_settings_obj.enable_cache and \
                automl_settings_obj.preprocess and found_data_store:

            try:
                start = time.time()
                transformed_data_context = TransformedDataContext(X={},
                                                                  run_id=parent_run_id,
                                                                  run_targets='remote',
                                                                  logger=logger,
                                                                  enable_cache=True,
                                                                  data_store=data_store)
                transformed_data_context._load_from_cache()
                end = time.time()
                logger.info("Time taken for loading from cache {}.".format(end-start))
                print("Time taken for loading from cache {}.".format(end-start))

            except Exception as e:
                logger.warning("Error while loading from cache, defaulting to redo {}".format(e))
                transformed_data_context = None
        return transformed_data_context

    def _start_run(automl_settings_obj, run_id, training_percent, iteration,
                  pipeline_spec, pipeline_id,
                  dataprep_json, script_directory,
                  entry_point,
                  logger,
                  transformed_data_context=None):
        logger.info("Starting the run.")
        if transformed_data_context is None:
            logger.info("transformed_data_context is None, loading data now.")
            fit_iteration_parameters_dict = _prepare_data(
                dataprep_json=dataprep_json,
                automl_settings_obj=automl_settings_obj,
                script_directory=script_directory,
                entry_point=entry_point,
                logger=logger
            )

            fit_iteration_parameters_dict = _get_auto_cv_dict(fit_iteration_parameters_dict, automl_settings_obj, logger)

            result = fit_pipeline(
                pipeline_script=pipeline_spec,
                automl_settings=automl_settings_obj,
                run_id=run_id,
                fit_iteration_parameters_dict = fit_iteration_parameters_dict,
                train_frac=training_percent/100,
                iteration=iteration,
                pipeline_id=pipeline_id,
                remote=True,
                child_run_metrics=Run.get_context(_batch_upload_metrics=False),
                logger=logger)
        else:
            if automl_settings_obj.n_cross_validations is None and transformed_data_context.X_valid is None:
                automl_settings_obj.n_cross_validations = _get_cv_from_transformed_data_context(
                    transformed_data_context, logger)
            result = fit_pipeline(
                pipeline_script=pipeline_spec,
                automl_settings=automl_settings_obj,
                run_id=run_id,
                train_frac=training_percent/100,
                iteration=iteration,
                pipeline_id=pipeline_id,
                remote=True,
                child_run_metrics=Run.get_context(_batch_upload_metrics=False),
                logger=logger,
                transformed_data_context=transformed_data_context)
        logger.info("Run finished.")
        return result

    def _post_run(result, run_id, automl_settings, logger):
        print("for Run Id : ", run_id)
        print("result : ", result)
        if len(result['errors']) > 0:
            err_type = next(iter(result['errors']))
            inner_ex = result['errors'][err_type]['exception']
            inner_ex.error_type = ErrorTypes.Client
            log_traceback(inner_ex, logger)
            raise RuntimeError(inner_ex) from inner_ex

        score = result[automl_settings['primary_metric']]
        duration = result['fit_time']
        print("Score : ", score)
        print("Duration : ", duration)
        print("Childrun completed successfully!")
        logger.info("Childrun completed successfully!")

    def driver_wrapper(
        script_directory, automl_settings, run_id, training_percent,
        iteration, pipeline_spec, pipeline_id, dataprep_json, entry_point,
        **kwargs
    ):
        automl_settings_obj = _AutoMLSettings.from_string_or_dict(automl_settings)
        logger, sdk_has_custom_dimension_logger = _init_logger(automl_settings_obj)
        if sdk_has_custom_dimension_logger:
            logger.update_default_properties({
                "parent_run_id": _get_parent_run_id(run_id),
                "child_run_id": run_id
            })
        logger.info("[RunId:{}]: remote automl driver begins.".format(run_id))

        try:
            script_directory = _init_directory(directory=script_directory, logger=logger)

            automl_settings_obj, found_data_store, data_store = _get_automl_settings(
                automl_settings=automl_settings, logger=logger)

            transformed_data_context = _load_transformed_data_context_from_cache(
                automl_settings_obj=automl_settings_obj,
                parent_run_id=_get_parent_run_id(run_id),
                found_data_store=found_data_store,
                data_store=data_store,
                logger=logger
            )
            result = _start_run(automl_settings_obj=automl_settings_obj,
                            run_id=run_id,
                            training_percent=training_percent,
                            iteration=iteration,
                            pipeline_spec=pipeline_spec,
                            pipeline_id=pipeline_id,
                            dataprep_json=dataprep_json,
                            script_directory=script_directory,
                            entry_point=entry_point,
                            logger=logger,
                            transformed_data_context=transformed_data_context)
            _post_run(result=result, run_id=run_id, automl_settings=automl_settings, logger=logger)
        except Exception as e:
            logger.error("driver_wrapper meets exceptions. {}".format(e))
            log_traceback(e, logger)
            raise Exception(e)

        logger.info("[RunId:{}]: remote automl driver finishes.".format(run_id))
        return result

    def setup_wrapper(
        script_directory, dataprep_json, entry_point, automl_settings, task_type,
        preprocess, enable_subsampling, num_iterations,
        **kwargs
    ):
        automl_settings_obj = _AutoMLSettings.from_string_or_dict(automl_settings)

        logger, sdk_has_custom_dimension_logger = _init_logger(automl_settings_obj)
        try:
            child_run_id = Run.get_submitted_run()._run_id
            parent_run_id = _get_parent_run_id(child_run_id)
            if sdk_has_custom_dimension_logger:
                logger.update_default_properties({
                    "parent_run_id": parent_run_id,
                    "child_run_id": child_run_id
                })
            logger.info("[ParentRunId:{}]: remote setup script begins.".format(parent_run_id))
            script_directory = _init_directory(directory=script_directory, logger=logger)

            logger.info("Preparing data for set problem info now.")

            fit_iteration_parameters_dict = _prepare_data(
                dataprep_json=dataprep_json,
                automl_settings_obj=automl_settings_obj,
                script_directory=script_directory,
                entry_point=entry_point,
                logger=logger
            )
            fit_iteration_parameters_dict = _get_auto_cv_dict(fit_iteration_parameters_dict, automl_settings_obj, logger)

            print("Setting Problem Info now.")
            _set_problem_info_for_setup(
                fit_iteration_parameters_dict=fit_iteration_parameters_dict,
                automl_settings_obj=automl_settings_obj,
                task_type=task_type,
                preprocess=preprocess,
                enable_subsampling=enable_subsampling,
                num_iterations=num_iterations,
                logger=logger)
        except Exception as e:
            logger.error("setup_wrapper meets exceptions. {}".format(e))
            log_traceback(e, logger)
            raise Exception(e)

        _post_setup(logger=logger)
        logger.info("[ParentRunId:{}]: remote setup script finishes.".format(parent_run_id))
        return # PLACEHOLDER for RemoteScript helper functions

workspace_name = "main" # PLACEHOLDER
subscription_id = "d49067c6-0d06-4a1d-84e1-f0904327214e" # PLACEHOLDER
resource_group = "ezio" # PLACEHOLDER
experiment_name = "mslearn-bike-rental" # PLACEHOLDER
iteration = "0"  # PLACEHOLDER
run_id = "AutoML_27220109-27e7-4f3b-882c-3b786c7f6b08_0"  # PLACEHOLDER
entry_point = "get_data.py" # PLACEHOLDER
automl_settings = {'enable_early_stopping':True,'enable_ensembling':True,'enable_stack_ensembling':True,'ensemble_iterations':15,'enable_onnx_compatible_models':False,'save_mlflow':True,'max_cores_per_iteration':-1,'send_telemetry':True,'blacklist_algos':['TensorFlowDNN','TensorFlowLinearRegressor'],'whitelist_models':['RandomForest','LightGBM'],'compute_target':'Alexios','enable_dnn':False,'enable_code_generation':True,'experiment_exit_score':0.085,'experiment_timeout_minutes':30,'featurization':{'_blocked_transformers':[],'_column_purposes':{},'_transformer_params':{'Imputer':[]},'_drop_columns':None},'hyperdrive_config':None,'grain_column_names':None,'is_timeseries':False,'iteration_timeout_minutes':30,'max_concurrent_iterations':2,'metric_operation':'minimize','model_explainability':True,'n_cross_validations':None,'name':'mslearn-bike-rental','path':'./sample_projects/mslearn-bike-rental','primary_metric':'normalized_root_mean_squared_error','region':'centralindia','resource_group':'ezio','subscription_id':'d49067c6-0d06-4a1d-84e1-f0904327214e','task_type':'regression','validation_size':None,'test_size':None,'vm_type':'STANDARD_DS11_V2','workspace_name':'main','label_column_name':'rentals','enable_batch_run':True,'dataset_id':'37bbc6d9-9f05-4528-99cd-efc508cc2aa9'} # PLACEHOLDER
pipeline_spec = "{\"objects\":[{\"spec_class\":\"preproc\",\"class_name\":\"MaxAbsScaler\",\"module\":\"sklearn.preprocessing\",\"param_args\":[],\"param_kwargs\":{},\"prepared_kwargs\":{}},{\"spec_class\":\"sklearn\",\"class_name\":\"LightGBMRegressor\",\"module\":\"automl.client.core.common.model_wrappers\",\"param_args\":[],\"param_kwargs\":{\"min_data_in_leaf\":20},\"prepared_kwargs\":{}}],\"pipeline_id\":\"faf12f74cf9bbd358ca5525682c5030d36f7be7c\",\"module\":\"sklearn.pipeline\",\"class_name\":\"Pipeline\"}" # PLACEHOLDER
pipeline_id = "faf12f74cf9bbd358ca5525682c5030d36f7be7c" # PLACEHOLDER
dataprep_json = "{\"datasets\":0,\"training_data\":{\"datasetId\":\"37bbc6d9-9f05-4528-99cd-efc508cc2aa9\"}}" # PLACEHOLDER
mltable_data_json = None # PLACEHOLDER
training_percent = 100 # PLACEHOLDER
enable_streaming = False # PLACEHOLDER

if enable_streaming is not None:
    print("Set enable_streaming flag to", enable_streaming)
    automl_settings['enable_streaming']=enable_streaming

print("run_id in the real script: ", run_id)
project_dir = "/tmp/azureml_runs/" + run_id


def new_run():
    global script_directory
    result = driver_wrapper(
        script_directory=script_directory,
        automl_settings=automl_settings,
        run_id=run_id,
        training_percent=training_percent,
        iteration=iteration,
        pipeline_spec=pipeline_spec,
        pipeline_id=pipeline_id,
        dataprep_json=dataprep_json,
        mltable_data_json=mltable_data_json,
        entry_point=entry_point
    )
    return result

if __name__ == '__main__':
    result = new_run()
    print(result)
