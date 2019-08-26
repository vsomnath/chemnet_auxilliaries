import numpy as np
import logging
import tensorflow as tf
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr=False
import os
from datetime import datetime as dt
import argparse

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()

import deepchem as dc
from deepchem.metrics import to_one_hot
from deepchem.models import ChemCeption
from deepchem.models.tensorgraph.optimizers import RMSProp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

loaders = {
      'bace_c': dc.molnet.load_bace_classification,
      'bace_r': dc.molnet.load_bace_regression,
      'bbbp': dc.molnet.load_bbbp,
      'clearance': dc.molnet.load_clearance,
      'clintox': dc.molnet.load_clintox,
      'delaney': dc.molnet.load_delaney,
      'hiv': dc.molnet.load_hiv,
      'hopv': dc.molnet.load_hopv,
      'kaggle': dc.molnet.load_kaggle,
      'kinase': dc.molnet.load_kinase,
      'lipo': dc.molnet.load_lipo,
      'muv': dc.molnet.load_muv,
      'nci': dc.molnet.load_nci,
      'pcba': dc.molnet.load_pcba,
      'pcba_146': dc.molnet.load_pcba_146,
      'pcba_2475': dc.molnet.load_pcba_2475,
      'ppb': dc.molnet.load_ppb,
      'qm7': dc.molnet.load_qm7_from_mat,
      'qm7b': dc.molnet.load_qm7b_from_mat,
      'qm8': dc.molnet.load_qm8,
      'qm9': dc.molnet.load_qm9,
      'sampl': dc.molnet.load_sampl,
      'sider': dc.molnet.load_sider,
      'thermosol': dc.molnet.load_thermosol,
      'tox21': dc.molnet.load_tox21,
      'toxcast': dc.molnet.load_toxcast}

classification_datasets = ['bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'pcba_146', 'pcba_2475', 'sider', 'tox21', 'toxcast']
regression_datasets = ['bace_r', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo', 'nci', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl', 'thermosol']

metric_types = {'bace_c': dc.metrics.roc_auc_score,
      'bace_r': dc.metrics.rms_score,
      'bbbp': dc.metrics.roc_auc_score,
      'clearance': dc.metrics.rms_score,
      'clintox': dc.metrics.roc_auc_score,
      'delaney': dc.metrics.mae_score,
      'hiv': dc.metrics.roc_auc_score,
      'hopv': dc.metrics.mae_score,
      'kaggle': dc.metrics.mae_score,
      'lipo': dc.metrics.mae_score,
      'muv': dc.metrics.roc_auc_score,
      'nci': dc.metrics.mae_score,
      'pcba': dc.metrics.prc_auc_score,
      'pcba_146': dc.metrics.prc_auc_score,
      'pcba_2475': dc.metrics.prc_auc_score,
      'ppb': dc.metrics.mae_score,
      'qm7': dc.metrics.mae_score,
      'qm7b': dc.metrics.mae_score,
      'sampl': dc.metrics.rms_score,
      'thermosol': dc.metrics.rms_score,
      'sider': dc.metrics.roc_auc_score,
      'tox21': dc.metrics.roc_auc_score,
      'toxcast': dc.metrics.roc_auc_score
      }

#loaders = {"tox21": dc.molnet.load_tox21, "hiv": dc.molnet.load_hiv, "sampl": dc.molnet.load_sampl, "bbbp": dc.molnet.load_bbbp}

def get_task_mode(dataset):
    if dataset in classification_datasets:
        return "classification"
    elif dataset in regression_datasets:
        return "regression"
    else:
        raise ValueError("Datasets {} does not have a mode defined".format(dataset))

def compute_loss_on_valid(valid, model, tasks, mode, verbose=True):
    loss_fn = model._loss_fn
    outputs = model.predict(valid, transformers=[])

    if mode == "classification":
        labels = to_one_hot(valid.y.flatten(), 2).reshape(-1, len(tasks), 2)
    else:
        labels = valid.y

    loss_tensor = loss_fn([outputs], [labels], weights=[valid.w])
    if tf.executing_eagerly():
        loss = loss_tensor.numpy()
    else:
        loss = model.session.run(loss_tensor)

    if verbose:
        logger.info("Computed loss on validation set: {}".format(loss))
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_spec", default="std", help="Image specification to load")
    parser.add_argument("--early_stopping_epoch", default=10, type=int, help="Number of epochs to test early stopping after")
    parser.add_argument("--use_augment", action='store_true', help="Whether to perform real-time augmentation")
    parser.add_argument("--base_filters", default=16, type=int, help="Number of base filters to use")
    parser.add_argument("--inception_per_block", default=3, type=int, help="Number of inception layers per block")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size used for training")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate used for fine-tuning")
    parser.add_argument("--restore_exp", help="Experiment to restore from")
    parser.add_argument("--train_all", action="store_true", help="Whether to only train all variables")
    parser.add_argument("--dataset", default="tox21", help="Dataset to train on")

    args = parser.parse_args()

    layers_per_block = args.inception_per_block
    inception_blocks = {"A": layers_per_block, "B": layers_per_block, "C": layers_per_block}

    mode = get_task_mode(args.dataset)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DIRNAME = os.path.join(os.environ.get("SCRATCH", ROOT_DIR), "deepchem-data")
    hparams_dir = "filters_{}_blocklayers_{}_imgspec_{}".format(args.base_filters, layers_per_block, args.img_spec)
    restore_dir = os.path.join(DIRNAME, "chemception", hparams_dir, "best-models", args.restore_exp + "/")
    logger.info("Restore dir: {}".format(restore_dir))

    load_fn = loaders[args.dataset]
    tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=DIRNAME, save_dir=DIRNAME, img_spec=args.img_spec, split="stratified")

    metric_type = metric_types[args.dataset]

    task_averager = np.mean
    if len(tasks) == 1:
        task_averager = None

    metric = dc.metrics.Metric(metric_type, task_averager=task_averager, mode=mode, verbose=False)
    train, valid, test = dataset

    # Setup directory for experiment
    exp_name = dt.now().strftime("%d-%m-%Y--%H-%M-%S")
    model_dir_1 = os.path.join(DIRNAME, args.dataset + "-finetune", "chemception", hparams_dir, exp_name)

    # Optimizer and logging
    optimizer = RMSProp(learning_rate=args.learning_rate)

    logger.info("Args used: {}".format(args))
    logger.info("Num_tasks: {}".format(len(tasks)))

    tensorboard = True
    if tf.executing_eagerly():
        tensorboard = False

    # Dummy model is based on pretrained model, only used for restore and variable copy
    dummy_model = ChemCeption(n_tasks=100, img_spec=args.img_spec,
                        inception_blocks=inception_blocks,
                        base_filters=args.base_filters, augment=args.use_augment,
                        model_dir=None, mode="regression",
                        n_classes=2, batch_size=args.batch_size,
                        optimizer=None, tensorboard=tensorboard,
                        tensorboard_log_frequency=100)

    finetune_model = ChemCeption(n_tasks=len(tasks), img_spec=args.img_spec,
                        inception_blocks=inception_blocks,
                        base_filters=args.base_filters, augment=args.use_augment,
                        model_dir=model_dir_1, mode=mode,
                        n_classes=2, batch_size=args.batch_size,
                        optimizer=optimizer, tensorboard=tensorboard,
                        tensorboard_log_frequency=100)

    finetune_model.load_from_pretrained(source_model=dummy_model, assignment_map=None, model_dir=restore_dir, include_top=False)

    train, valid, test = dataset

    logger.info("Created model dir at {}".format(model_dir_1))
    best_models_dir_1 = os.path.join(DIRNAME, args.dataset + "-finetune", "chemception", hparams_dir, "best-models", exp_name)

    loss_old = compute_loss_on_valid(valid, finetune_model, tasks, mode=mode)
    logger.info("Saving best model so far")
    finetune_model.save_checkpoint(model_dir=best_models_dir_1)

    train_scores = finetune_model.evaluate(train, [metric], [])
    valid_scores = finetune_model.evaluate(valid, [metric], [])
    test_scores = finetune_model.evaluate(test, [metric], [])

    logger.info("Train-{}: {}".format(metric.name, train_scores[metric.name]))
    logger.info("Valid-{}: {}".format(metric.name, valid_scores[metric.name]))
    logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))

    for rep_num in range(10):
        logger.info("Training model for {} epochs.".format(args.early_stopping_epoch))
        if args.train_all:
            finetune_model.fit(train, nb_epoch=args.early_stopping_epoch, checkpoint_interval=0)
        else:
            var_list = finetune_model.model.trainable_variables[-2:]
            finetune_model.fit(train, nb_epoch=args.early_stopping_epoch, variables=var_list, checkpoint_interval=0)
        loss_new = compute_loss_on_valid(valid, finetune_model, tasks, mode=mode, verbose=False)

        train_scores = finetune_model.evaluate(train, [metric], [])
        valid_scores = finetune_model.evaluate(valid, [metric], [])
        test_scores = finetune_model.evaluate(test, [metric], [])

        logger.info("Train-{}: {}".format(metric.name, train_scores[metric.name]))
        logger.info("Valid-{}: {}".format(metric.name, valid_scores[metric.name]))
        logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))

        logger.info("Computed loss on validation set after {} epochs: {}".format(args.early_stopping_epoch, loss_new))
        if loss_new > loss_old:
            logger.info("No improvement in validation loss. Enforcing early stopping.")
            break

        logger.info("Saving best model so far")
        finetune_model.save_checkpoint(model_dir=best_models_dir_1)
        loss_old = loss_new


if __name__ == "__main__":
    main()
