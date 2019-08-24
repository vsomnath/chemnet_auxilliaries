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

import deepchem as dc
from deepchem.metrics import to_one_hot
from deepchem.models import ChemCeption
from deepchem.models.tensorgraph.optimizers import RMSProp, ExponentialDecay

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--early_stopping_epoch", default=10, type=int, help="Number of epochs to check early stopping after for")
    parser.add_argument("--use_augment", action='store_true', help="Whether to perform real-time augmentation")
    parser.add_argument("--base_filters", default=16, type=int, help="Number of base filters to use")
    parser.add_argument("--inception_per_block", default=3, type=int, help="Number of inception layers per block")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size used for training")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate used.")

    args = parser.parse_args()

    layers_per_block = args.inception_per_block
    inception_blocks = {"A": layers_per_block, "B": layers_per_block, "C": layers_per_block}

    DIRNAME = os.path.join(os.environ.get("SCRATCH", "./"), "deepchem-data")
    load_fn = dc.molnet.load_chembl25
    tasks, dataset, transformers = load_fn(featurizer="smiles2img", data_dir=DIRNAME, save_dir=DIRNAME, img_spec=args.img_spec, split="stratified")

    metric_type = dc.metrics.rms_score
    task_averager = np.mean

    metric = dc.metrics.Metric(metric_type, task_averager=task_averager, mode=mode, verbose=False)
    train, valid, test = dataset

    # Setup directory for experiment
    exp_name = dt.now().strftime("%d-%m-%Y--%H-%M-%S")
    hparams_dir = "filters_{}_blocklayers_{}_imgspec_{}".format(args.base_filters, layers_per_block, args.img_spec)
    model_dir_1 = os.path.join(DIRNAME, "chembl25", "chemception", hparams_dir, exp_name)

    # Optimizer and logging
    optimizer = RMSProp(learning_rate=args.learning_rate)

    logger.info("Args used: {}".format(args))
    logger.info("Num_tasks: {}".format(len(tasks)))

    ###### TRAINING FIRST PART WITH CONSTANT LEARNING RATE ###############
    model = ChemCeption(n_tasks=len(tasks), img_spec=args.img_spec,
                        inception_blocks=inception_blocks,
                        base_filters=args.base_filters, augment=args.use_augment,
                        model_dir=model_dir_1, mode=mode,
                        n_classes=2, batch_size=args.batch_size,
                        optimizer=optimizer, tensorboard=True,
                        tensorboard_log_frequency=1000)
    model._ensure_built()

    train, valid, test = dataset

    logger.info("Created model dir at {}".format(model_dir_1))
    best_models_dir_1 = os.path.join(DIRNAME, chembl25, "chemception", hparams_dir, "best-models", exp_name)
    logger.info("Saving best model so far")
    model.save_checkpoint(model_dir=best_models_dir_1)

    loss_old = compute_loss_on_valid(valid, model, tasks, mode=mode)
    logger.info("Computed loss on validation set {}".format(loss_old))

    train_scores = model.evaluate(train, [metric], [])
    valid_scores = model.evaluate(valid, [metric], [])
    test_scores = model.evaluate(test, [metric], [])

    logger.info("Train-{}: {}".format(metric.name, train_scores[metric.name]))
    logger.info("Valid-{}: {}".format(metric.name, valid_scores[metric.name]))
    logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))

    for rep_num in range(5):
        logger.info("Training model for {} epochs.".format(args.early_stopping_epoch))
        model.fit(train, nb_epoch=args.early_stopping_epoch, checkpoint_interval=0)
        loss_new = compute_loss_on_valid(valid, model, tasks, mode=mode, verbose=False)

        train_scores = model.evaluate(train, [metric], [])
        valid_scores = model.evaluate(valid, [metric], [])
        test_scores = model.evaluate(test, [metric], [])

        logger.info("Train-{}: {}".format(metric.name, train_scores[metric.name]))
        logger.info("Valid-{}: {}".format(metric.name, valid_scores[metric.name]))
        logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))

        logger.info("Computed loss on validation set after {} epochs: {}".format(args.early_stopping_epoch, loss_new))
        if loss_new > loss_old:
            logger.info("No improvement in validation loss. Enforcing early stopping.")
            break

        logger.info("Saving best model so far")
        model.save_checkpoint(model_dir=best_models_dir_1)
        loss_old = loss_new

if __name__ == "__main__":
    main()
