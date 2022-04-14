import seml
import random
from sacred import Experiment
from tsp.data import get_DTSP_training_data
from tsp.solvers.dtsp import DTSP
from tsp.training import train_tsp

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(seed_model, epochs, lr, batch_size, batches_per_epoch, weight_decay):
    random.seed(0)
    train, val, _, name = get_DTSP_training_data()
    model = DTSP(seed=seed_model)
    return_dict = train_tsp(model, train, val, name, epochs=epochs, lr=lr, batch_size=batch_size,
                                 batches_per_epoch=batches_per_epoch, weight_decay=weight_decay)
    print(return_dict)
    return return_dict
