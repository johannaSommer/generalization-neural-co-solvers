import seml
from sacred import Experiment
from train_nsat import train_nsat
from sat.solvers.neurosat import NeuroSAT
from sat.data import get_SAT_training_data

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
def run(epochs, seed_model, batch_size, augment, adv_rate):
    model = NeuroSAT(seed=seed_model)
    train, val, _, name = get_SAT_training_data("SAT-10-40")
    return_dict = train_nsat(model, train, val, name, epochs=epochs,
                             batch_size=batch_size, augment=augment, aug_rate=adv_rate)
    return return_dict
