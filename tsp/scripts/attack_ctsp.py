import numpy as np
from tsp.data import GoogleTSPReader
from tsp.utils import *
from tsp.attacks import attack_rand, attack_opt
import argparse
from tsp.data import get_CTSP_training_data
from tsp.solvers.ctsp import get_convtsp_model


def val_tsp(net, val, pert="random", batch_size=20):
    net.eval()
    dataset = GoogleTSPReader(20, -1, batch_size, val)
    batches_per_epoch = dataset.max_iter
    dataset = iter(dataset)
    running_loss, running_pred_tour_len, running_gt_tour_len = 0.0, 0.0, 0.0
    running_nb_data, running_nb_batch = 0, 0
    losses, preds, gts = [], [], []

    for batch_num in range(batches_per_epoch):
        try:
            batch = next(dataset)
        except StopIteration:
            break
        x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes = unroll(batch)
        batch_attack = {
            'coords': [x.cpu() for x in x_nodes_coord],
            'routes': [y.tolist() for y in y_nodes],
            'og_batch': batch        
        }
        if pert == "random":
            outs = attack_rand(batch_attack, net)  

        elif pert == "opt":
            outs = attack_opt(batch_attack, net) 
        else:
            raise NotImplementedError()    
        (x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes) = outs
        y_preds, loss = net.forward(outs)

        pred_tour_len, gt_tour_len = padded_get_stats(y_preds, batch)
        losses.extend(loss.tolist())
        preds.extend(pred_tour_len.tolist())
        gts.extend(gt_tour_len.tolist())
        print(batch_num)
        print(np.array(losses).mean(), np.array(preds).mean(), np.array(gts).mean())
        
        running_nb_data += batch_size
        running_loss += batch_size * loss.mean().data.item()
        running_pred_tour_len += batch_size * pred_tour_len.mean()
        running_gt_tour_len += batch_size * gt_tour_len.mean()
        running_nb_batch += 1

    loss = running_loss / running_nb_data
    pred_length = running_pred_tour_len / running_nb_data
    gt_length = running_gt_tour_len / running_nb_data
    
    return loss, pred_length, gt_length 



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Attack')
    parser.add_argument('-seed', default=0, action='store_true', help='Python random seed')
    parser.add_argument('-opt', default=False, action='store_true', help='Optimized or random attack')
    parser.add_argument('-model_path', default="../trained_models/trained_ctsp.pt", action='store_true', help='Model to be attacked')
    args = parser.parse_args()

    model, _ = get_convtsp_model(args.seed)
    model.cuda()
    params = torch.load(args.model_path)
    params_new = dict()
    for key in params.keys():
        params_new[key[7:]] = params[key]
    model.load_state_dict(params_new)
    train, val, test, name = get_CTSP_training_data()
    res = val_tsp(model, val, pert="opt" if args.opt else "random")
    print(res)