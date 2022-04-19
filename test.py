from cv2 import mean
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import torch
from datetime import datetime
import math
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.dataloader import LSDataset
from utils.dataloader import LSDataset_val
from model.LSnet import * 
import neptune.new as neptune
from sklearn.metrics import mean_squared_error , mean_absolute_percentage_error, mean_absolute_error
run = neptune.init(
    project="jm-ss/baby-eval-Integrated",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlN2JlZDM2Ni1iM2E0LTRmOWItODBlMi00MzM1ZDg4Zjc3OWYifQ==",
)

def main():
    day = datetime.today().strftime('%Y-%m-%d')
    parser = argparse.ArgumentParser(description = "Pytorch Lifespan estimation")
    parser.add_argument('--loss_func', type= str, default='L2')
    parser.add_argument('--lr', type = float, default = 0.00001)
    parser.add_argument('--optim', type = str, default='SGD')
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--save_loc', type = str, default = 'output/')
    args = parser.parse_args()

    params = {"learning_rate": args.lr, "optimizer": args.optim, "loss":args.loss_func, "eopch": args.epoch}
    run["parameters"]=params

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    batch_size=100

    test_dataset= LSDataset_val()
    print("test size : ",len(test_dataset))

    test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)

    model = Build14(in_ch=7, out_ch=1)

    model.load_state_dict(torch.load('/mnt/e/Workspace/jm/Projects/baby_eval/output/SL1_SGD_300_0010413_best.pth'))
    model.to(device)
    model.eval()

    if args.loss_func == 'L2':
        criterion = torch.nn.MSELoss()
    elif args.loss_func == 'L1':
        criterion = torch.nn.L1Loss()
    criterion = [torch.nn.SmoothL1Loss(), torch.nn.SmoothL1Loss()]

    epochs=args.epoch

    for epoch in range(1):
        iter = math.ceil(len(test_dataset)/batch_size)
        # Evaluation
        loss_val=0
        acc_hi=0
        acc_fnls=0

        loss_val=0
        mae_hi =0
        mape_hi =0
        acc=0
        mape_fnls =0
        mae_fnls=0
        smape_fnls =0
        model.eval()
        with torch.no_grad():
            correct_hi = 0
            correct_fnls=0
            for data, hi_gt, fnls_gt in test_loader:
                x = data.to(device)
                y1_val = hi_gt.to(device)
                y2_val = fnls_gt.to(device)

                y1_pred_val, y2_pred_val = model(x)

                val_loss_hi = criterion[0](y1_pred_val, y1_val)
                val_loss_fnls = criterion[1](y2_pred_val, y2_val)
                val_loss =   val_loss_hi +  val_loss_fnls

                loss_val +=val_loss.item()

                # prediction = np.round(y_pred_eval.squeeze().cpu().numpy())
                hi_pred = y1_pred_val.squeeze().cpu().numpy()
                fnls_pred = y2_pred_val.squeeze().cpu().numpy()
                hi_gt = y1_val.squeeze().cpu().numpy() 
                fnls_gt = y2_val.squeeze().cpu().numpy() 
                mae_hi += mean_absolute_error(hi_gt, hi_pred)
                # smape_hi += np.mean(100*2*np.abs(hi_gt-hi_pred)/(hi_pred+hi_gt)*len(hi_pred))
                mape_hi += np.mean(100* np.abs(hi_gt-hi_pred)/hi_gt*len(hi_pred))
                mae_fnls += mean_absolute_error(fnls_gt, fnls_pred)
                mape_fnls += np.mean(100* np.abs(fnls_gt-fnls_gt)/fnls_gt*len(fnls_gt))
                # smape_fnls += np.mean(100*2*np.abs(fnls_gt-fnls_pred)/(fnls_gt+fnls_pred)*len(fnls_pred))

                for i in range(len(fnls_gt)):
                    if int(hi_pred[i])==int(hi_gt[i]):
                        correct_hi+=1
                    if int(fnls_pred[i])==int(fnls_gt[i]):
                        correct_fnls+=1

        acc_hi = correct_hi/len(test_dataset)
        acc_fnls = correct_fnls/len(test_dataset)


        loss_val = loss_val/ len(test_dataset)

        mae_hi = mae_hi/iter
        mape_hi = mape_hi/iter        
        mae_fnls = mae_fnls/iter
        mape_fnls = mape_fnls/iter

        run["val/loss"].log(loss_val)
        run['val/acc_hi'].log(acc_hi)
        run['val/acc_fnls'].log(acc_fnls)
        run['val/mae_hi'].log(mae_hi)
        run['val/mape_hi'].log(mape_hi)
        run['val/mae_fnls'].log(mae_fnls)

        print("Epoch : {}, acc : {}, val_loss : {}, mae : {}, mape : {}".format(epoch, acc_fnls, loss_val,  mae_fnls, smape_fnls))
        print(y1_val)
        print(y1_pred_val)
        print(y2_val)
        print(y2_pred_val)

    run.stop()

if __name__ =="__main__":
    main()
    