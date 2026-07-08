import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=25
            ):
    

    logging.info("start training")
    counter = 0

    for epoch in range(epoch_start, max_epochs):
        print(f"🚀 Epoch {epoch+1}/{max_epochs} - {time.strftime('%H:%M:%S')}")
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid"
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
            torch.save(state, path_aux)
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        batch_count = 0
        #for targets_offence_severity, targets_action, mvclips, action in dataloader:
        for batch in dataloader:
            if batch is None:
                continue
            targets_offence_severity, targets_action, mvclips, action = batch

            '''
            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            '''
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            targets_offence_severity = targets_offence_severity.to(device)
            targets_action = targets_action.to(device)
            mvclips = mvclips.to(device).float()

            batch_count += 1
            if batch_count % 20 == 0:
                print(f"📦 {set_name} B{batch_count}/{len(dataloader)}")

            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
          
        gc.collect()
        torch.cuda.empty_cache()
    
        data["Actions"] = actions
        with open(os.path.join(model_name, prediction_file), "w") as outfile: 
            json.dump(data, outfile)

        if total_loss == 0:
            return os.path.join(model_name, prediction_file), 0.0, 0.0

        return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss



# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
    all_y_true_sev = []
    all_y_pred_sev = []
    all_y_true_act = []
    all_y_pred_act = []
    batch_count = 0


    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            #_, _, mvclips, action = batch
            targets_offence_severity, targets_action, mvclips, action = batch

            mvclips = mvclips.float()
            if torch.cuda.is_available():
                mvclips = mvclips.cuda()

            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                #conf matrix
                true_sev = targets_offence_severity.detach().cpu().view(-1).tolist()
                true_act = targets_action.detach().cpu().view(-1).tolist()

                pred_sev = preds_sev.detach().cpu().view(-1).tolist()
                pred_act = preds_act.detach().cpu().view(-1).tolist()

                all_y_true_sev.extend(true_sev)
                all_y_pred_sev.extend(pred_sev)
                all_y_true_act.extend(true_act)
                all_y_pred_act.extend(pred_act)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)


                true_sev = targets_offence_severity.detach().cpu().view(-1).tolist()
                true_act = targets_action.detach().cpu().view(-1).tolist()

                pred_sev = preds_sev.detach().cpu().view(-1).tolist()
                pred_act = preds_act.detach().cpu().view(-1).tolist()

                all_y_true_sev.extend(true_sev)
                all_y_pred_sev.extend(pred_sev)
                all_y_true_act.extend(true_act)
                all_y_pred_act.extend(pred_act)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values

    gc.collect()
    torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)
   
    pd.DataFrame({
    "y_true_offence_severity": all_y_true_sev,
    "y_pred_offence_severity": all_y_pred_sev,
    "y_true_action": all_y_true_act,
    "y_pred_action": all_y_pred_act,
    }).to_csv(f"{set_name}_predictions.csv", index=False)

    sev_class_names = ["No offence", "Offence 1.0", "Offence 3.0", "Offence 5.0"]
    act_class_names = [INVERSE_EVENT_DICTIONARY["action_class"][k] for k in sorted(INVERSE_EVENT_DICTIONARY["action_class"].keys())]

    cm_sev = confusion_matrix(all_y_true_sev, all_y_pred_sev, labels=list(range(len(sev_class_names))))
    cm_act = confusion_matrix(all_y_true_act, all_y_pred_act, labels=list(range(len(act_class_names))))

    pd.DataFrame(cm_sev, index=sev_class_names, columns=sev_class_names).to_csv(
        f"{set_name}_confusion_offence_severity.csv"
    )
    pd.DataFrame(cm_act, index=act_class_names, columns=act_class_names).to_csv(
        f"{set_name}_confusion_action.csv"
    )

    per_class_acc_sev = np.divide(
        cm_sev.diagonal(),
        cm_sev.sum(axis=1),
        out=np.zeros(cm_sev.shape[0], dtype=float),
        where=cm_sev.sum(axis=1) != 0,
    )
    per_class_acc_act = np.divide(
        cm_act.diagonal(),
        cm_act.sum(axis=1),
        out=np.zeros(cm_act.shape[0], dtype=float),
        where=cm_act.sum(axis=1) != 0,
    )

    pd.DataFrame({
        "class_name": sev_class_names,
        "accuracy": per_class_acc_sev
    }).to_csv(f"{set_name}_per_class_accuracy_offence_severity.csv", index=False)

    pd.DataFrame({
        "class_name": act_class_names,
        "accuracy": per_class_acc_act
    }).to_csv(f"{set_name}_per_class_accuracy_action.csv", index=False)

    return prediction_file

