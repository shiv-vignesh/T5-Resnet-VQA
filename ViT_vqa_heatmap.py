import torch 
from torch.utils.data import DataLoader

from dataset_utils.vit_vqa_daquar_dataset  import DaquarDataset, DaquarVitT5CollateFn
from dataset_utils.wup_measure import wup_measure
from model.vit_vqa_model import VitVQAModel

import os, json
from tqdm import tqdm

import numpy as np
import cv2

import matplotlib.pyplot as plt

def load_model(lang_model_name:str,answer_spaces:list, model_path:str, device:str="cuda"):

    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    model = VitVQAModel(
        "google/vit-base-patch16-224-in21k",
        lang_model_name,
        answer_spaces=len(answer_spaces),
        device=device
    )

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )

    model.to(device)

    return model, device

def create_dataloader(dataset_dir:str, csv_file:str, images_dir:str, answer_spaces:list):

    dataset = DaquarDataset(
        dataset_dir,
        csv_file,
        images_dir,
        type="train"
    )

    dataloader = DataLoader(
        dataset, batch_size=4, collate_fn=DaquarVitT5CollateFn(
            "google/vit-base-patch16-224-in21k",
            "t5-base",
            answer_spaces=answer_spaces,
            eval_mode=True
        )
    )   

    return dataloader

def convert_logits_to_predictions(lm_logits:torch.tensor):
    
    scores = torch.exp(lm_logits)
    predicted_indices = torch.argmax(scores, dim=1)

    return predicted_indices

def generate_heatmaps(model:VitVQAModel, 
                      dataloader:DataLoader, 
                      output_dir:str, 
                      device:torch.device,
                      get_mask=False):

    model.eval()

    valid_predictions = []
    valid_targets = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(f'{output_dir}/perfect_match_predictions'):
        os.makedirs(f'{output_dir}/perfect_match_predictions')

    if not os.path.exists(f'{output_dir}/imperfect_match_predictions'):
        os.makedirs(f'{output_dir}/imperfect_match_predictions')

    for idx, data_items in tqdm(enumerate(dataloader)):
        for k,v in data_items.items():
            if torch.is_tensor(v):                    
                data_items[k] = v.to(device)    

        answers = data_items["answers"]
        questions = data_items["questions"]
        annotation_ids = data_items["annotation_ids"]
        questions = data_items["questions"]
        images_file_paths = data_items["image_fns"]   

        del data_items["answers"]
        del data_items["questions"]   
        del data_items["annotation_ids"] 
        del data_items["image_fns"]       

        lm_logits, loss, attention_tensors = model.generate_answers(**data_items)         
        predicted_indices = convert_logits_to_predictions(lm_logits)
        valid_predictions.extend(predicted_indices.tolist())
        valid_targets.extend(annotation_ids.tolist())   

        batch_size = lm_logits.size(0)
        #last layer attention #[12, bs, 12, 197, 197]
        attention_tensors = torch.stack(attention_tensors).squeeze(1)
     
        image_arrs = [cv2.imread(image_fp) for image_fp in images_file_paths]
        image_arrs = [cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB) for image_arr in image_arrs]

        for batch_idx in range(batch_size):
            
            #[12, 12, 197, 197]
            batch_attention_tensors = attention_tensors[:, batch_idx, :, :, :]            
            
            #[12, 197, 197]            
            batch_attention_tensors = torch.mean(batch_attention_tensors, dim=1)
            # batch_attention_tensors = batch_attention_tensors[:, -1, :, :]

            #[12, 197, 197] 
            residual_att = torch.eye(batch_attention_tensors.size(1)).to(device) #[197, 197]
            aug_att_mat = batch_attention_tensors + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
            joint_attentions[0] = aug_att_mat[0]

            #[12, 197, 197]
            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))          

            img = image_arrs[batch_idx]  

            mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
            if get_mask:
                result = cv2.resize(mask / mask.max(), img.size)
            else:        
                mask = cv2.resize(mask / mask.max(), (img.shape[1], img.shape[0]))[..., np.newaxis]
                result = (mask * img).astype("uint8")
                result = cv2.applyColorMap(result, cv2.COLORMAP_JET)

                # result = result * 0.4 + img
            
            predicted_idx = predicted_indices[batch_idx]
            target_idx = annotation_ids[batch_idx]

            prediction = dataloader.collate_fn.answer_spaces[predicted_idx]
            target = dataloader.collate_fn.answer_spaces[target_idx]
            wups_score = wup_measure(prediction, target)

            if wups_score == 1.0:

                if not os.path.exists(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/heatmap_{idx}_{batch_idx}.png', result)
                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', img)

                with open(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":dataloader.collate_fn.answer_spaces[predicted_idx],
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx]
                    }, f)

            else:

                if not os.path.exists(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/sample_heatmap_{idx}_{batch_idx}.png', result)

                with open(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":dataloader.collate_fn.answer_spaces[predicted_idx],
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx]
                    }, f)

    wups_scores = []
    for prediction, target in zip(valid_predictions, valid_targets):
        prediction = dataloader.collate_fn.answer_spaces[prediction]
        target = dataloader.collate_fn.answer_spaces[target]

        wups_score = wup_measure(prediction, target)
        wups_scores.append(wups_score)        

    avg_wups_score = sum(wups_scores)/len(wups_scores)

    print(f'Average Test WUPS Score: {avg_wups_score:.4f}')

def convert_logits_to_predictions_topk(lm_logits:torch.tensor, topk:int):
    
    scores = torch.exp(lm_logits)
    predicted_indices = torch.topk(scores, dim=1, k=topk)

    return predicted_indices    


def generate_heatmaps_topk(model:VitVQAModel, dataloader:DataLoader, output_dir:str, device:torch.device,get_mask=False, topk=5):

    model.eval()

    valid_predictions = []
    valid_targets = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(f'{output_dir}/perfect_match_predictions'):
        os.makedirs(f'{output_dir}/perfect_match_predictions')

    if not os.path.exists(f'{output_dir}/imperfect_match_predictions'):
        os.makedirs(f'{output_dir}/imperfect_match_predictions')

    for idx, data_items in tqdm(enumerate(dataloader)):
        for k,v in data_items.items():
            if torch.is_tensor(v):                    
                data_items[k] = v.to(device)        

        answers = data_items["answers"]
        questions = data_items["questions"]
        annotation_ids = data_items["annotation_ids"]
        questions = data_items["questions"]
        images_file_paths = data_items["image_fns"]   

        del data_items["answers"]
        del data_items["questions"]   
        del data_items["annotation_ids"] 
        del data_items["image_fns"]       

        lm_logits, loss, attention_tensors = model.generate_answers(**data_items)         
        # predicted_indices = convert_logits_to_predictions(lm_logits)
        predicted_indices = convert_logits_to_predictions_topk(lm_logits, topk=topk)[1]

        batch_size = lm_logits.size(0)
        #last layer attention #[12, bs, 12, 197, 197]
        attention_tensors = torch.stack(attention_tensors).squeeze(1)
     
        image_arrs = [cv2.imread(image_fp) for image_fp in images_file_paths]
        image_arrs = [cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB) for image_arr in image_arrs]

        for batch_idx in range(batch_size):
            
            #[12, 12, 197, 197]
            batch_attention_tensors = attention_tensors[:, batch_idx, :, :, :]            
            
            #[12, 197, 197]            
            batch_attention_tensors = torch.mean(batch_attention_tensors, dim=1)
            # batch_attention_tensors = batch_attention_tensors[:, -1, :, :]

            #[12, 197, 197] 
            residual_att = torch.eye(batch_attention_tensors.size(1)).to(device) #[197, 197]
            aug_att_mat = batch_attention_tensors + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
            joint_attentions[0] = aug_att_mat[0]

            #[12, 197, 197]
            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))          

            img = image_arrs[batch_idx]  

            mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
            if get_mask:
                result = cv2.resize(mask / mask.max(), img.size)
            else:        
                mask = cv2.resize(mask / mask.max(), (img.shape[1], img.shape[0]))[..., np.newaxis]
                result = (mask * img).astype("uint8")
                result = cv2.applyColorMap(result, cv2.COLORMAP_JET)

                # result = result * 0.4 + img
            
            topk_predicted_idx = predicted_indices[batch_idx]
            target_idx = annotation_ids[batch_idx]

            is_ans_present = False
            true_answer_prediction = ""

            for predicted_idx in topk_predicted_idx:
                prediction = dataloader.collate_fn.answer_spaces[predicted_idx]
                target = dataloader.collate_fn.answer_spaces[target_idx]
                wups_score = wup_measure(prediction, target)

                if wups_score == 1.0:
                    is_ans_present = True
                    true_answer_prediction = prediction

            if is_ans_present:

                if not os.path.exists(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/heatmap_{idx}_{batch_idx}.png', result)
                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', img)

                top_k_predictions = [dataloader.collate_fn.answer_spaces[idx] for idx in topk_predicted_idx]

                with open(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":true_answer_prediction,
                        "topk_answers":top_k_predictions,
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx],
                        "wups_score":wups_score
                    }, f)

            else:

                if not os.path.exists(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/sample_heatmap_{idx}_{batch_idx}.png', result)
                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', img)

                
                top_k_predictions = [dataloader.collate_fn.answer_spaces[idx] for idx in topk_predicted_idx]

                with open(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":true_answer_prediction,
                        "topk_answers":top_k_predictions,
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx],
                        "wups_score":wups_score
                    }, f)


if __name__ == "__main__":

    dataset_dir = "../DAQUAR_dataset"
    csv_file = "test_modified_v2.csv"
    images_dir = "images"

    answer_spaces = json.load(open('DAQUAR_dataset_Logs_286_classes_ViT_without_O-Label/model_checkpoints/model_ckpt_info.json'))["answer_spaces"]

    model, device = load_model(
        "t5-base", 
        answer_spaces,
        "DAQUAR_dataset_Logs_286_classes_ViT_without_O-Label/model_checkpoints/best-model.pt","cuda"
    )

    dataloader = create_dataloader(
        dataset_dir, csv_file, images_dir, answer_spaces
    )

    with torch.no_grad():
        generate_heatmaps(
                model, dataloader, "ViT_validation_heatmaps_organized", device
            )