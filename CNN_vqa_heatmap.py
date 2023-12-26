import torch 
from torch.utils.data import DataLoader

from dataset_utils.resnet_vqa_daquar_dataset import DaquarDataset, DaquarFasterRcnnT5CollateFn
from dataset_utils.wup_measure import wup_measure
from model.faster_rcnn_vqa_model import FasterRcnnVQAModel
from model.resnet_vqa_model import ResnetVQAModel

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import cv2

import os, json
from tqdm import tqdm

def load_faster_rcnn_model(lang_model_name:str,answer_spaces:list, model_path:str, device:str="cuda"):

    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    model = FasterRcnnVQAModel(
        "faster-rcnn",
        lang_model_name,
        answer_spaces=len(answer_spaces), 
        device=device
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    return model, device

def load_resnet_model(vision_model_name:str, lang_model_name:str,answer_spaces:list, model_path:str, device:str="cuda"):

    if device == "cuda":
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    model = ResnetVQAModel(
        vision_model_name,
        lang_model_name,
        answer_spaces=len(answer_spaces), 
        device=device
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
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
        dataset, batch_size=4, collate_fn=DaquarFasterRcnnT5CollateFn(
            interpolation_strategy="bilinear_interpolation",
            resizing_dimensions=(256, 256),
            lang_model="t5-base",
            image_transforms=["Normalize","ToTensorV2"],           
            answer_spaces=answer_spaces,
            eval_mode=True
        )
    )    

    return dataloader

def convert_logits_to_predictions(lm_logits:torch.tensor):
    
    scores = torch.exp(lm_logits)
    predicted_indices = torch.argmax(scores, dim=1)

    return predicted_indices

def convert_logits_to_predictions_topk(lm_logits:torch.tensor, topk:int):
    
    scores = torch.exp(lm_logits)
    predicted_indices = torch.topk(scores, dim=1, k=topk)

    return predicted_indices    

def generate_heatmaps(model:FasterRcnnVQAModel, dataloader:DataLoader, output_dir:str, device:torch.device):

    # cam_extractor = SmoothGradCAMpp(model.vision_model, ['body.layer4'])

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
        image_file_paths = data_items["image_fns"]
        annotation_ids = data_items["annotation_ids"]
        questions = data_items["questions"]

        del data_items["answers"]
        del data_items["questions"]   
        del data_items["annotation_ids"] 
        del data_items["image_fns"]

        # ['0', '1', '2', '3', 'pool'] - image_feature_maps_dict
        lm_logits, _, image_feature_maps_dict = model.generate_answers(**data_items)
        # predicted_indices = convert_logits_to_predictions(lm_logits)
        predicted_indices = convert_logits_to_predictions(lm_logits)
        valid_predictions.extend(predicted_indices.tolist())
        valid_targets.extend(annotation_ids.tolist())  

        #[bs, 256, 4, 4]
        batch_features = image_feature_maps_dict["pool"] if "pool" in image_feature_maps_dict else image_feature_maps_dict["features"]
        
        for batch_idx, features in enumerate(batch_features):       

            #[256, 4, 4] -> [4, 4]
            cam = torch.mean(features, dim=0).squeeze()
            heatmap = (cam - cam.min()) / (cam.max() - cam.min())
            heatmap = heatmap.cpu().numpy()

            image = cv2.imread(image_file_paths[batch_idx])
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            alpha = 0.9
            superimposed_img = cv2.addWeighted(
                image, 1 - 0.5, heatmap, alpha, 0
            )

            predicted_idx = predicted_indices[batch_idx]
            target_idx = annotation_ids[batch_idx]

            prediction = dataloader.collate_fn.answer_spaces[predicted_idx]
            target = dataloader.collate_fn.answer_spaces[target_idx]
            wups_score = wup_measure(prediction, target)

            if wups_score == 1.0:

                if not os.path.exists(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/heatmap_{idx}_{batch_idx}.png', superimposed_img)
                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', image)

                with open(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":dataloader.collate_fn.answer_spaces[predicted_idx],
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx],
                        "wups_score":wups_score
                    }, f)

            else:

                if not os.path.exists(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/sample_heatmap_{idx}_{batch_idx}.png', superimposed_img)
                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', image)

                with open(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/predictions.json','w+') as f:
                    json.dump({
                        "question":questions[batch_idx].question_text,
                        "predicted_answer":dataloader.collate_fn.answer_spaces[predicted_idx],
                        "target_answer":dataloader.collate_fn.answer_spaces[target_idx],
                        "wups_score":wups_score
                    }, f)

    wups_scores = []
    for prediction, target in zip(valid_predictions, valid_targets):
        prediction = dataloader.collate_fn.answer_spaces[prediction]
        target = dataloader.collate_fn.answer_spaces[target]

        wups_score = wup_measure(prediction, target)
        wups_scores.append(wups_score)        

    avg_wups_score = sum(wups_scores)/len(wups_scores)

    print(f'Average Test WUPS Score: {avg_wups_score:.4f}')


def generate_heatmaps_topk(model:FasterRcnnVQAModel, dataloader:DataLoader, output_dir:str, device:torch.device, topk=5):

    # cam_extractor = SmoothGradCAMpp(model.vision_model, ['body.layer4'])

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
        image_file_paths = data_items["image_fns"]
        annotation_ids = data_items["annotation_ids"]
        questions = data_items["questions"]

        del data_items["answers"]
        del data_items["questions"]   
        del data_items["annotation_ids"] 
        del data_items["image_fns"]

        # ['0', '1', '2', '3', 'pool'] - image_feature_maps_dict
        lm_logits, _, image_feature_maps_dict = model.generate_answers(**data_items)
        # predicted_indices = convert_logits_to_predictions(lm_logits)
        predicted_indices = convert_logits_to_predictions_topk(lm_logits, topk=topk)[1]

        #[bs, 256, 4, 4]
        batch_features = image_feature_maps_dict["pool"] if "pool" in image_feature_maps_dict else image_feature_maps_dict["features"]
        
        for batch_idx, features in enumerate(batch_features):       

            #[256, 4, 4] -> [4, 4]
            cam = torch.mean(features, dim=0).squeeze()
            heatmap = (cam - cam.min()) / (cam.max() - cam.min())
            heatmap = heatmap.cpu().numpy()

            image = cv2.imread(image_file_paths[batch_idx])
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            alpha = 0.9
            superimposed_img = cv2.addWeighted(
                image, 1 - 0.5, heatmap, alpha, 0
            )

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
                    break

            if is_ans_present:

                if not os.path.exists(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions'):
                    os.makedirs(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions')

                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/heatmap_{idx}_{batch_idx}.png', superimposed_img)
                cv2.imwrite(f'{output_dir}/perfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', image)

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

                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/sample_heatmap_{idx}_{batch_idx}.png', superimposed_img)
                cv2.imwrite(f'{output_dir}/imperfect_match_predictions/{idx}_{batch_idx}_heatmap_predictions/original_{idx}_{batch_idx}.png', image)

                
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
    # answer_spaces = open(f'{dataset_dir}/answer_spaces_single_word_threshold_5_without_O.txt').readlines()

    answer_spaces = json.load(open('DAQUAR_dataset_Logs_286_classes_Resnet34_60_Epochs/model_checkpoints/model_ckpt_info.json'))["answer_spaces"]

    model, device = load_faster_rcnn_model(
        "t5-base", 
        answer_spaces,
        "DAQUAR_dataset_Logs_286_classes_FasterRCNN_without_O-Label_60_Epochs/model_checkpoints/best-model.pt","cuda"
    )

    # model, device = load_resnet_model(
    #     "resnet50",
    #     "t5-base", 
    #     answer_spaces,
    #     "DAQUAR_dataset_Logs_286_classes_Resnet50_v2_60_Epochs/model_checkpoints/best-model.pt","cuda"
    # )

    dataloader = create_dataloader(
        dataset_dir, csv_file, images_dir, answer_spaces
    )

    with torch.no_grad():
        generate_heatmaps(
                model, dataloader, "faster_RCNN_validation_heatmaps_organized", device
            )