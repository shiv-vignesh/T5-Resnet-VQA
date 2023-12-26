## Packages to install 

    !pip3 install torch
    !pip3 install torchvision
    !pip3 install transformers
    !pip3 install opencv-python
    !pip3 install pandas 
    !pip3 install albumentations
    !pip3 install nltk
    !pip3 install rouge-score
    !pip3 install wandb
    !pip3 install matplotlib

Alternatively, you can also use requirements.txt file. However, I have experienced slow installation through that process. 

## Directory Structure

    model 
    |___faster_rcnn_vqa_model.py
    |___multi_head_vision_text_attn.py
    |___resnet_vqa_model.py
    |___vit_vqa_model.py

    dataset_utls
    |___enums.py
    |___resbet_vqa_daquar_dataset.py
    |___utils.py
    |___vit_vqa_daquar_dataset.py
    |___wups_measure.py

    trainer
    |___callbacks.py
    |___faster_rcnn_vqa_trainer.py
    |___logger.py
    |___vit_vqa_trainer.py

    train_faster_rcnn_vqa.py 
    train_vit_vqa.py
    
    vit_daquar_config.json 
    
    CNN_vqa_heatmap.py
    ViT_vqa_heatmap.py

## Training Execution 

<b>Note VGG-16 in this context is faster-rcnn. As faster-rcnn is pre-trained with VGG16 backbone. </b>

- Modify the following kwargs in `vit_daquar_config.json`. 
    `model_kwargs` > `vision_model_name` : `[googlvit-base-patch16-224-in21k, resnet34, resnet50, faster-rcnn]`.

    `model_kwargs` > `language_model_name` : `[t5-base]`.

    `model_kwargs` > `language_model_name` : `[t5-base]`.

    `dataset_kwargs` > `dataset_kwargs` : `train_batch_size and test_batch_size`.

    `trainer_kwargs` > `output_dir`. Model Checkpoint at test predictions at each epoch are stored in this directory. 

- <b>Ensure that the DAQUAR_dataset directory is the same level as the file. Else modify</b>

    `root_dir`
    `train_csv_file`
    `test_csv_file`
    `images_dir`
    `answer_spaces_file`

- To train `VGG-16 (faster-rcnn)/Resnet18/Resnet34/Resnet50`
    `train_faster_rcnn_vqa.py`

- To train `ViT Transformer`
    `train_vit_vqa.py`

## Testset Execution 

To test and plot feature or attention maps of the trained model. 

- Ensure the model checkpoints are available. 
- To test CNN based models `VGG-16 (faster-rcnn)/Resnet18/Resnet34/Resnet50`, Use `CNN_vqa_heatmap.py`.

    - There are two methods available to load the models. 
        - `load_faster_rcnn_model()` used to load faster_rcnn (VGG-16) model. Pass the path to `best-model.pt`. 
        - `load_resnet_model()` used to load Resnet34, Resnet50 model. Pass the path to `best-model.pt`. 
        
        - `dataset_dir`: root directory
        - `images_dir` : images_directory name with root. 
        - `csv_file` : test_csv.file name.

        - `generate_heatmaps()` pass the output directory to store the correct and incorrect predictions along with the heatmap plots. Also prints the Average WUPS-Score on the testset. 

        - `generate_heatmaps_topk()` pass the output directory to store the correct and incorrect predictions along with the heatmap plots. Also prints the Average WUPS-Score on the testset. `topk` values to get predictions. 

- To test Vision Transformer based models `ViT`, Use `ViT_vqa_heatmap.py`.

    - There are two methods available to load the models. 
        - `load_model()` used to load ViT model. Pass the path to `best-model.pt`.  
        
        - `dataset_dir`: root directory
        - `images_dir` : images_directory name with root. 
        - `csv_file` : test_csv.file name.

        - `generate_heatmaps()` pass the output directory to store the correct and incorrect predictions along with the heatmap plots. Also prints the Average WUPS-Score on the testset. 

        - `generate_heatmaps_topk()` pass the output directory to store the correct and incorrect predictions along with the heatmap plots. Also prints the Average WUPS-Score on the testset. `topk` values to get predictions.         





                