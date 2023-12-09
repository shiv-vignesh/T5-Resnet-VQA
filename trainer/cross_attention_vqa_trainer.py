import os, json, time 

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from .logger import Logger
from .callbacks import EarlyStopping
from dataset_utils.vit_vqa_dataset import VitT5CollateFn, OKVQADataset
from dataset_utils.vit_vqa_daquar_dataset import DaquarVitT5CollateFn, DaquarDataset
from dataset_utils.wup_measure import wup_measure
from dataset_utils.utils import convert_time_to_readable_format
from model.vit_vqa_model import CrossAttentionVitVQAModel

from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer

import wandb

class CrossAttentionVQATrainer:

    def __init__(
        self, 
        model:CrossAttentionVitVQAModel,
        trainer_kwargs:dict,
        optimizer_kwargs:dict,
        lr_scheduler_kwargs:dict,
        callbacks_kwargs:dict,
        dataset_kwargs:dict):

        wandb.init(
            project="IDAI-610-Term-Project_05Dec__263_ans_space_RIT_GPU",
            config={
                "dataset_kwargs":dataset_kwargs,
                "optimizer_kwargs":optimizer_kwargs,
                "trainer_kwargs":trainer_kwargs,
                "lr_scheduler_kwargs":lr_scheduler_kwargs,
                "callbacks_kwargs":callbacks_kwargs
            }
        )
        
        self.model = model 

        self.is_training = trainer_kwargs["is_training"]
        self.first_val_epoch = trainer_kwargs["first_val_epoch"]
        self.metric_eval_mode = trainer_kwargs["metric_eval_mode"]
        self.metric_average_mode = trainer_kwargs["metric_average_mode"]
        self.epochs = trainer_kwargs["epochs"]
        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]
        self.monitor_test = trainer_kwargs["monitor_test"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        self.device_count = torch.cuda.device_count()
        self.mxp_training = trainer_kwargs["mxp_training"]
        self.loss_combination_strategy = trainer_kwargs["loss_combination_strategy"]
        self.learnable_parameter = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")        
        self.model.to(self.device)

        self.output_dir = trainer_kwargs["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)        

        self.logger = Logger(trainer_kwargs)

        self.logger = Logger(trainer_kwargs)

        prev_layer_name = ""
        for name, param in self.model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name != prev_layer_name:
                prev_layer_name = layer_name
                self.logger.log_block("{:<70} {:<30} {:<30} {:<30}".format('Name','Weight Shape','Total Parameters', 'Trainable'))
            self.logger.log_message("{:<70} {:<30} {:<30} {:<30}".format(name, str(param.data.shape), param.data.numel(), param.requires_grad))

        # log if loaded from checkpoint
        if trainer_kwargs["load_from_checkpoint"]:
            if os.path.exists(os.path.join(self.output_dir, "checkpoint-model.pt")):
                self.logger.log_line()
                self.logger.log_message(f"Loaded from Checkpoint: {os.path.join(self.output_dir, 'checkpoint-model.pt')}")
            elif os.path.exists(os.path.join(self.output_dir, "best-model.pt")):
                self.logger.log_line()
                self.logger.log_message(f"Loaded from Best Model: {os.path.join(self.output_dir, 'best-model.pt')}")

        # self._init_dataloader(dataset_kwargs)
        self._init_daquar_dataloader(dataset_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Root Data Directory: {dataset_kwargs["root_data_dir"]}')

        # self.logger.log_message(f'Train Images Directory: {dataset_kwargs["train_images_dir"]}')
        # self.logger.log_message(f'Test Images Directory: {dataset_kwargs["test_images_dir"]}')
        # self.logger.log_message(f'Train Dataset: {dataset_kwargs["train_data_json_fn"]}')
        # self.logger.log_message(f'Test Dataset: {dataset_kwargs["test_data_json_fn"]}')

        self.logger.log_message(f'Images Directory: {dataset_kwargs["images_dir"]}')
        self.logger.log_message(f'Train Dataset: {dataset_kwargs["train_csv_file"]}')
        self.logger.log_message(f'Test Dataset: {dataset_kwargs["test_csv_file"]}')

        self.logger.log_message(f'Answer Spaces: {dataset_kwargs["answer_spaces_file"]}')
        self.logger.log_new_line()    

        self.num_training_steps = self.total_train_batch*self.epochs
        self.num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        self.num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])

        self._init_optimizer(optimizer_kwargs, trainer_kwargs["load_from_checkpoint"])

        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()        

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line()                    

        self._init_lr_scheduler(lr_scheduler_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
        self.logger.log_new_line()
        for k, v in self.lr_scheduler.state_dict().items():
            self.logger.log_message("{:<30} {}".format(k, v))        

        self._init_callbacks(callbacks_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Callbacks: {self.callbacks.__class__.__name__}')
        self.logger.log_new_line()
        self.logger.log_message("{:<30} {}".format('save_final_model', self.callbacks.save_final_model))
        self.logger.log_message("{:<30} {}".format('patience', self.callbacks.patience))
        self.logger.log_message("{:<30} {}".format('threshold', self.callbacks.threshold))
        self.logger.log_message("{:<30} {}".format('mode', self.callbacks.mode))

        # put model to device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        self.logger.log_line()
        self.logger.log_message(f'Device: {self.model.device} and Device Count: {self.device_count}')
        self.logger.log_new_line()

    def _init_daquar_dataloader(self, dataset_kwargs:dict):
        
        tokenizer_name = dataset_kwargs["language_model_tokenizer"]
        image_preprocessor = dataset_kwargs["image_preprocessor_model"]

        def init_dataloader_helper(root_data_dir:str, 
                                   csv_file_path:str, 
                                   images_dir:str, 
                                   batch_size:int,
                                   type:str):



            dataset = DaquarDataset(
                root_dir=root_data_dir,
                csv_file_path=csv_file_path,
                images_dir=images_dir,
                type=type
            )

            if type == "val":

                return DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    collate_fn=DaquarVitT5CollateFn(
                        image_model=image_preprocessor,
                        lang_model=tokenizer_name,
                        answer_spaces=answer_spaces,
                        eval_mode=True
                    )
                )

            if type == "train":

                return DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    collate_fn=DaquarVitT5CollateFn(
                        image_model=image_preprocessor,
                        lang_model=tokenizer_name,
                        answer_spaces=answer_spaces
                    )
                )
            
        root_dir = dataset_kwargs["root_data_dir"]

        train_csv_fn = dataset_kwargs["train_csv_file"]
        test_csv_fn = dataset_kwargs["test_csv_file"]

        images_dir = dataset_kwargs["images_dir"]

        train_batch_size = dataset_kwargs["train_batch_size"]
        test_batch_size = dataset_kwargs["test_batch_size"]

        answer_spaces_fp = dataset_kwargs["answer_spaces_file"]
        answer_spaces_fp = os.path.join(root_dir, answer_spaces_fp)
        answer_spaces = open(answer_spaces_fp).readlines()

        self.train_dataloader = init_dataloader_helper(
            root_dir, train_csv_fn, images_dir, train_batch_size, type="train"
        )

        self.test_dataloader = init_dataloader_helper(
            root_dir, test_csv_fn, images_dir, test_batch_size, type="val"
        )

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10        


    def _init_dataloader(self, dataset_kwargs:dict):
        
        tokenizer_name = dataset_kwargs["language_model_tokenizer"]
        image_preprocessor = dataset_kwargs["image_preprocessor_model"]

        def init_dataloader_helper(root_data_dir:str, 
                                   dataset_json_fp:str, 
                                   images_dir:str, 
                                   batch_size:int,
                                   type:str):

            dataset_json_fp = os.path.join(root_data_dir, dataset_json_fp)
            images_dir = os.path.join(root_data_dir, images_dir)

            dataset = OKVQADataset(
                dataset_json_fp, images_dir, type
            )

            if type == "val":

                return DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    collate_fn=VitT5CollateFn(
                        image_model=image_preprocessor,
                        lang_model=tokenizer_name,
                        answer_spaces=answer_spaces,
                        eval_mode=True
                    )
                )

            if type == "train":

                return DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    collate_fn=VitT5CollateFn(
                        image_model=image_preprocessor,
                        lang_model=tokenizer_name,
                        answer_spaces=answer_spaces
                    )
                )
            
        root_dir = dataset_kwargs["root_data_dir"]
        train_data_json_fn = dataset_kwargs["train_data_json_fn"]
        test_data_json_fn = dataset_kwargs["test_data_json_fn"]

        train_images_dir = dataset_kwargs["train_images_dir"]
        test_images_dir = dataset_kwargs["test_images_dir"]

        train_batch_size = dataset_kwargs["train_batch_size"]
        test_batch_size = dataset_kwargs["test_batch_size"]

        answer_spaces_fp = dataset_kwargs["answer_spaces_file"]
        answer_spaces_fp = os.path.join(root_dir, answer_spaces_fp)
        answer_spaces = open(answer_spaces_fp).readlines()

        self.train_dataloader = init_dataloader_helper(
            root_dir, train_data_json_fn, train_images_dir, train_batch_size, type="train"
        )

        self.test_dataloader = init_dataloader_helper(
            root_dir, test_data_json_fn, test_images_dir, test_batch_size, type="val"
        )

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10        


    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"])    

    def _init_optimizer(self, optimizer_kwargs:dict, load_from_checkpoint:bool):
        param_dict = []

        param_dict.append({
            "params":self.model.vit_embeddings.parameters(), "lr": optimizer_kwargs["vision_lr"], "model_name":"Vision Embeddings"
        })

        param_dict.append({
            "params":self.model.vit_encoder_layers.parameters(), "lr": optimizer_kwargs["vision_lr"], "model_name":"Vision Encoders"
        })

        param_dict.append({
            "params":self.model.roberta_embeddings.parameters(), "lr": optimizer_kwargs["lm_encoder_lr"], "model_name":"Language Embeddings"
        })

        param_dict.append({
            "params":self.model.roberta_encoder_layers.parameters(), "lr": optimizer_kwargs["lm_encoder_lr"], "model_name":"Language Encoders"
        })

        param_dict.append({
            "params":self.model.attention_pooler.parameters(), "lr": optimizer_kwargs["classifier_lr"], "model_name":"Attention Pooler Layers"
        })        

        param_dict.append({
            "params":self.model.classification_layer.parameters(), "lr": optimizer_kwargs["classifier_lr"],  "model_name":"Classifier Layer"
        })                

        self.optimizer = getattr(
            torch.optim, optimizer_kwargs["type"]
        )(param_dict, **optimizer_kwargs["kwargs"])

        if os.path.exists(os.path.join(self.output_dir, "state_dict_checkpoint.pt")) and load_from_checkpoint:
            self.logger.log_line()
            self.logger.log_message('Loaded Optimizer from Checkpoint')            
            checkpoint_path = os.path.join(self.output_dir, 'state_dict_checkpoint.pt')

            checkpoint = torch.load(checkpoint_path)
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.optimizer_to_device()        

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"]

        num_training_steps = self.total_train_batch*self.epochs
        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    def train(self):
        self.logger.log_line()
        self.logger.log_message(f"Start Training: Max Epoch {self.epochs}")
        self.logger.log_new_line()
        self.total_training_time = 0.0

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch
                self.logger.log_line()

                self.train_one_epoch()

                if self.monitor_test:
                    self.valid_one_epoch()

        except KeyboardInterrupt:
            self.callbacks.exit_training(self.model)
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            wandb.finish()
            exit(1)
            
        wandb.finish()
    
    def train_one_epoch(self):

        self.model.train()
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0

        train_predictions = []
        train_targets = []

        for batch_idx, data_items in enumerate(self.train_dataloader):
            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)

            step_begin_time = time.time()
            loss, lm_logits = self.train_one_step(data_items)
            step_end_time = time.time()
            
            total_loss += loss
            ten_percent_batch_total_loss += loss

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)
            
            predicted_indices = self.convert_logits_to_predictions(lm_logits)
            train_predictions.extend(predicted_indices.tolist())
            train_targets.extend(data_items["annotation_ids"].tolist())

            layer_name_lrs = {param_group["model_name"]: param_group["lr"] for param_group in self.optimizer.param_groups}
            log_lrs = "" # log lr of each layers
            for layer_name, lr in layer_name_lrs.items():
                log_lrs += f" - {layer_name} lr: {lr:.2e}"

            if self.total_train_batch < 10:
                msg = f'Epoch: {self.cur_epoch} - iteration {batch_idx}/{self.total_train_batch} - total loss {total_loss:.4f}'
                self.logger.log_message(message=msg)            

            elif (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch

                sec_per_batch_log_message = f" - secs/batch {convert_time_to_readable_format(round(average_time, 4))}"
                message = f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f}" + log_lrs + sec_per_batch_log_message
                self.logger.log_message(message=message)

                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0 

        self.total_training_time += epoch_training_time
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Average Loss {total_loss/self.total_train_batch} - Epoch Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))} - Total Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))}")
        
        wups_scores = []

        for prediction, target in zip(train_predictions, train_targets):
            
            prediction = self.train_dataloader.collate_fn.answer_spaces[prediction]
            target = self.train_dataloader.collate_fn.answer_spaces[target]
            
            wups_score = wup_measure(prediction, target)
            wups_scores.append(wups_score)

        avg_wups_score = sum(wups_scores)/len(wups_scores)
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Average Loss {total_loss/self.total_train_batch} - Average WUPS Score: {avg_wups_score:.4f} - Epoch Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))} - Total Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))}")

        # _, overall_metric_dict = self.compute_rouge_metric(
        #     train_predictions, train_targets, self.train_dataloader.collate_fn.answer_spaces
        # )

        # print(overall_metric_dict)
        # exit(1)
        
        wandb.log(
            {
                "epoch":self.cur_epoch,
                "train_avg_loss":total_loss/self.total_train_batch,
                "train_avg_wups":avg_wups_score
                }
        )

    def train_one_step(self, data_items):

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            lm_logits, loss = self.model(**data_items)
            loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            self.optimizer.step()        
        self.lr_scheduler.step()

        return loss.item(), lm_logits            

    def valid_one_epoch(self):
        
        self.model.eval()
        
        total_valid_loss = 0.0 
        valid_predictions = []
        valid_targets = []

        with torch.no_grad():
            for epoch, data_items in enumerate(self.test_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.device)

                del data_items["answers"]
                del data_items["questions"]
                
                lm_logits, loss = self.model(**data_items)
                total_valid_loss += loss.item()

                predicted_indices = self.convert_logits_to_predictions(lm_logits)
                valid_predictions.extend(predicted_indices.tolist())
                valid_targets.extend(data_items["annotation_ids"].tolist())                

        wups_scores = []

        for prediction, target in zip(valid_predictions, valid_targets):

            prediction = self.test_dataloader.collate_fn.answer_spaces[prediction]
            target = self.test_dataloader.collate_fn.answer_spaces[target]

            wups_score = wup_measure(prediction, target)
            wups_scores.append(wups_score)

        avg_wups_score = sum(wups_scores)/len(wups_scores)
        avg_valid_loss = total_valid_loss/len(self.test_dataloader)
        
        self.logger.log_line()
        self.logger.log_message(f'Epoch #{self.cur_epoch}: Average Validation Loss: {avg_valid_loss:.4f} - Average WUPS Score: {avg_wups_score:.4f}')
        self.logger.log_new_line()

        wandb.log(
            {
                "epoch":self.cur_epoch,
                "valid_avg_loss":avg_valid_loss,
                "valid_avg_wups":avg_wups_score
                }
        )

    def convert_logits_to_predictions(self, lm_logits:torch.tensor):
        
        scores = torch.exp(lm_logits)
        predicted_indices = torch.argmax(scores, dim=1)

        return predicted_indices

    def compute_rouge_metric(self, predictions:list, targets:list, answer_spaces:list):

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        predictions = [answer_spaces[idx] for idx in predictions]
        targets = [answer_spaces[idx] for idx in targets]

        output_dict = defaultdict(dict) #store predictions
        overall_metric_dict = defaultdict(lambda:defaultdict(int))

        for idx, (prediction, target) in enumerate(zip(predictions, targets)):
            scores = scorer.score(target, prediction)

            for score_type, scores in scores.items():

                overall_metric_dict[score_type]["precision"] += scores.precision
                overall_metric_dict[score_type]["recall"] += scores.recall
                overall_metric_dict[score_type]["fmeasure"] += scores.fmeasure

        for score_type, metrics in overall_metric_dict.items():
            for metric, value in metrics.items():
                value = value/len(predictions)
                overall_metric_dict[score_type][metric] = value


        return output_dict, overall_metric_dict

    def optimizer_to_device(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)            