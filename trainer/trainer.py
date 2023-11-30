import os, json, time 

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from .logger import Logger
from .callbacks import EarlyStopping
from dataset_utils.dataset import VQADataset, BatchCollateFn
from dataset_utils.utils import convert_time_to_readable_format
from model.vqa_model import VQAModel

from collections import defaultdict

import wandb

class VQATrainer:

    def __init__(
        self, 
        model:VQAModel,
        trainer_kwargs:dict,
        optimizer_kwargs:dict,
        lr_scheduler_kwargs:dict,
        callbacks_kwargs:dict,
        dataset_kwargs:dict):

        wandb.init(
            project="IDAI-610-Term-Project",
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

        self._init_dataloader(dataset_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Root Data Directory: {dataset_kwargs["root_data_dir"]}')
        self.logger.log_message(f'Train Images Directory: {dataset_kwargs["train_images_dir"]}')
        self.logger.log_message(f'Test Images Directory: {dataset_kwargs["test_images_dir"]}')
        self.logger.log_message(f'Training Annotation: {dataset_kwargs["train_annotations_json_fn"]}')
        self.logger.log_message(f'Training Questions: {dataset_kwargs["train_questions_json_fn"]}')
        self.logger.log_message(f'Testing Annotation: {dataset_kwargs["test_annotations_json_fn"]}')
        self.logger.log_message(f'Testing Questions: {dataset_kwargs["test_questions_json_fn"]}')
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

        self._init_lr_scheduler(lr_scheduler_kwargs, trainer_kwargs["load_from_checkpoint"])
        self.logger.log_line()
        self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
        self.logger.log_new_line()
        for k, v in self.lr_scheduler.state_dict().items():
            self.logger.log_message("{:<30} {}".format(k, v))        
        
        self.logger.log_message(f'Train Preprocessing Kwargs:')
        self.logger.log_new_line()
        for k,v in dataset_kwargs["train_preprocessing_kwargs"].items():
            self.logger.log_message(f'{k}: {v}')
        
        self.logger.log_new_line()
        self.logger.log_message(f'Test Preprocessing Kwargs:')
        self.logger.log_new_line()
        for k,v in dataset_kwargs["test_preprocessing_kwargs"].items():
            self.logger.log_message(f'{k}: {v}')
        
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

    def _init_optimizer(self, optimizer_kwargs:dict, load_from_checkpoint:bool):

        '''
        Modify function when only part of the model needs to be fine-tuned. 
        '''

        # lm_encoder_params = set(self.model.language_model.encoder.state_dict().keys())
        # lm_decoder_params = set(self.model.language_model.decoder.state_dict().keys())

        # common_params = lm_decoder_params.intersection(lm_encoder_params)

        # print("Common Parameters:")
        # for param in common_params:
        #     print(param)

        # exit(1)        

        param_dict = []

        if self.model.vision_model.fine_tune:
            param_dict.append({
                "params":self.model.vision_model.parameters(), "lr": optimizer_kwargs["vision_lr"], "model_name":"Vision Model"
            })

        if self.model.language_model.fine_tune_encoder and self.model.language_model.fine_tune_decoder:
            param_dict.append({
                "params":self.model.language_model.parameters(), "lr": optimizer_kwargs["lm_encoder_lr"], "model_name":"LM Encoder and Decoder"
            })
        # '''
        # #TODO, Fix this error at this ValueError: some parameters appear in more than one parameter group. 
        # Occurs when self.model.lang_model.encoder and self.model.lang_model.decoder are loaded into params_dict
        
        # Add prefix string model.encoder and mode.decoder to common parameters. 
        # '''
        elif self.model.language_model.fine_tune_encoder:
            param_dict.append({
                "params":self.model.language_model.encoder.parameters(), "lr": optimizer_kwargs["lm_decoder_lr"], "model_name":"LM Decoder"
            })

        elif self.model.language_model.fine_tune_decoder:
            param_dict.append({
                "params":self.model.language_model.decoder.parameters(), "lr": optimizer_kwargs["lm_decoder_lr"], "model_name":"LM Decoder"
            })

        param_dict.append({
            "params":self.model.answer_classifier.parameters(), "lr":optimizer_kwargs["classifier_lr"], "model_name":"Answer Classifier"
        })

        param_dict.append({
            "params":self.model.question_type_classifier.parameters(), "lr":optimizer_kwargs["classifier_lr"], "model_name":"QuestionType Classifier"
        })        

        self.optimizer = getattr(torch.optim, optimizer_kwargs["type"])(param_dict, **optimizer_kwargs["kwargs"])

        if os.path.exists(os.path.join(self.output_dir, "state_dict_checkpoint.pt")) and load_from_checkpoint:
            self.logger.log_line()
            self.logger.log_message('Loaded Optimizer from Checkpoint')            
            checkpoint_path = os.path.join(self.output_dir, 'state_dict_checkpoint.pt')

            checkpoint = torch.load(checkpoint_path)
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.optimizer_to_device()

    def _init_dataloader(self, dataset_kwargs:dict):
        def init_dataloader_helper(annotations_json_fn, questions_json_fn, images_dir, 
                                   resizing_w, resizing_h, interpolation_strategy, 
                                   batch_size, image_transforms, lang_model, type):
            annotations_json = json.load(open(
                f'{root_data_dir}/{annotations_json_fn}'
            ))

            questions_json = json.load(open(
                f'{root_data_dir}/{questions_json_fn}'
            ))

            vqa_dataset = VQADataset(
            annotations_json, questions_json, f'{root_data_dir}/{images_dir}',type
            )            

            if type == "train":
                return DataLoader(
                    vqa_dataset, batch_size=batch_size,
                    collate_fn=BatchCollateFn(
                        resizing_dimensions=(resizing_w, resizing_h), 
                        interpolation_strategy=interpolation_strategy,
                        image_transforms=image_transforms,
                        lang_model=lang_model
                    )
                )

            if type == "val":
                return DataLoader(
                    vqa_dataset, batch_size=batch_size,
                    collate_fn=BatchCollateFn(
                        resizing_dimensions=(resizing_w, resizing_h), 
                        interpolation_strategy=interpolation_strategy,
                        image_transforms=image_transforms,
                        lang_model=lang_model,
                        eval_mode=True
                    )
                )

        root_data_dir = dataset_kwargs["root_data_dir"]
        train_preprocessing_kwargs = dataset_kwargs["train_preprocessing_kwargs"]
        resizing_w, resizing_h = train_preprocessing_kwargs["resizing_width"], train_preprocessing_kwargs["resizing_height"]
        interpolation_strategy = train_preprocessing_kwargs["interpolation_strategy"]
        train_image_transforms = train_preprocessing_kwargs["image_transforms"]
        lang_model = train_preprocessing_kwargs["language_model_name"]  

        self.train_dataloader = init_dataloader_helper(
            annotations_json_fn=dataset_kwargs["train_annotations_json_fn"],
            questions_json_fn=dataset_kwargs["train_questions_json_fn"],
            images_dir=dataset_kwargs["train_images_dir"],
            resizing_w=resizing_w,
            resizing_h=resizing_h,
            interpolation_strategy=interpolation_strategy,
            batch_size=train_preprocessing_kwargs["batch_size"],
            image_transforms=train_image_transforms,
            lang_model=lang_model,
            type="train"
        )

        self.train_batch_size = self.train_dataloader.batch_size
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10

        test_preprocessing_kwargs = dataset_kwargs["test_preprocessing_kwargs"]
        resizing_w, resizing_h = test_preprocessing_kwargs["resizing_width"], train_preprocessing_kwargs["resizing_height"]
        interpolation_strategy = test_preprocessing_kwargs["interpolation_strategy"]
        test_image_transforms = test_preprocessing_kwargs["image_transforms"]
        lang_model = test_preprocessing_kwargs["language_model_name"]  

        self.test_dataloader = init_dataloader_helper(
            annotations_json_fn=dataset_kwargs["test_annotations_json_fn"],
            questions_json_fn=dataset_kwargs["test_questions_json_fn"],
            images_dir=dataset_kwargs["test_images_dir"],
            resizing_w=resizing_w,
            resizing_h=resizing_h,
            interpolation_strategy=interpolation_strategy,
            batch_size=test_preprocessing_kwargs["batch_size"],
            image_transforms=test_image_transforms,
            lang_model=lang_model,
            type="val"
        )

    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"])    

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict, load_from_checkpoint:bool=True):

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)

    def train(self):

        self.logger.log_line()
        self.logger.log_message(f"Start Training: Max Epoch {self.epochs}")
        self.logger.log_new_line()
        self.total_training_time = 0.0

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch
                self.logger.log_line()

                # self.train_one_epoch()
                '''
                #TODO, self.callbacks.save_state_dict_checkpoint(epoch, self.lr_scheduler, self.optimizer) 
                #TODO, valid_one_epoch()
                '''

                if self.monitor_test:
                    avg_valid_loss, avg_qt_loss, avg_answer_loss, epoch_predictions = self.valid_one_epoch()

                    with open(f'{self.output_dir}/{epoch}_predictions_target.json','w+') as f:
                        json.dump(epoch_predictions, f)

                # if self.cur_epoch >= self.first_val_epoch:
                #     pass

        except KeyboardInterrupt:
            self.callbacks.exit_training(self.model)
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            wandb.finish()
            exit(1)
            
        wandb.finish()

    def train_one_epoch(self):
        '''
        #TODO, monitor train predictions. Use answer_logits, 
        '''
        
        self.model.train()
        total_loss = 0.0 
        total_answer_loss, total_question_type_loss = 0, 0
        ten_percent_batch_total_loss = 0
        ten_percent_batch_ans_loss = 0
        ten_percent_batch_qt_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0

        for batch_idx, data_items in enumerate(self.train_dataloader):

            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)
                    # print(f'{k}\t{data_items[k].device}\t{self.device}\t{self.model.device}')
                    #         
            step_begin_time = time.time()
            
            if self.mxp_training:
                loss_sum, answer_loss, question_type_loss, answer_logits, question_type_scores = self.train_one_mxp_step(data_items)    
            else:
                loss_sum, answer_loss, question_type_loss, answer_logits, question_type_scores = self.train_one_step(data_items)
            
            step_end_time = time.time()
            
            total_loss += loss_sum
            total_answer_loss += answer_loss
            total_question_type_loss += question_type_loss

            ten_percent_batch_total_loss += loss_sum
            ten_percent_batch_ans_loss += answer_loss
            ten_percent_batch_qt_loss += question_type_loss

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)

            layer_name_lrs = {param_group["model_name"]: param_group["lr"] for param_group in self.optimizer.param_groups}
            log_lrs = "" # log lr of each layers
            for layer_name, lr in layer_name_lrs.items():
                log_lrs += f" - {layer_name} lr: {lr:.2e}"

            if self.total_train_batch < 10:
                msg = f'Epoch: {self.cur_epoch} - iteration {batch_idx}/{self.total_train_batch} - total loss {total_loss:.4f} - answer loss {answer_loss:.4f} - question type loss {question_type_loss:.4f}'
                self.logger.log_message(message=msg)

            elif (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_ans_loss = ten_percent_batch_ans_loss/self.ten_percent_train_batch
                average_qt_loss = ten_percent_batch_qt_loss/self.ten_percent_train_batch

                average_time = ten_percent_training_time/self.ten_percent_train_batch
                sec_per_batch_log_message = f" - secs/batch {convert_time_to_readable_format(round(average_time, 4))}"

                message = f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f} - total answer loss {average_ans_loss:.4f} - total question type loss {average_qt_loss:.4f}" + log_lrs + sec_per_batch_log_message
                self.logger.log_message(message=message)
                ten_percent_batch_total_loss = 0
                ten_percent_batch_ans_loss = 0
                ten_percent_batch_qt_loss = 0
                ten_percent_training_time = 0                

        self.total_training_time += epoch_training_time
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Average Loss {total_loss/self.total_train_batch} - Average Answer Loss {total_answer_loss/self.total_train_batch} - Average QT Loss {total_question_type_loss/self.total_train_batch} - Epoch Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))} - Total Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))}")
        wandb.log(
            {
                "epoch":self.cur_epoch,
                "train_avg_loss":total_loss/self.total_train_batch,
                "train_avg_answer_loss":total_answer_loss/self.total_train_batch,
                "train_avg_qt_loss":total_question_type_loss/self.total_train_batch
                }
        )

    def train_one_step(self, data_items):
        '''
        #TODO, experiment with weighted total loss; tl = alpha*ans_loss + beta*qt_loss
        '''
        # for k,v in data_items.items():
        #     if torch.is_tensor(v):                    
        #         # data_items[k] = v.to(self.device)
        #         print(f'{k}\t{data_items[k].device}\t{self.device}\t{self.model.device}')

        self.optimizer.zero_grad()        

        with torch.set_grad_enabled(True):
            answer_logits, question_type_scores, answer_loss, question_type_loss = self.model(
                **data_items
            )
            ''' TODO, change this code with KLD loss'''

            # if self.loss_combination_strategy == "dynamic_weighted":
            #     weight = torch.sigmoid(self.learnable_parameter).to(self.device)
            #     total_loss = weight * answer_loss + (1-weight) * question_type_loss
            
            # elif self.loss_combination_strategy == "CE_KLD":
            #     pass 

            total_loss = answer_loss + question_type_loss

            total_loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            self.optimizer.step()        
        
        self.lr_scheduler.step()
        return total_loss.item(), answer_loss.item(), question_type_loss.item(), answer_logits, question_type_scores

    def train_one_mxp_step(self, data_items):
        self.optimizer.zero_grad()        
        scaler = GradScaler(enabled=True)  

        with torch.set_grad_enabled(True), autocast(enabled=True):
            answer_logits, answer_loss, question_type_scores, question_type_loss = self.model(
                **data_items
            )
            ''' TODO, change this code with KLD loss'''
            
            if self.loss_combination_strategy == "dynamic_weighted":
                weight = torch.sigmoid(self.learnable_parameter).to(self.device)
                total_loss = weight * answer_loss + (1-weight) * question_type_loss
            
            elif self.loss_combination_strategy == "CE_KLD":
                pass 
            
            scaler.scale(total_loss).backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            scaler.step(self.optimizer)
            scaler.update()
        
        self.lr_scheduler.step()
        return total_loss, answer_loss, question_type_loss, answer_logits, question_type_scores

    def valid_one_epoch(self):
        
        self.model.eval()
        total_valid_loss = 0.0 
        total_answer_loss = 0.0 
        total_qt_loss = 0.0

        epoch_predictions = defaultdict(lambda:defaultdict(int))

        with torch.no_grad():
            for epoch, data_items in enumerate(self.test_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.device)

                _, valid_answer_loss, _, valid_question_type_loss = self.valid_one_step(data_items)
                total_valid_loss += (valid_answer_loss + valid_question_type_loss)
                total_answer_loss += valid_answer_loss
                total_qt_loss += valid_question_type_loss 

                ''' 
                indices - [bs, topk, max_len]; log_prob - [bs, topk]
                '''
                indices, log_prob = self.model.generate_answers(
                    data_items["question_input_ids"], data_items["question_attention_masks"],
                    data_items["image_tensors"], data_items["question_type_ids"],
                    self.test_dataloader.collate_fn.tokenizer.pad_token_id, 
                    self.test_dataloader.collate_fn.tokenizer.pad_token_id
                )
                
                batch_topk_predictions = []
                batch_topk_prediction_log_prob = []

                batch_answers = data_items["answers"]
                batch_questions = data_items["questions"]

                for batch_idx in range(log_prob.shape[0]): 
                    batch_idx_predictions = []
                    batch_idx_prediction_log_prob_scores = []

                    for k in range(log_prob.shape[1]):
                        token_ids = indices[batch_idx, k, :]
                        decoded_str = self.test_dataloader.collate_fn.tokenizer.batch_decode(
                        token_ids
                        )

                        batch_idx_predictions.append(decoded_str)
                        batch_idx_prediction_log_prob_scores.append(log_prob[batch_idx][k].item())

                    batch_topk_predictions.extend(batch_idx_predictions)
                    batch_topk_prediction_log_prob.extend(batch_idx_prediction_log_prob_scores)

                for idx, question in enumerate(batch_questions):
                    answer = batch_answers[idx]

                    epoch_predictions[f'{epoch}_{idx}']["question"] = question.question_text
                    epoch_predictions[f'{epoch}_{idx}']["answer"] = answer
                    epoch_predictions[f'{epoch}_{idx}']['topk_predictions'] = batch_topk_predictions
                    epoch_predictions[f'{epoch}_{idx}']['topk_predictions_scores'] = batch_topk_prediction_log_prob


        avg_valid_loss = total_valid_loss/len(self.test_dataloader)
        avg_answer_loss = total_answer_loss/len(self.test_dataloader)
        avg_qt_loss = total_qt_loss/len(self.test_dataloader)

        self.logger.log_line()
        self.logger.log_message(f'Epoch #{self.cur_epoch}: Average Validation Loss: {avg_valid_loss:.4f} - Average Answer Loss: {avg_answer_loss:.4f} - Average Question Type Loss: {avg_qt_loss:.4f}')
        self.logger.log_new_line()

        wandb.log(
            {
                "epoch":self.cur_epoch,
                "valid_avg_loss":avg_valid_loss,
                "valid_avg_answer_loss":avg_answer_loss,
                "valid_avg_qt_loss":avg_qt_loss
                }
        )


        return avg_valid_loss, avg_qt_loss, avg_answer_loss, epoch_predictions
   
    def valid_one_step(self, data_items):
        
        answers = data_items["answers"]
        questions = data_items["questions"]

        del data_items["answers"]
        del data_items["questions"]
        
        answer_logits, question_type_scores, answer_loss, question_type_loss = self.model(
            **data_items
        )          

        data_items["questions"] = questions 
        data_items["answers"] = answers

        return answer_logits, answer_loss.item(), question_type_scores, question_type_loss.item()

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