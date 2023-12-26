'''
callback classes to monitor the training process
'''

from .logger import Logger
import torch
import os
import json

class Callbacks:
    def __init__(self,
                 logger: Logger, 
                 output_dir: str,
                 save_final_model: bool=False):
        self.logger = logger
        self.save_final_model = save_final_model
        self.output_dir = f'{output_dir}/model_checkpoints'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.best_score = 0.0
        self.mode = "max"
    
    def better_score(self, score: float) -> bool:
        '''
        check if score improves
        '''
        if self.mode == "max":
            return (score-self.best_score) > self.threshold
        elif self.mode == "min":
            return (self.best_score-score) > self.threshold

    def save_checkpoint(self, model, epoch, answer_spaces):
        '''
        save new best model
        '''
        self.logger.log_message(f"Saving new best-model with F-Score: {self.best_score:.4f}")
        torch.save(model.state_dict(), os.path.join(self.output_dir, "best-model.pt"))

        with open(f'{self.output_dir}/model_ckpt_info.json','w+') as f:
            json.dump({
                "epoch":epoch, 
                "best_score":self.best_score,
                "answer_spaces":answer_spaces
            }, f)

    def exit_training(self, model):
        '''
        quit training
        '''
        self.logger.log_block(f"Exiting from training early. Best model score: {self.best_score:.4f}. Saving final model: {self.save_final_model} ")
        if self.save_final_model:
            self.logger.log_message("Saving model ...")
            torch.save(model.state_dict(), os.path.join(self.output_dir, "final-model.pt"))
            self.logger.log_message("Done.")
        exit(1)


class EarlyStopping(Callbacks):
    '''
    exit training when model stops improving for more than patient number of epochs

    Parameters:
    ===========
    `logger`: logger object
    `output_dir`: output path to save log file and model checkpoint
    `save_final_model`: saving model before exiting the training
    `patience`: number of epochs to ignore before early stopping
    `mode`: `max` or `min`. max means model is looking for higher score
    `threshold`: value to determine if this epoch is bad
    '''
    def __init__(self, 
                 logger: Logger,
                 output_dir: str,
                 save_final_model :bool=False,
                 patience: int=5, 
                 mode :str="max", 
                 threshold: float=0.01):
        super().__init__(logger, output_dir, save_final_model)

        self.patience = patience
        self.mode = mode
        self.threshold = threshold

        self.best_score = 0.0 if self.mode == "max" else float("inf")
        self.num_bad_epoch = 0

        # load model-metrics.json if exists
        if os.path.exists(f'{self.output_dir}/model-metrics.json'):
            metrics = json.load(open(f'{self.output_dir}/model-metrics.json'))
            self.best_score = metrics["best-score"]

    def __call__(self, model, score: float):
        '''
        save model for new best score, else check for early stopping condition
        '''
        if self.better_score(score):
            self.best_score = score
            self.num_bad_epoch = 0
            self.save_checkpoint(model)
            with open(f'{self.output_dir}/model-metrics.json','w+') as f:
                metrics = {"best-score" : score}
                json.dump(metrics, f)
        else:
            self.num_bad_epoch += 1
            self.logger.log_new_line()
            self.logger.log_message(f"Bad Epoch. Total num bad epoch: {self.num_bad_epoch}")
            if self.num_bad_epoch >= self.patience:
                self.exit_training(model)

    def save_epoch_checkpoint(self, model):

        self.logger.log_new_line()
        self.logger.log_message(f'Saving Epoch Checkpoint')
        torch.save(model.state_dict(), os.path.join(self.output_dir, "checkpoint-model.pt"))

    def save_state_dict_checkpoint(self, epoch, lr_scheduler, optimizer):

        torch.save({
            'epoch':epoch,
            'scheduler':lr_scheduler.state_dict(),
            'optimizer':optimizer.state_dict()
        },
        os.path.join(self.output_dir, 'state_dict_checkpoint.pt'))