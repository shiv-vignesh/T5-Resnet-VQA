import torch
import torch.nn as nn 
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration, T5Config, ViTModel, RobertaModel

class CrossAttentionVitVQAModel(nn.Module):

    def __init__(
            self, 
            vision_model_name:str="google/vit-base-patch16-224-in21k",
            language_model_name:str="roberta-base",
            fine_tune_lm_encoder:bool=True, 
            fine_tune_lm_decoder:bool=True,
            fine_tune_vision:bool=True,
            device="cpu"):
        
        super().__init__()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        if self.vision_model_name == "google/vit-base-patch16-224-in21k":
            self.vision_model = ViTModel.from_pretrained(self.vision_model_name)
        
        if self.language_model_name == "roberta-base":
            self.lang_model = RobertaModel.from_pretrained(self.language_model_name)

    def forward(self, 
                question_input_ids:torch.tensor, 
                question_attention_masks:torch.tensor, 
                annotation_ids:torch.tensor,
                pixel_values:torch.tensor,
                question_type_ids:torch.tensor):
        
        image_embedding_output = self.vision_model.embeddings(
            pixel_values
        )    

        text_embedding_output = self.lang_model.embeddings(
            question_input_ids
        )

        

class VitVQAModel(nn.Module):

    def __init__(
            self, 
            vision_model_name:str,
            language_model_name:str,
            answer_spaces:int,
            fine_tune_lm_encoder:bool=True, 
            fine_tune_lm_decoder:bool=True,
            fine_tune_vision:bool=True,
            device="cpu"):
        
        super().__init__()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        if self.vision_model_name == "google/vit-base-patch16-224-in21k":
            self.vision_model = ViTModel.from_pretrained(self.vision_model_name)
        
        if self.language_model_name == "t5-base":
            self.lang_model = T5ForConditionalGeneration.from_pretrained(self.language_model_name)

        self.fusing_layer = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # self.classification_layer = nn.Linear(
        #     768, T5Config().vocab_size
        # )

        self.classification_layer = nn.Linear(
            768, answer_spaces
        )

        self.fine_tune_lm_encoder = fine_tune_lm_encoder
        self.fine_tune_lm_decoder = fine_tune_lm_decoder
        self.fine_tune_vision = fine_tune_vision

        self.device = device

    def forward(self, 
                question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                annotation_ids:torch.tensor,
                pixel_values:torch.tensor,
                question_type_ids:torch.tensor=None):
        
        
        #torch.Size([bs, 768])
        with torch.no_grad():
            pooled_outputs = self.vision_model(pixel_values).pooler_output

        #torch.Size([bs, longest_len, 768])
        encoder_outputs = self.lang_model.encoder(
            input_ids = question_input_ids,
            attention_mask=question_attention_masks
        ).last_hidden_state

        #obtaining the [QUESTION] special token if T5 else [CLS]
        cls_token_embedding = encoder_outputs[:,0,:]

        #[bs, 768+768]
        concatenated_embeddings = torch.cat([
            pooled_outputs, cls_token_embedding
        ], dim=1)

        #[bs, 768]
        fused_embedding = self.fusing_layer(concatenated_embeddings)
        # decoder_input_ids = self.lang_model.encoder._shift_right()

        #[bs, 1, 768]
        decoder_outputs = self.lang_model.decoder(
            encoder_hidden_states=fused_embedding.unsqueeze(1),
            # input_ids=question_input_ids[:,0].unsqueeze(1),
            input_ids=decoder_question_input_ids,
            attention_mask=decoder_question_attention_masks
        ).last_hidden_state

        last_index_of_ones = torch.max(torch.where(decoder_question_attention_masks.to(self.device) == 1, torch.arange(decoder_question_attention_masks.to(self.device).size(1)).to(self.device), torch.zeros_like(decoder_question_attention_masks.to(self.device)).to(self.device)).to(self.device), dim=1).values
        ans_token_embedding = [decoder_outputs[batch_idx, idx, :] for batch_idx, idx in enumerate(last_index_of_ones)]

        # [batch_size, 768]
        ans_token_embedding = torch.stack(ans_token_embedding, dim=0)
        
        lm_logits = self.classification_layer(ans_token_embedding)
        lm_logits = nn.functional.log_softmax(lm_logits, dim=-1)

        loss = nn.NLLLoss()(lm_logits, annotation_ids)

        return lm_logits, loss 
