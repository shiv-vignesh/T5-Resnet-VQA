import torch
import torch.nn as nn 
import torch.nn.functional as F

from .faster_rcnn_vqa_model import AttentionPooler
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from transformers import T5ForConditionalGeneration, T5Config, ViTModel, RobertaModel

class CrossAttentionVitVQAModel(nn.Module):

    def __init__(
            self, 
            answer_spaces:int,
            vision_model_name:str="google/vit-base-patch16-224-in21k",
            language_model_name:str="roberta-base",
            fine_tune_lm_encoder:bool=True, 
            fine_tune_lm_decoder:bool=True,
            fine_tune_vision:bool=False,
            device="cpu"):
        
        super().__init__()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        if self.vision_model_name == "google/vit-base-patch16-224-in21k":
            vision_model = ViTModel.from_pretrained(self.vision_model_name)
        
        if self.language_model_name == "roberta-base":
            lang_model = RobertaModel.from_pretrained(self.language_model_name)

        self.roberta_embeddings = lang_model.embeddings
        self.vit_embeddings = vision_model.embeddings

        self.roberta_encoder_layers = lang_model.encoder.layer
        self.vit_encoder_layers = vision_model.encoder.layer
        
        self.num_layers = min(len(self.roberta_encoder_layers), len(self.vit_encoder_layers))
        self.layer_norm = nn.LayerNorm(768)

        self.attention_pooler = AttentionPooler(768)
        self.classification_layer = nn.Linear(
            768, answer_spaces
        )

        self.device = device
        self.fine_tune_lm_encoder = fine_tune_lm_encoder
        self.fine_tune_vision = fine_tune_vision

    def forward(self, 
                question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                annotation_ids:torch.tensor,
                pixel_values:torch.tensor,
                answer_input_ids:torch.tensor,
                answer_attention_masks:torch.tensor,   
                image_tensors:torch.tensor,             
                question_type_ids:torch.tensor=None):
        
        if self.fine_tune_vision:
            image_embedding_output = self.vit_embeddings(
                pixel_values
            )    

        else:
            with torch.no_grad():
                image_embedding_output = self.vit_embeddings(
                    pixel_values
                )    

        text_embedding_output = self.roberta_embeddings(
            question_input_ids
        )

        concatenated_embeddings = torch.concat([
            image_embedding_output, text_embedding_output
        ], dim=1)

        concatenated_hidden_states = concatenated_embeddings
        image_hidden_states = image_embedding_output

        for i in range(self.num_layers):
            roberta_layer = self.roberta_encoder_layers[i]
            vit_layer = self.vit_encoder_layers[i]

            roberta_hidden_states = roberta_layer(concatenated_hidden_states)[0]
            
            if self.fine_tune_vision:
                image_hidden_states = vit_layer(image_hidden_states)[0]
            else:
                with torch.no_grad():
                    image_hidden_states = vit_layer(image_hidden_states)[0]

            concatenated_hidden_states = torch.concat([roberta_hidden_states[:,197:, :], image_hidden_states], dim=1)
            concatenated_hidden_states = self.layer_norm(concatenated_hidden_states + concatenated_embeddings)

        pooled_outputs = self.attention_pooler(concatenated_hidden_states)
        lm_logits = self.classification_layer(pooled_outputs)
        lm_logits = nn.functional.log_softmax(lm_logits, dim=-1)

        if annotation_ids != None:        
            loss = nn.NLLLoss()(
                lm_logits, annotation_ids)

            return lm_logits, loss

        else:
            return lm_logits, None

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
        
        elif self.vision_model_name == "faster-rcnn":
            self.vision_model = fasterrcnn_resnet50_fpn(pretrained=True)

        if self.language_model_name == "t5-base":
            self.lang_model = T5ForConditionalGeneration.from_pretrained(self.language_model_name)

        self.fusing_layer = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classification_layer = nn.Linear(
            768, answer_spaces
        )

        self.fine_tune_lm_encoder = fine_tune_lm_encoder
        self.fine_tune_lm_decoder = fine_tune_lm_decoder
        self.fine_tune_vision = fine_tune_vision

        self.device = device
        self.num_beams = 2
        self.max_answer_length = 5

    def forward(self, 
                question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                annotation_ids:torch.tensor,
                pixel_values:torch.tensor,
                image_tensors:torch.tensor,
                answer_input_ids:torch.tensor,
                answer_attention_masks:torch.tensor,                
                question_type_ids:torch.tensor=None,
                ):
        
        
        #torch.Size([bs, 768])
        if self.vision_model_name == "google/vit-base-patch16-224-in21k":
            with torch.no_grad():
                vision_embeddings = self.vision_model(pixel_values)
                pooled_outputs = vision_embeddings.pooler_output
        
        if self.vision_model_name == "faster-rcnn":
            self.vision_model.eval()
            with torch.no_grad():
                vision_embeddings = self.vision_model.backbone(image_tensors)
                print(vision_embeddings['pool'].size())
                exit(1)

            #[bs, 197, 768]
            # vision_last_hidden_states = vision_embeddings.last_hidden_state

        # pooled_outputs = self.vision_model(pixel_values).pooler_output

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

        ''' 
        Commented For Decoder and Generate. 
        '''

        # lm_logits = self.classification_layer(fused_embedding)
        # lm_logits = nn.functional.log_softmax(lm_logits, dim=-1)

        # loss = nn.NLLLoss()(lm_logits, annotation_ids)

        # return lm_logits, loss 

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

        loss = nn.NLLLoss()(
            lm_logits, annotation_ids)

        return lm_logits, loss 

    def generate_answers(self, question_input_ids:torch.tensor, 
                decoder_question_input_ids:torch.tensor,
                question_attention_masks:torch.tensor, 
                decoder_question_attention_masks:torch.tensor,
                annotation_ids:torch.tensor,
                pixel_values:torch.tensor,
                answer_input_ids:torch.tensor,
                answer_attention_masks:torch.tensor,                 
                question_type_ids:torch.tensor=None):
        
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
        
        model_inputs = self.prepare_input_ids_for_generation(decoder_question_input_ids)
        expanded_attention_mask = self.prepare_attention_masks(decoder_question_attention_masks)


        for idx in range(self.max_answer_length):
            # model_inputs = self.prepare_input_ids_for_generation(decoder_question_input_ids)

            #[bs, 1, 768]
            decoder_outputs = self.lang_model.decoder(
                encoder_hidden_states=fused_embedding.unsqueeze(1),
                input_ids=model_inputs,
                attention_mask=expanded_attention_mask
            ).last_hidden_state

            last_index_of_ones = torch.max(torch.where(decoder_question_attention_masks.to(self.device) == 1, torch.arange(decoder_question_attention_masks.to(self.device).size(1)).to(self.device), torch.zeros_like(decoder_question_attention_masks.to(self.device)).to(self.device)).to(self.device), dim=1).values
            ans_token_embedding = [decoder_outputs[batch_idx, idx, :] for batch_idx, idx in enumerate(last_index_of_ones)]
            lm_logits = nn.functional.log_softmax(ans_token_embedding, dim=-1)
            


    def prepare_attention_masks(self, attention_mask:torch.tensor):
        expanded_attention_mask = attention_mask.unsqueeze(1)
        return expanded_attention_mask.expand(attention_mask.shape[0], self.num_beams, attention_mask.shape[1])
    
    def prepare_input_ids_for_generation(self, input_ids:torch.tensor):

        expanded_input_ids = input_ids.unsqueeze(1)        
        return expanded_input_ids.expand(input_ids.shape[0], self.num_beams, input_ids.shape[1])
