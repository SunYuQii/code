'''

'''

import torch.nn as nn
import torch
from models.utils import PointWiseFeedForward
from models.BaseModel import BaseSeqModel

class SASRecBackbone(nn.Module):
    def __init__(self, device, args) -> None:
        super().__init__()


        self.dev = device
        # to be Q for self-attention
        
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
           
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

           
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

          
            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, log_seqs):
        '''
        -input:
        seqs:[batch,max_len,hidden_size]
        log_seqs:[batch,max_len]
        '''

        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        
        # [batch,max_len]
        timeline_mask = (log_seqs == 0)
        # broadcast in last dim
        
        # [batch,max_len,hidden_size]
        seqs *= ~timeline_mask.unsqueeze(-1)

        # time dim len for enforce causality
      
        tl = seqs.shape[1]
        
        # shape[max_len,max_len]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        
        for i in range(len(self.attention_layers)):
           
            seqs = torch.transpose(seqs, 0, 1)
           
            # [max_len,batch,hidden_size]
            Q = self.attention_layernorms[i](seqs)
           
            # [max_len,batch,hidden_size]
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,)
                                                    #   attn_mask=attention_mask)
            seqs = Q + mha_outputs 
            seqs = torch.transpose(seqs, 0, 1) #  [batch,max_len,hidden_size]

            seqs = self.forward_layernorms[i](seqs) 
            seqs *=  ~timeline_mask.unsqueeze(-1) 
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C) 

        return log_feats
    
class SASRec(BaseSeqModel):
    
    def __init__(self, user_num, item_num, device, args):
        
        super(SASRec, self).__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        self.item_num = item_num
        # self.dev = device


        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        # TO IMPROVE
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)
       
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)


        self.backbone = SASRecBackbone(device, args)


        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        # self.filter_init_modules = []
       
        self._init_weights()

    
    def _get_embedding(self, log_seqs):
        '''
        -input
        log_seqs [batch,max_len]
        -return
        [batch,max_len,hidden_size]
        '''


        min_idx = log_seqs.min().item()
        max_idx = log_seqs.max().item()
        # print(f'Input indices - Min: {min_idx}, Max: {max_idx}')
        # print(f"in embedding is {self.item_num}")

        if min_idx < 0 or max_idx >= self.item_num + 2:
            raise ValueError(f'Index out of range: min_idx={min_idx}, max_idx={max_idx}, item_num={self.item_num}')

        item_seq_emb = self.item_emb(log_seqs)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        '''
        item embedding+position embedding
        Get the representation of given sequence
        -input
        log_seq:[batch,max_len]
        positions:[batch,max_len]
        '''
        # [batch,max_len,hidden_size]
        seqs = self._get_embedding(log_seqs)
        
        seqs *= self.item_emb.embedding_dim ** 0.5
       
        # [batch,max_len,hidden_size]
        seqs += self.pos_emb(positions.long())
       
        # [batch,max_len,hidden_size]
        seqs = self.emb_dropout(seqs)

        
        log_feats = self.backbone(seqs, log_seqs)

        return log_feats


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions): # for training        
        '''
        Used to calculate pos and neg logits for loss
        -input
        seq:[batch,max_len]
        pos:[batch,max_len]
        neg:[batch,max_len]
        positions:[batch,max_len]
        '''

        # (bs, max_len, hidden_size)
        log_feats = self.log2feats(seq, positions)

        # (bs, max_len, hidden_size)
        pos_embs = self._get_embedding(pos)

        # (bs, max_len, hidden_size)
        neg_embs = self._get_embedding(neg)


        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # [batch,max_len]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

    
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        # do not calculate the padding units
        
        # [batch,max_len]
        indices = (pos != 0)
      
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        # pos_loss, neg_loss = self.loss_func(pos_logits, pos_labels), self.loss_func(neg_logits, neg_labels)
        # loss = pos_loss[indices] + neg_loss[indices]
        loss = (pos_loss + neg_loss).sum()

        return loss


    def predict(self,
                seq, 
                item_indices, 
                positions,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs'''
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
    

    def get_user_emb(self,
                     seq,
                     positions,
                     **kwargs):
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        
        return final_feat

class SASRecAug(SASRec):
    '''
    '''

    def __init__(self, user_num, item_num, device, args):
        
        super(SASRecAug, self).__init__(user_num, item_num, device, args)

    def forward(self, 
            seq, 
            pos, 
            neg, 
            positions,
            weight): # for training        
        '''
        Used to calculate pos and neg logits for loss
        -input
        seq:[batch,max_len]
        pos:[batch,max_len]
        neg:[batch,max_len]
        positions:[batch,max_len]

        '''

        # (bs, max_len, hidden_size)
        log_feats = self.log2feats(seq, positions)

        # (bs, max_len, hidden_size)
        pos_embs = self._get_embedding(pos)

        # (bs, max_len, hidden_size)
        neg_embs = self._get_embedding(neg)


        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # [batch,max_len]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        
        # [batch,max_len],[batch,max_len]
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        # do not calculate the padding units
        
        # [batch,max_len]
        indices = (pos != 0)
        
        pos_loss, neg_loss = self.loss_func(pos_logits, pos_labels), self.loss_func(neg_logits, neg_labels)
        loss = pos_loss + neg_loss
        
        weight_loss = loss*weight.unsqueeze(1)
        weight_loss = weight_loss[indices].sum()

        return weight_loss