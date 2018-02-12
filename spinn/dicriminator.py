__author__ = 'Administrator'
import numpy as np

# PyTorch
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, to_gpu, MLP, Linear, HeKaimingInitializer, LayerNormalization, SimpleTreeLSTM
from spinn.util.misc import Args, Vocab

def build_goldtree(data_manager, initial_embeddings, vocab_size,
                   FLAGS, context_args, **kwargs):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_gtree = GoldenBinaryTree

    return model_gtree(model_dim=FLAGS.model_dim,
                       word_embedding_dim=FLAGS.word_embedding_dim,
                       vocab_size=vocab_size,
                       initial_embeddings=initial_embeddings,
                       embedding_keep_rate=FLAGS.embedding_keep_rate,
                       use_sentence_pair=use_sentence_pair,
                       composition_ln=FLAGS.composition_ln,
                       context_args=context_args
                       )

def build_discriminator(FLAGS, context_args, **kwargs):
    model_dis = Discriminator

    return model_dis(mlp_input_dim=FLAGS.model_dim,
                     mlp_dim=FLAGS.mlp_dim,
                     num_classes=2,
                     num_mlp_layers=FLAGS.num_mlp_layers,
                     mlp_ln=FLAGS.mlp_ln,
                     classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
                     )


class Discriminator(nn.Module):
    def __init__(self,
                 mlp_input_dim,
                 mlp_dim,
                 num_classes,
                 num_mlp_layers,
                 mlp_ln,
                 classifier_keep_rate=0.0):
        super(Discriminator, self).__init__()
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.discriminator = MLP(mlp_input_dim, mlp_dim, num_classes,
                                 num_mlp_layers, mlp_ln, self.classifier_dropout_rate)

    def forward(self, input, **kwargs):
        output = self.discriminator(input)
        return output



class GoldenBinaryTree(nn.Module):
    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=None,
                 composition_ln=None,
                 context_args=None,
                 **kwargs):
        super(GoldenBinaryTree, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.embedding_dropout_rate = 1. - embedding_keep_rate
        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # embedding layer
        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors)

        # sequential encoder layer
        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

        # binary tree lstm encoder
        self.binary_tree_lstm = SimpleTreeLSTM(
            model_dim / 2,
            composition_ln=composition_ln
        )

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        embeds = self.embed(x)
        embeds = self.reshape_input(embeds, batch_size, seq_length)  # batch_size, seq_length, emb_dim
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)  # batch_size, seq_length, model_dim
        embeds = torch.cat([b.unsqueeze(0)
                            for b in torch.chunk(embeds, batch_size, 0)], 0)
        embeds = F.dropout(
            embeds,
            self.embedding_dropout_rate,
            training=self.training)

        return embeds

    def forward(self,
                sentences,
                left_nodes,
                right_nodes,
                write_nodes,
                example_lengths=None,
                **kwargs):
        x, example_lengths = self.unwrap(sentences, example_lengths)
        emb = self.run_embed(x)

        batch_size, seq_len, model_dim = emb.data.size()
        # example_lengths_var = to_gpu(
        #     Variable(torch.from_numpy(example_lengths))).long()

        l, r, w, w_mask = self.unwrap_tree(left_nodes, right_nodes, write_nodes)

        max_depth = l.size(1)

        nonterminal_inits = to_gpu(Variable(torch.zeros(batch_size, seq_len, model_dim)))
        leaf_inputs = torch.cat([emb, nonterminal_inits], 1)
        leaf_inputs = leaf_inputs.view(batch_size*2*seq_len, -1)
        l_s = l[:, 0]
        r_s = r[:, 0]
        w_s = w[:, 0]
        #print(w_s)
        #print(str(batch_size) +" : " +str(seq_len))
        #print(torch.arange(l_s.data.size(0)).long() * (batch_size * 2 * seq_len))
        ls_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + l_s
        rs_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + r_s
        ws_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + w_s
        #print(leaf_inputs.size(0))
        #print(leaf_inputs.size(1))
        
        #print(ws_idx)
        lefts_hidden = leaf_inputs[ls_idx]
        rights_hidden = leaf_inputs[rs_idx]
        state = self.binary_tree_lstm(lefts_hidden, rights_hidden)
        #print(state)
        leaf_inputs[ws_idx] = state
        for i in range(1, max_depth - 1):
            l_t = l[:, i]
            r_t = r[:, i]
            w_t = w[:, i]
            w_m = w_mask[:, i]
            w_m = w_m.unsqueeze(1).float()
            #print(w_m)
            l_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + l_t
            r_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + r_t
            w_idx = to_gpu(Variable(torch.arange(batch_size).long() * (2 * seq_len), volatile=not self.training)) + w_t
            left_hidden = leaf_inputs[l_idx]
            right_hidden = leaf_inputs[r_idx]
            new_state = self.binary_tree_lstm(left_hidden, right_hidden)
            leaf_inputs[w_idx] = new_state
            #print(state.size(0))
            #print(state.size(1))
            #print(new_state.size(0))
            #print(new_state.size(1))
            #print(w_m.size(0))
            #print(new_state)
            state = (1 - w_m) * state + w_m * new_state
            
        #h = self.wrap(state)
        #ft = self.build_features(h)

        return state

    def unwrap_tree(self, lefts, rights, writes):
        max_len = lefts.shape[1]
        left_prem = lefts[:, :, 0]
        left_hyp = lefts[:, :, 1]
        left = np.concatenate([left_prem, left_hyp], axis=0)
        right_prem = rights[:, :, 0]
        right_hyp = rights[:, :, 1]
        right = np.concatenate([right_prem, right_hyp], axis=0)
        write_prem = writes[:, :, 0]
        write_hyp = writes[:, :, 1]
        write = np.concatenate([write_prem, write_hyp], axis=0)
        #print("left")
        #print(left)
        #print("write")
        #print(write)

        l = to_gpu(Variable(torch.from_numpy(left), volatile=not self.training))
        r = to_gpu(Variable(torch.from_numpy(right), volatile=not self.training))
        w = to_gpu(Variable(torch.from_numpy(write), volatile=not self.training))

        l = l - (l.ge(200).int() * (200 - max_len))
        #print("left new")
        #print(l)
        r = r - (r.ge(200).int() * (200 - max_len))
        w = w - (w.ge(201).int() * (201 - max_len))
        w_mask = w.ge(0).long()
        w = w + (w.le(0).int() * (2 * max_len))
        #print("write new")
        #print(w)
        #print("write mask")
        #print(w_mask)
        return l.long(), r.long(), w.long(), w_mask

    def unwrap(self, sentences, lengths=None):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, lengths)
        return self.unwrap_sentence(sentences, lengths)

    def unwrap_sentence_pair(self, sentences, lengths=None):
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        if lengths is not None:
            len_prem = lengths[:, 0]
            len_hyp = lengths[:, 1]
            lengths = np.concatenate([len_prem, len_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(
            x), volatile=not self.training)), lengths

    def unwrap_sentence(self, sentences, lengths=None):
        return to_gpu(Variable(torch.from_numpy(sentences),
                               volatile=not self.training)), lengths

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            features = torch.cat(features, 1)
        else:
            features = h
        return features

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = ([hh[:batch_size], hh[batch_size:]])
        return h

    def wrap_sentence(self, hh):
        return hh

    # --- From Choi's 'treelstm.py' ---
