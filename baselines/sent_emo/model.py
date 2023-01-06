# baseline model for sentiment and emotion classification

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification


class RobertaBase(nn.Module):

    def __init__(self):
        super(RobertaBase, self).__init__()

        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', cls_token="[CLS]",
                                                                      sep_token="[SEP]")
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 256)

    def forward(self, text_inputs):
        # tokenize and complete forward pass with roberta over the batch of inputs
        self.tokenizer.tokenize()

class MultimodalModelBase(nn.Module):
    """
    An encoder to take a sequence of inputs and
    produce a sequence of intermediate representations
    todo: try replacing text rnn with BERT model
    """
    def __init__(self, params):
        super(MultimodalModelBase, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        self.text_rnn = nn.LSTM(
            input_size=self.text_input_size,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # self.text_batch_norm = nn.BatchNorm1d(num_features=params.text_gru_hidden_dim)

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_acoustic_rnn_lyrs,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        if params.add_avging:
            # set size of input dim
            self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
            # set size of hidden
            self.fc_hidden = 50
            # set acoustic fc layer 1
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, self.fc_hidden)
            # self.ac_fc_batch_norm = nn.BatchNorm1d(self.fc_hidden)
        else:
            # set size of input dim
            self.fc_input_dim = (
                params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim
            )
            # set size of hidden
            self.fc_hidden = 100
            # set acoustic fc layer 1
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, self.fc_hidden)

        # set acoustic fc layer 2
        self.acoustic_fc_2 = nn.Linear(self.fc_hidden, params.audio_dim)

        # initialize speaker, gender embeddings
        self.speaker_embedding = None
        self.gender_embedding = None

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
            self.speaker_embedding = nn.Embedding(
                params.num_speakers, params.speaker_emb_dim
            )

        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim
            self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        # self.fc_batch_norm = nn.BatchNorm1d(params.fc_hidden_dim)

        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        acoustic_len_input=None,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()

            short_embs = F.dropout(self.short_embedding(text_input), 0.1)

            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            all_embs = text_input

        # flatten_parameters() decreases memory usage
        self.text_rnn.flatten_parameters()

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)
        # encoded_text = self.text_batch_norm(encoded_text)

        if acoustic_len_input is not None:
            print("acoustic length input is used")
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input.clamp(max=1500),
                batch_first=True,
                enforce_sorted=False,
            )
            (
                packed_acoustic_output,
                (acoustic_hidden, acoustic_cell),
            ) = self.acoustic_rnn(packed_acoustic)
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)

        else:
            if len(acoustic_input.shape) > 2:
                # get average of dim 1, as this is the un-averaged acoustic info
                encoded_acoustic = torch.mean(acoustic_input, dim=1)
            else:
                encoded_acoustic = acoustic_input

        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        )
        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        )

        # combine modalities as required by architecture
        inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        # return the output
        return output


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """

    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.output_dim
        self.inter_fc_prediction_dim = params.final_hidden_dim
        self.dropout = params.dropout

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.inter_fc_prediction_dim)
        self.fc2 = nn.Linear(self.inter_fc_prediction_dim, self.output_dim)

    def forward(self, combined_inputs):
        out = torch.relu(F.dropout(self.fc1(combined_inputs), self.dropout))
        out = torch.relu(self.fc2(out))

        return out


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """
    def __init__(self, params):
        super(MultitaskModel, self).__init__()
        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = MultimodalModelBase(params)

        # set output layers
        self.sent_predictor = PredictionLayer(params, params.output_0_dim)
        self.emo_predictor = PredictionLayer(params, params.output_1_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        acoustic_len_input=None,
    ):
        # call forward on base model
        final_base_layer = self.base(
            acoustic_input,
            text_input,
            length_input=length_input,
            acoustic_len_input=acoustic_len_input,
        )

        sent_out = self.sent_predictor(final_base_layer)
        emo_out = self.emo_predictor(final_base_layer)

        return sent_out, emo_out
