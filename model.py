import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassPredictor(nn.Module):
    def __init__(self, input_size, num_classes, drop_prob):
        super(ClassPredictor, self).__init__()
        
        self.input_dout = nn.Dropout(drop_prob) 
        
        hidden_1 = 120
        hidden_2 = 80        
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.l_relu1 = nn.LeakyReLU()
        self.dout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.l_relu2 = nn.LeakyReLU()
        self.dout2 = nn.Dropout(0.1)

        self.out = nn.Linear(hidden_2, num_classes)
        
        nn.init.orthogonal_(self.fc1.weight).requires_grad_().cuda()
        nn.init.orthogonal_(self.fc2.weight).requires_grad_().cuda()
        nn.init.orthogonal_(self.out.weight).requires_grad_().cuda()


    def forward(self, x):
        ## x: (input_size)

        # Manually use dropout for the Segment BiGRU output
        x = self.input_dout(x)
        
        a1 = self.fc1(x)
        h1 = self.l_relu1(a1)
        dout1 = self.dout1(h1)

        a2 = self.fc2(dout1)
        h2 = self.l_relu2(a2)
        dout2 = self.dout2(h2)

        # y: (num_classes)
#         y = self.out(h2)
        y = self.out(dout2)

        return y


class BiLSTM_model(nn.Module):

	def __init__(self, input_dim, hidden_dim, n_layers, classes):
		super(BiLSTM_model, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.classes = classes

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

		self.fc = ClassPredictor(self.hidden_dim*2, self.classes, 0.2)

	def forward(self, x):

		hidden = (torch.zeros(self.n_layers*2, x.shape[0], self.hidden_dim).to('cuda'), torch.zeros(self.n_layers*2, x.shape[0], self.hidden_dim).to('cuda'))

		out, hidden = self.lstm(x, hidden)
		# out_maxpool = F.adaptive_max_pool1d(out.permute(0,2,1),1).squeeze()
		out = torch.mean(out, dim=1).squeeze()

		# out = torch.cat([out_maxpool, out_avgpool])

		# print(out.shape)
		out = self.fc(out)

		return out


class BiGRU_model(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, classes):
        super(BiGRU_model, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.classes = classes

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

        self.fc = ClassPredictor(self.hidden_dim*2, self.classes, 0.2)

    def forward(self, x):

        hidden = torch.zeros(self.n_layers*2, x.shape[0], self.hidden_dim).to('cuda')

        out, hidden = self.gru(x, hidden)
        # out_maxpool = F.adaptive_max_pool1d(out.permute(0,2,1),1).squeeze()
        out = torch.mean(out, dim=1).squeeze()

        # out = torch.cat([out_maxpool, out_avgpool])

        # print(out.shape)
        out = self.fc(out)

        return out


class new_model(nn.Module):

    def __init__(self, input_dim, hidden_dim, segment_hidden_dim, n_layers, classes):
        super(new_model, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.segment_hidden_dim = segment_hidden_dim
        self.n_layers = n_layers
        self.classes = classes

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

        self.segment_lstm = nn.LSTM(self.hidden_dim*2, self.segment_hidden_dim, self.n_layers, batch_first=True, bidirectional=True)

        self.fc = ClassPredictor(self.segment_hidden_dim*2, self.classes, 0.2)

    def init_hidden_state(self, batch_size):
        # h1 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to('cuda').float()
        # h2 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to('cuda').float()
        # return (h1, h2)

        # h0 = torch.empty(self.n_layers * 2, batch_size, self.hidden_dim).float()
        # h0 = nn.init.orthogonal_(h0)
        # h0 = h0.requires_grad_().to('cuda')
        # return h0

        h1 = torch.empty(self.n_layers * 2, batch_size, self.hidden_dim).float()
        h1 = nn.init.orthogonal_(h1)
        h1 = h1.requires_grad_().to('cuda')
        h2 = torch.empty(self.n_layers * 2, batch_size, self.hidden_dim).float()
        h2 = nn.init.orthogonal_(h2)
        h2 = h2.requires_grad_().to('cuda')
        return (h1, h2)

    def init_segment_hidden_state(self, batch_size):
        # h1 = torch.zeros(self.n_layers*2, batch_size, self.segment_hidden_dim).to('cuda').float()
        # h2 = torch.zeros(self.n_layers*2, batch_size, self.segment_hidden_dim).to('cuda').float()
        # return (h1, h2)

        # sh0 = torch.empty(self.n_layers * 2, batch_size, self.segment_hidden_dim).float()
        # sh0 = nn.init.orthogonal_(sh0)
        # sh0 = sh0.requires_grad_().to('cuda')
        # return sh0

        h1 = torch.empty(self.n_layers * 2, batch_size, self.segment_hidden_dim).float()
        h1 = nn.init.orthogonal_(h1)
        h1 = h1.requires_grad_().to('cuda')
        h2 = torch.empty(self.n_layers * 2, batch_size, self.segment_hidden_dim).float()
        h2 = nn.init.orthogonal_(h2)
        h2 = h2.requires_grad_().to('cuda')
        return (h1, h2)
        

    def forward(self, x, segment_indices, labels):
        batch_size = x.size(0)

        x = x.float()
        h = self.init_hidden_state(batch_size)
        out, _ = self.lstm(x, h)

        # print('Out shape: ', out.shape)
        concat_input= []
        concat_labels = []
        for i in range(len(segment_indices)-1):
            start_segment = int(segment_indices[i])
            end_segment = int(segment_indices[i+1])

            segment_features = out[:, start_segment:end_segment, :]
            segment_label = labels[start_segment]

            concat_input.append(segment_features)
            concat_labels.append(segment_label)

        # print('Concat labels shape: ', len(concat_labels))
        # print('Concat inputs shape: ', len(concat_input))
        segment_h = self.init_segment_hidden_state(batch_size)

        full_features = []
        for inp in concat_input:
            segment_out, _ = self.segment_lstm(inp, segment_h)
            segment_out = F.adaptive_max_pool1d(segment_out.permute(0,2,1),1).squeeze()
            full_features.append(segment_out.unsqueeze(0))

        # print('!! The shape: !! ', torch.stack(full_features, 1).squeeze(0).shape)
        out = self.fc(torch.stack(full_features, 1).squeeze(0))

        return out, torch.tensor(concat_labels).to('cuda')



# if __name__ == "__main__":

# 	model = BiLSTM_model(10, 30, 3)
# 	inp = torch.randn(32, 15, 10)

# 	hidden_state = torch.randn(6, 32, 30)
# 	cell_state = torch.randn(6, 32, 30)

# 	hidden = (hidden_state, cell_state)

# 	out, hidden = model(inp, hidden)

# 	print('Shape of out: ', out.shape, '\n')
# 	print('Shape of hidden: ', hidden[0].shape, '\n')
