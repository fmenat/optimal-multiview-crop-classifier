import torch

class Lambda(torch.nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)
    
class CustomTD(torch.nn.Module):
    "Custom Time-Distributed Linear layer with Conv2D kernel (1, dims)"
    def __init__(self, input_dim, output_dim, **args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.func= torch.nn.Conv2d(1, self.output_dim, (1, self.input_dim), **args)
    def forward(self, x):
        return self.func(x[:,None,:,:])[:,:,:,0].transpose(1,2)


class TemporalAttentionPool(torch.nn.Module):
    def __init__(self, input_dim, mode="key"):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.h_dim = 32

        if self.mode == "location": #the vector inside the forward is the query - original and LIn
            self.k_layer = torch.nn.Identity()
            self.qins_layer = CustomTD(self.input_dim, 1) #query inside

        elif self.mode == "key":
            self.k_layer = torch.nn.Sequential(CustomTD(self.input_dim, 32), torch.nn.Tanh())#IENCO - KEY
            self.qins_layer = CustomTD(32, 1, bias=False)#IENCO - query vector inside

        elif self.mode == "self-query":
            #you can also compute the query PSE-based -- quite variable.. require further analysis
            self.k_layer = CustomTD(input_dim, self.h_dim)
            self.q_layer = CustomTD(input_dim, self.h_dim)

        if self.mode in ["location", "key"]:
            self.score_layer = Lambda(lambda x: self.qins_layer(x[0]) ) #only based on key, not query
        else:
            self.score_layer = Lambda(lambda x: torch.bmm(x[0], x[1].transpose(1,2)) ) #it is not scaled

    def forward(self, x):
        #print("input ", x.shape)
        k = self.k_layer(x)
        #print("key",k.shape)

        if self.mode == "self-query":
            q = self.q_layer(x)
            q = q.sum(dim=1, keepdim=True) #master_query
            #print("query",q.shape)
        else:
            q = None  #model-based

        scores = self.score_layer([k, q])
        #print("scores ",scores.shape)
        distr = torch.nn.Softmax(dim=1)(scores)
        return torch.bmm(x.transpose(2,1), distr)[:,:,0] #apply agregation (temporal weighted sum)
