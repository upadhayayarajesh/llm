import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import tensorly as tl
from tensorly.decomposition import tensor_train
import math

def get_tensor_shape(shape):

    tt_shape = shape
    return tt_shape

def get_tt_rank(r, tt_shape):
    tt_rank = [1]
    for i in range(len(tt_shape)-1):
        tt_rank.append(r)
    tt_rank.append(1)
    return tt_rank

class LoRATTLinearWrapper(nn.Module):
        def __init__(self, module: nn.Module, flag, tt_shape, tt_rank, alpha:int):
            super().__init__()

            self.base_module = module
            self.flag = flag  ### for future, currently not required
            self.in_features, self.out_features = self.base_module.weight.shape

            # self.bottleneck = bottleneck
            self.alpha=alpha
            self.tt_rank = tt_rank
            self.tt_shape = tt_shape
            self.W_delta=torch.zeros((self.in_features, self.out_features)) ### create a torch matrix W_delta of shape (in_feature, out_feature) and initialize iit to all 0s
            self.reset_parameters() ### this function will basically allocate random valuesa from gaussian distribution to W_delta


            #make 10d torch
            self.W_10d = self.reshape_to_10d(torch.tensor(self.W_delta)) ### decompose the W_delta to high dimentional tensor based on the TT shapes
            # self.W_up_5d = self.reshape_to_10d(torch.tensor(self.W_up))

            #tt cores
            ### create model parameters. The paramaters will be multiple tensors, where the shape of each tensor is determined by provided ranks and TT shapes.
            ### Now, these tensors will be initialized randomly. Later, we'll transfer the values of W_delta to these paramaters
            self.tt_cores = nn.ParameterList([nn.Parameter(self.init_core(*shape)) for shape in self.get_tt_shapes()])
            # tl.set_backend('pytorch')

            ### using tensor train, decompose the W_delta into multiple tensors based on the ranks and shapes provided
            self.tt_cores_dummy = tensor_train(self.W_10d.detach().numpy(), self.tt_rank)

            ### transfer the values of tt_cores_dummy to self.tt_cores which are the newly added parameters of the model
            for i in range(len(self.tt_cores)):
                self.tt_cores[i].data = torch.tensor(self.tt_cores_dummy[i], dtype=torch.float32)

            self.tt_cores.requires_grad= True ### make self.tt_cores trainable

            self.base_module.weight.requires_grad = False ### make base_module's parameters non-trainable

            ### MAKE THE BIAS NON-TRAINABL
            if self.base_module.bias is not None:
                    self.base_module.bias.requires_grad = False
            # self.reset_parameters()


        def get_tt_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes

        def reshape_to_10d(self, tensor):
            return tensor.reshape(*self.tt_shape)

        def reset_parameters(self):

            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.W_delta, a=math.sqrt(8))
            # nn.init.kaiming_uniform_(self.tt_cores_up, a=math.sqrt(5))

        def init_core(self, *shape):
            std = 1.0 / math.sqrt(shape[1])
            return torch.randn(*shape) * std

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.alpha > 0:
                # device = torch.device("cuda")
                # self.core_weights_down = torch.einsum('ijk,klm,mno,opq,qrs->jlnpr', torch.tensor(self.tt_cores_down[0]), torch.tensor(self.tt_cores_down[1]), torch.tensor(self.tt_cores_down[2]), torch.tensor(self.tt_cores_down[3]), torch.tensor(self.tt_cores_down[4]))
                ### need to multiply all the tensors to gvet the original shape of the high dimensional tensor (the tensor before decomposition)
                if len(self.tt_shape) == 4:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq->jlnp', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3])
                if len(self.tt_shape) == 6:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu->jlnprt', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5])
                if len(self.tt_shape) == 7:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw->jlnprtv', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6])
                if len(self.tt_shape) == 8:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy->jlnprtvx', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7])
                if len(self.tt_shape) == 10:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc->jlnprtvxzb', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7], self.tt_cores[8], self.tt_cores[9])
                if len(self.tt_shape) == 12:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc,cde,efg->jlnprtvxzbdf', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7], self.tt_cores[8], self.tt_cores[9], self.tt_cores[10], self.tt_cores[11])

                
                # adapted_weight = self.base_module.weight + self.alpha * (self.core_weights_down.reshape(self.in_features, self.bottleneck) @ self.core_weights_up.reshape(self.bottleneck, self.out_features))
                adapted_weight = self.base_module.weight + self.alpha * (self.tt_weights.reshape(self.in_features, self.out_features))
                # adapted_weight = torch.einsum('ij,klmnop->ij', self.base_module.weight + self.alpha * (self.tt_weights))

                return F.linear(x, adapted_weight, self.base_module.bias)
            else:
                return F.linear(x, self.base_module.weight, self.base_module.bias)