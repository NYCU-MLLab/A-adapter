import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Union, Callable
from accelerate import Accelerator


class PGD(object):
    """
    https://arxiv.org/abs/1706.06083
    """

    def __init__(self, model, emb_name="word_embeddings", epsilon=1., alpha=0.3, K=3):
        """Doing PGD attack, default epsilon= 1., alpha(trade-off)= 0.3, K(round)= 3 times"""
        # emb_name = your model's embbeding name, e.g., Bert -> Bert.embeddings.word_embeddings
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
        self.K = K
        self.accelerator = Accelerator()

    def get_delta(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
    
    def attack(self, batch):
        self.backup_grad()
        for t in range(self.K):
            self.get_delta(is_first_attack=(t == 0))
            if t != self.K-1:
                self.model.zero_grad()
            else:
                self.restore_grad()
            outputs_adv = self.model(**batch)
            loss_adv = outputs_adv.loss
            self.accelerator.backward(loss_adv)
        self.restore()


class FGM(object):
    """
    https://arxiv.org/abs/1605.07725
    """

    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        """Doing FGM attack, default epsilon=1."""
        # emb_name = your model's embbeding name, e.g., Bert -> Bert.embeddings.word_embeddings
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
        self.accelerator = Accelerator()

    def get_delta(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
    def attack(self, batch):
        self.get_delta()
        # just do once
        outputs_adv = self.model(**batch)
        loss_adv = outputs_adv.loss
        self.accelerator.backward(loss_adv)
        # restore the parameters in embbeding
        self.restore()


class FreeLB(object):
    '''
    https://arxiv.org/abs/1909.11764
    '''

    def __init__(self, adv_K=3, adv_lr=1e-1, adv_init_mag=2e-2, adv_max_norm=1.0, adv_norm_type='l2', base_model='bert'):
        """
        Simple setting: adv_K * adv_lr ≈ adv_max_norm
        In FreeLB appendix A.2, {eps: adv_max_norm, alpha: adv_lr, m: adv_K}
        GLUE tasks Hyper-parameters:
                  eps     adv_lr   k
            MNLI  2e-1    1E-1     2
            QNLI  1.5e-1  1E-1     2
            QQP   4.5e-1  1.5E-1   2
            RTE   1.5e-1  3E-2     3
            SST-2 6e-1    1E-1     2
            MRPC  4e-1    4E-2     3
            CoLA  2e-1    2.5E-2   3
            STS-B 3e-1    1E-1     3
            WNLI  1e-2    5e-3     2
        """
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def get_embeds_init(self, model, batch):
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(
                model.module, self.base_model).embeddings.word_embeddings(batch["input_ids"])
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])

        return embeds_init

    def getDelta(self, embeds_init, batch):
        attention_mask = batch["attention_mask"]
        delta = None
        batch_size = embeds_init.shape[0]
        length = embeds_init.shape[-2]
        dim = embeds_init.shape[-1]
        # check shape
        attention_mask = attention_mask.view(-1, length)
        embeds_init = embeds_init.view(-1, length, dim)
        if self.adv_init_mag > 0:      
            input_mask = attention_mask.to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(
                    embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                # 1/sqrt(Ns) * U(-eps,eps) ... Alg.Line 4
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(
                    embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            # no delta
            delta = torch.zeros_like(embeds_init)

        return delta.view(batch_size, length, dim)

    def updateDelta(self, delta, delta_grad, embeds_init):
        batch_size = delta.shape[0]
        length = delta.shape[-2]
        dim = delta.shape[-1]
        delta = delta.view(-1, length, dim)
        delta_grad = delta_grad.view(-1, length, dim)

        if self.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(
                delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            # ... Alg. Line 11, update delta via gradient ascend
            delta = (delta + self.adv_lr * delta_grad / denorm).detach()
            if self.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(
                    delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                reweights = (self.adv_max_norm / delta_norm *
                             exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.adv_max_norm)
                delta = (delta * reweights).detach()
        elif self.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(
                0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + self.adv_lr * delta_grad / denorm).detach()
            if self.adv_max_norm > 0:
                delta = torch.clamp(delta, -self.adv_max_norm,
                                    self.adv_max_norm).detach()
            else:
                raise ValueError(
                    "Norm type {} not specified.".format(self.adv_norm_type))

        return delta.view(batch_size, length, dim)


class ALUM(object):
    """
    https://arxiv.org/abs/2004.08994
    """

    def __init__(self, adv_lr=1e-3, alpha=1, noise_rate=1e-5, adv_norm_type='l2', base_model='bert'):
        self.eps = 1e-6
        self.adv_lr = adv_lr
        self.alpha = alpha
        self.noise_rate = noise_rate
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def kl(self, inputs, targets, reduction="sum"):
        loss = F.kl_div(F.log_softmax(inputs, dim=-1),
                        F.softmax(targets, dim=-1),
                        reduction=reduction)
        return loss

    def adv_project(self, grad):
        if self.adv_norm_type == 'l2':
            direction = grad / \
                (torch.norm(grad, dim=-1, keepdim=True) + self.eps)
        elif self.adv_norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.eps)
        return direction

    def get_embeds(self, model, batch):
        if isinstance(model, torch.nn.DataParallel):
            embeds = getattr(model.module, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])
        else:
            embeds = getattr(model, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])
        return embeds

    def VAT(self, model, batch, logits):
        embed = self.get_embeds(model, batch)
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        # Line.3 sample from Gaussian
        noise = embed.data.new(embed.size()).normal_(0, 1) * self.noise_rate
        noise.requires_grad_()
        # Line.4 as origin paper, select K=1 for reducing computation
        new_embed = embed.data.detach() + noise
        adv_output = model(inputs_embeds=new_embed,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        adv_logits = adv_output.logits
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()

        # if gradient vanishing or explosion
        if torch.isnan(norm) or torch.isinf(norm):
            return None

        # Line.6
        noise = noise + self.adv_lr * delta_grad
        # projection
        noise = self.adv_project(noise)
        new_embed = embed.data.detach() + noise
        new_embed = new_embed.detach()
        # Line.8
        adv_output = model(inputs_embeds=new_embed,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        adv_logits = adv_output.logits
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        # adversarial loss: For pre-trained->10, fine-tuning->1
        adv_loss = (adv_loss_f + adv_loss_b) * self.alpha

        return adv_loss
    

class SMARTLoss(nn.Module):
    """
    To measure the different between normal x and adversarial example x+delta
    """
    def __init__(
        self,
        model: Callable,
        inner_loop: int = 2,
        alpha: float = 1e-3, 
        epsilon: float = 1e-5,
        init_noise: float = 1e-5,
        lambdas: int = 10,
        base_model: str = "bert"
        ):
        super().__init__()
        self.eval_fn = self.eval 
        self.loss_fn = self.to_list(self.loss)
        self.norm_fn = self.do_norm
        self.inner_loop = inner_loop 
        self.alpha = alpha
        self.epsilon = epsilon 
        self.init_noise = init_noise
        self.lambdas = lambdas
        self.model = model
        self.base_model = base_model
    
    def to_list(self, x):
        return x if isinstance(x, list) else [x]

    def do_norm(self, x, norm_type=float('inf')):
        return torch.norm(x, p=norm_type, dim=-1, keepdim=True)
    
    def get_embeds(self, model, batch):
        if isinstance(model, torch.nn.DataParallel):
            embeds = getattr(model.module, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])
        else:
            embeds = getattr(model, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])
        return embeds
    
    def loss(self, inputs, target, reduction='batchmean', loss_type="KL"):
        """
        Default: KL_Div to reduce computation, or you can choose Jeffray Div, or use JS Div.
        """
        if loss_type == "KL":
            KL_Div = F.kl_div( F.log_softmax(inputs, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
            return KL_Div
        elif loss_type == "JF":
            Jeffray_Div = F.kl_div(
                F.log_softmax(inputs, dim=-1, dtype=torch.float32),
                F.softmax(target.detach(), dim=-1, dtype=torch.float32),
                reduction=reduction,
            ) + F.kl_div(
                F.log_softmax(target, dim=-1, dtype=torch.float32),
                F.softmax(inputs.detach(), dim=-1, dtype=torch.float32),
                reduction=reduction,
            )
            return Jeffray_Div
        else:
            m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + F.softmax(inputs.detach(), dim=-1, dtype=torch.float32)
            m = 0.5 * m
            JS_Div = F.kl_div(
                F.log_softmax(inputs, dim=-1, dtype=torch.float32), m, reduction=reduction) + F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
            
            return JS_Div
    
    def eval(self, embed, batch):
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.model(inputs_embeds=embed, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        return logits

    def forward(self, embed: Tensor, state: Union[Tensor, List[Tensor]], batch) -> Tensor:
        states = self.to_list(state)
        delta = torch.randn_like(embed, requires_grad=True) * self.init_noise

        for i in range(self.inner_loop + 2):
            # +delta.requires_grad_()
            delta.requires_grad_()
            # Compute perturbation
            adversarial_example = embed + delta
            states_perturbed = self.to_list(self.eval_fn(adversarial_example, batch))
            adv_loss = 0
            # Compute adversarial loss
            for batch_number in range(len(states)):
                adv_loss += self.loss_fn[batch_number](states_perturbed[batch_number], states[batch_number].detach())
            if i == self.inner_loop + 1: 
                return adv_loss
            # Compute delta gradient+clone().detach()
            delta_gradient = torch.autograd.grad(adv_loss, delta)[0].clone().detach()
            # Gradient ascent 
            new_delta = delta + self.alpha * delta_gradient
            delta_norm = self.norm_fn(new_delta)
            delta = new_delta / (delta_norm + self.epsilon)
            # Reset noise gradients for next step
            # delta = delta.detach().requires_grad_()
            
class Aadapter(object):
    def __init__(self, adv_K=3, adv_lr=1e-1, adv_init_mag=2e-2, adv_max_norm=1.0, adv_norm_type='l2', base_model='bert'):
        """
        Simple hyperparameters setting: adv_K * adv_lr ≈ adv_max_norm
        """
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def get_embeds_init(self, model, batch):
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(
                model.module, self.base_model).embeddings.word_embeddings(batch["input_ids"])
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(
                batch["input_ids"])

        return embeds_init

    def getDelta(self, embeds_init, batch):
        attention_mask = batch["attention_mask"]
        delta = None
        batch_size = embeds_init.shape[0]
        length = embeds_init.shape[-2]
        dim = embeds_init.shape[-1]
        # check shape
        attention_mask = attention_mask.view(-1, length)
        embeds_init = embeds_init.view(-1, length, dim)
        if self.adv_init_mag > 0:      
            input_mask = attention_mask.to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(
                    embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                # 1/sqrt(Ns) * U(-eps,eps) ... Alg.Line 4
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(
                    embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            # no delta
            delta = torch.zeros_like(embeds_init)

        return delta.view(batch_size, length, dim)

    def updateDelta(self, delta, delta_grad, embeds_init):
        batch_size = delta.shape[0]
        length = delta.shape[-2]
        dim = delta.shape[-1]
        delta = delta.view(-1, length, dim)
        delta_grad = delta_grad.view(-1, length, dim)

        if self.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(
                delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + self.adv_lr * delta_grad / denorm).detach()
            if self.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(
                    delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                reweights = (self.adv_max_norm / delta_norm *
                             exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()
        elif self.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(
                0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + self.adv_lr * delta_grad / denorm).detach()
            if self.adv_max_norm > 0:
                delta = torch.clamp(delta, -self.adv_max_norm,
                                    self.adv_max_norm).detach()
            else:
                raise ValueError(
                    "Norm type {} not specified.".format(self.adv_norm_type))

        return delta.view(batch_size, length, dim)
    
    def KL_term(self, model, batch_store, batch):
        # origin P
        outputs = model(input_ids=batch_store,
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        labels=batch['labels'])
        logits_p = outputs.logits
        # estimate Q
        adv_outputs = model(input_ids=None,
                            inputs_embeds= batch["inputs_embeds"],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['labels'])
        logits_q = adv_outputs.logits
        KL_Div = -F.kl_div( F.log_softmax(logits_q, dim=-1, dtype=torch.float32), F.softmax(logits_p, dim=-1, dtype=torch.float32), reduction="batchmean")
        
        return KL_Div