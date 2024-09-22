import torch.nn.functional as F
import torch

def Consistent_compare(scoremap1, scoremap2, loss_attenion_cri, mode='None', size=[72, 24]):

    if mode == 'scaling':

        localization_map_normed_upsample1 = F.upsample_bilinear(scoremap1, size)
        localization_map_normed_upsample2 = F.upsample_bilinear(scoremap2, size)
        attention_loss = loss_attenion_cri(localization_map_normed_upsample1, localization_map_normed_upsample2)  #求mseloss
        return attention_loss
    else:
        return loss_attenion_cri(scoremap1, scoremap2)