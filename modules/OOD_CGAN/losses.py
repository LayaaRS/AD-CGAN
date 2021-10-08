import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda')
cuda = torch.cuda.is_available()


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1.0 - dis_out_real)) + torch.mean(F.relu(1.0 + dis_out_fake))


def loss_hinge_gen(dis_out_fake):
    return -torch.mean(dis_out_fake)


def gradient_penalty(critic, real_data, generated_data, gp_weight):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated, _, _ = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).cuda()
                                    if cuda else torch.ones(prob_interpolated.size()),
                                    create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
#         losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


def compute_infomax_loss(local_feat, global_feat, scale):
    # project the local and global features in the critic
    local_feat = torch.flatten(local_feat, start_dim=2, end_dim=3)
    global_feat = torch.unsqueeze(global_feat, 2)
    loss = infonce_loss(local_feat, global_feat)
    loss = scale * loss
    return loss


def infonce_loss(l, m):
    # create pos/neg samples for contrastive learning
    N, units, n_locals = l.size()
    _, _, n_multis = m.size()

    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)

    # Inner product for positive samples. Outer product for negative samples.
    u_p = torch.matmul(l_p, m).unsqueeze(2)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)
    # Mask the diagonal part of the negative tensor
    mask = torch.eye(N)[:, :, None, None].to(device)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (mask))  # (1 - n_mask)
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = torch.nn.functional.log_softmax(pred_lgt, dim=2)
    loss = -pred_log[:, :, 0].mean()
    return loss
