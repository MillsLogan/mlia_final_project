from torch.utils.tensorboard import SummaryWriter
import os, glob
import new_models
from tools import utils,losses
import sys
from torch.utils.data import DataLoader
from data_pre import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

class RegistrationNet(torch.nn.Module):
    def __init__(self, 
                img_size=(64,128,128),
                patch_size=16,
                encoder_dim=768,
                mlp_dim=3072,
                masking_ratio = 0.,   # the paper recommended 75% masked patches
                decoder_dim = 512,      # paper showed good results with just 512
                dec_num_layers = 6,       # anywhere from 1 to 8
                dec_num_heads=4,
                enc_num_layers=12,
                enc_num_heads=12,
                in_channels=2,
                out_channels=3):
        super().__init__()
        self.MAE = new_models.MAETransformer(
            img_size=img_size,
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            mlp_dim=mlp_dim,
            masking_ratio = masking_ratio,
            decoder_dim = decoder_dim,
            dec_num_layers = dec_num_layers,
            dec_num_heads=dec_num_heads,
            enc_num_layers=enc_num_layers,
            enc_num_heads=enc_num_heads,
            in_channels=in_channels,
            out_channels=out_channels
        )
        # self.cnn_encoder = CNN(in_channels=in_channels, out_channels=1)
        # self.cnn_decoder = CNN(in_channels=1, out_channels=out_channels) # Map to 3 channels for the deformation field

    def forward(self, x, masked_loss=False):
        # x = self.cnn_encoder(x) # Map 2 channels to 1 channel
        output, _ = self.MAE(x, masked_loss)
        # output = self.cnn_decoder(output) # Map 1 channel to 3 channels for deformation field
        return output

def main():
    logdir = './experiments/'
    save_loss_dir = './experiments/losses/'
    os.makedirs(save_loss_dir, exist_ok=True)
    batch_size = 2
    train_dir = './npdata/training'
    val_dir = './npdata/validation'
    lr = 5e-4
    reg_weight = 0.01 # Weight for flow loss
    epoch_start = 0
    max_epoch = 500
    reg_model_val = utils.register_model((64,128,128), 'nearest')
    reg_model_val.cuda()
    reg_model_bilin_train = utils.register_model((64,128,128), 'bilinear')
    reg_model_bilin_train.cuda()
    model = RegistrationNet(
        img_size=(64,128,128),
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        masking_ratio = 0.,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        dec_num_layers = 6,       # anywhere from 1 to 8
        dec_num_heads=4,
        enc_num_layers=12,
        enc_num_heads=12,
        in_channels=2,
        out_channels=3
    )

    # Change the model to accept 2 channel input
    # first_layer = model.patch_to_encoder
    # old_weights = first_layer.weight.data
    # new_weights = old_weights.repeat_interleave(2, dim=1)
    # first_layer.weight.data = new_weights / 2.0
    # model.patch_to_encoder = first_layer
    
    updated_lr = lr

    model.cuda()
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 4 class
                                       trans.NumpyType((np.float32, np.int16)),
                                        ])

    train_set = datasets.CardiacDataset(train_dir,transforms=train_composed)
    val_set = datasets.CardiacInferDataset(val_dir,transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    x, y = next(iter(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)

    # Load MAE backbone weights into Registration Network
    # if use_pretrained == 1:
    #     print('Loading Weights from the Path {}'.format(pretrained_path))
    #     MAE_dict = torch.load(pretrained_path)
    #     MAE_weights = MAE_dict['state_dict']

    #     model_dict = model.MAE.state_dict()
    #     MAE_weights = {k: v for k, v in MAE_weights.items() if k in model_dict}
    #     model_dict.update(MAE_weights)
    #     model.MAE.load_state_dict(model_dict)
    #     del model_dict, MAE_weights, MAE_dict
    #     print('Pretrained Weights Succesfully Loaded !')

    # elif use_pretrained == 0:
    #     print('No weights were loaded, all weights being used are randomly initialized!')

    # prepare deformation loss
    # cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)
    best_mse = 0
    writer = SummaryWriter(log_dir=logdir)

    training_grad_loss_vals = []
    training_mse_loss_vals = []
    training_weighted_grad_loss_vals = []
    
    validation_grad_loss_vals = []
    validation_mse_loss_vals = []
    validation_weighted_grad_loss_vals = []
    validation_dsc_vals = []

    mse_loss = torch.nn.MSELoss()
    grad_loss = losses.Grad3d(penalty='l2')
    

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        # Start Training
        idx = 0
        tot_sim_loss = 0
        tot_grad_loss = 0
        for x, y in train_loader:
            idx += 1
            model.train()
            # Turn off learning rate adjustment
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            x = x.cuda()
            y = y.cuda()
            x_in = torch.cat((x,y), dim=1)
            deformation_field = model(x_in)
            warped_output = reg_model_bilin_train([x, deformation_field])
            sim_loss = mse_loss(warped_output, y)
            tot_sim_loss += sim_loss.item()
            flow_loss = grad_loss(deformation_field, None)
            tot_grad_loss += flow_loss.item()
            training_grad_loss_vals.append(flow_loss.item())
            training_mse_loss_vals.append(sim_loss.item())
            training_weighted_grad_loss_vals.append(flow_loss.item() * reg_weight)
            if epoch > max_epoch // 2:
                loss = sim_loss + flow_loss * reg_weight
            else:
                loss = sim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{max_epoch}], Step [{idx}/{len(train_loader)}] - Loss: {sim_loss + flow_loss * reg_weight:.4f} - Sim Loss: {sim_loss:.4f} - Flow Loss: {flow_loss * reg_weight:.4f} - Flow Loss (unweighted): {flow_loss:.4f}")
        writer.add_scalar('Loss/train', tot_sim_loss + tot_grad_loss * reg_weight, epoch)
        writer.add_scalar('Sim_Loss/train', tot_sim_loss, epoch)
        writer.add_scalar('Flow_Loss/train', tot_grad_loss * reg_weight, epoch)
        np.savez(f'{save_loss_dir}/training_loss_epoch_{epoch+1}.npz',
                 training_mse_loss=np.array(training_mse_loss_vals),
                 training_grad_loss=np.array(training_grad_loss_vals),
                 training_weighted_grad_loss=np.array(training_weighted_grad_loss_vals))
        
        print(f"Epoch [{epoch+1}/{max_epoch}] - Training Losses Saved.")


        # Start Validation
        tot_sim_loss = 0
        tot_grad_loss = 0
        dsc_per_class = 0
        with torch.no_grad():
            for idx, (x, y, x_seg, y_seg) in enumerate(val_loader):
                model.eval()
                x = x.cuda()
                y = y.cuda()
                x_seg = x_seg.cuda()
                y_seg = y_seg.cuda()

                x_in = torch.cat((x, y), dim=1)
                deformation_field = model(x_in)
                
                moving_seg = x_seg[:, 0, ...].float()
                fixed_seg = y_seg[:, 0, ...].long()

                def_out = reg_model_bilin_train([moving_seg, deformation_field])

                dsc_per_class = utils.dice_val_per_class(def_out.long(), fixed_seg)
                validation_dsc_vals.append(dsc_per_class.cpu().numpy())
                warped_output = reg_model_bilin_train([x, deformation_field])
                sim_loss = mse_loss(warped_output, y)
                tot_sim_loss += sim_loss.item()
                flow_loss = grad_loss(deformation_field, None)
                tot_grad_loss += flow_loss.item()
                validation_grad_loss_vals.append(flow_loss.item())
                validation_mse_loss_vals.append(sim_loss.item())
                validation_weighted_grad_loss_vals.append(flow_loss.item() * reg_weight)
            writer.add_scalar('Loss/validate', tot_sim_loss + tot_grad_loss * reg_weight, epoch)
            writer.add_scalar('Sim_Loss/validate', tot_sim_loss, epoch)
            writer.add_scalar('Flow_Loss/validate', tot_grad_loss, epoch)
            print(f"Epoch [{epoch+1}/{max_epoch}] - Validation Loss: {tot_sim_loss + tot_grad_loss * reg_weight:.4f} - Sim Loss: {tot_sim_loss:.4f} - Flow Loss: {tot_grad_loss:.4f}")
            np.savez(f'{save_loss_dir}/validation_loss_epoch_{epoch+1}.npz',
                     validation_mse_loss=np.array(validation_mse_loss_vals),
                        validation_grad_loss=np.array(validation_grad_loss_vals),
                        validation_weighted_grad_loss=np.array(validation_weighted_grad_loss_vals),
                        validation_dsc=np.array(validation_dsc_vals))
            
                
        if (dsc_per_class.mean().item() >= best_mse):
            best_mse = dsc_per_class.mean().item()
            print('New Best MSE: {:.4f} at epoch {}'.format(best_mse, epoch+1))

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mse': best_mse,
                'optimizer': optimizer.state_dict(),
            }, save_dir='./checkpoints/', filename='Dice-{:.3f}.pth.tar'.format(dsc_per_class.mean().item()))
        
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out.unsqueeze(0))

        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
    writer.close()

# img = img.detach().cpu().numpy()[0, 0, :, 32, :, :]
def comput_fig(img):
    # img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    img = img.detach().cpu().numpy()[0, 0, :, 32, :, :]
    fig = plt.figure(figsize=(2, 2), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def comput_fig_grid(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='Best_Trans_Reg.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    os.makedirs('./experiments/', exist_ok=True)
    os.makedirs('./checkpoints/', exist_ok=True)
    main()