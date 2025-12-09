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


class CNN(torch.nn.Module):
    """
    Added to map the input image to the required encoder dimension
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # Maintain spatial dimensions
        self.norm1 = torch.nn.InstanceNorm3d(out_channels)
        self.act1 = torch.nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x

class RegistrationNet(torch.nn.Module):
    def __init__(self, 
                img_size=(64,128,128),
                patch_size=16,
                encoder_dim=768,
                mlp_dim=3072,
                masking_ratio = 0.75,   # the paper recommended 75% masked patches
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
            enc_num_heads=enc_num_heads
        )
        self.cnn_encoder = CNN(in_channels=in_channels, out_channels=1)
        self.cnn_decoder = CNN(in_channels=1, out_channels=out_channels) # Map to 3 channels for the deformation field

    def forward(self, x, masked_loss=True):
        x = self.cnn_encoder(x) # Map 2 channels to 1 channel
        output, _ = self.MAE(x, masked_loss)
        output = self.cnn_decoder(output) # Map 1 channel to 3 channels for deformation field
        return output

def main():
    batch_size = 2
    train_dir = './npdata/training'
    val_dir = './npdata/validation'
    lr = 0.0005
    epoch_start = 0
    max_epoch = 500
    reg_model = utils.register_model((64,128,128), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model((64,128,128), 'bilinear')
    reg_model_bilin.cuda()
    model = RegistrationNet(
        img_size=(64,128,128),
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
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

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]

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
    criterions += [losses.Grad3d(penalty='l2')]
    # cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)
    best_mse = 0
    writer = SummaryWriter(log_dir='MAE_TransRNet_log')

    Train_Loss = []
    MSE_val_Dice = []

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        # Start Training
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            warped_output = reg_model([x, output])
            loss = 0
            loss_vals = []
            sim_loss = criterions[0](warped_output, y)
            loss_vals.append(sim_loss)
            loss += sim_loss
            grad_loss = criterions[1](output, y)
            loss_vals.append(grad_loss)
            loss += grad_loss
            
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        Train_Loss.append(loss_all.avg)
        train_loss = np.array(Train_Loss)
        np.save('registration_experiments/bs_{}_Train_Loss_epoch_{}'.format(batch_size, epoch + 1),train_loss)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))


        # Start Validation
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                def_out = reg_model([x_seg[:, 0, ...].cuda().float(), output.cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg[:, 0, ...].long())
                eval_dsc.update(dsc.item(), x.size(0))
                print("{}-eval_dsc.avg:{}".format(idx,eval_dsc.avg))
        best_mse = max(eval_dsc.avg, best_mse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
        }, save_dir='./experiments/', filename='Dice-{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('MSE/validate', eval_dsc.avg, epoch)
        MSE_val_Dice.append(eval_dsc.avg)
        mse_val_dice = np.array(MSE_val_Dice)
        np.save('registration_experiments/bs_{}_MSE_Dice_epoch_{}'.format(batch_size, epoch + 1), mse_val_dice)
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
        loss_all.reset()
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
    os.makedirs('./registration_experiments/', exist_ok=True)
    main()