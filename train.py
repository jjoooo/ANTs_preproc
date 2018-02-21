import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score

# GPU training
def training(args, tr_batch, models, loss_fn, optimizer, cnt, model_path):

    # writing loss
    if not os.path.exists('./loss'):
        os.makedirs('./loss')

    loss_path = './loss/lr{}_ps{}_bce_loss'.format(args.learning_rate, args.patch_size)
    if os.path.isfile(loss_path):
        file_loss = open(loss_path, 'a')
        file_loss.write('\n\n------------------------------------------------------------\n\n')
    else:
        file_loss = open(loss_path, 'w')

    print('\nTraining start...')
    resnet_s = models[0]
    resnet_b = models[1]
    classifier = models[2]

    for img,_,p in tr_batch:

        optimizer.zero_grad()
        
        mid = int(args.patch_size/2)
        x1 = Variable(img[:,:,:mid]).cuda()
        x2 = Variable(img[:,:,mid:]).cuda()

        out_s = resnet_s.forward(x1)  
        out_b = resnet_b.forward(x2)

        concat_out = torch.cat([out_s,out_b],dim=1)

        out = classifier.forward(concat_out)

        target = Variable(_).float().cuda()
        target = target.view(args.batch_size,-1)

        loss = loss_fn(out, target)

        file_loss.write('batch {} \t: loss {}\n'.format(cnt-1, loss.data.cpu().numpy()[0]))
        print('batch {} \t-------> loss {}'.format(cnt-1, loss.data.cpu().numpy()[0]))

        loss.backward()
        
        optimizer.step()

        if cnt % 100 ==0:
            torch.save(resnet_s.state_dict(),model_path+'/mode_{}/miccai_{}.pkl'.format(args.n_mode,0))
            torch.save(resnet_b.state_dict(),model_path+'/mode_{}/miccai_{}.pkl'.format(args.n_mode,1))
            torch.save(classifier.state_dict(),model_path+'/mode_{}/miccai_{}.pkl'.format(args.n_mode,2))
        cnt += 1

        out_arr = out.data.cpu().numpy()  
        tar_arr = _.numpy()

    models = [resnet_s, resnet_b, classifier]
 
    print('Train done.')
    
    return models, cnt

def training_cpu(self, cnt, model_path):
    pass