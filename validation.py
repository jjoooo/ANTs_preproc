import os

import torch
import torch.nn as nn
from torch.autograd import Variable


# GPU validation
def validation(args, val_batch, models, ep):

    # Writing accuracy
    if not os.path.exists('./acc'):
        os.makedirs('./acc')
    
    acc_path = './acc/lr{}_ps{}_acc'.format(args.learning_rate, args.patch_size)
    if os.path.isfile(acc_path):
        file_acc = open(acc_path, 'a')
        file_acc.write('\n\n------------------------------------------------------------\n\n')
    else:
        file_acc = open(acc_path, 'w')

    ac = 0.0
    total = 0
    dsc_total = 0
    sum_out = 0
    ac_zero = 0.0
    thsd = 0.5
    print('\nValidation start...')
    resnet_s = models[0]
    resnet_b = models[1]
    classifier = models[2]

    for img,_,p in val_batch:
        mid = int(args.patch_size/2)

        x1 = Variable(img[:,:,:mid]).cuda()
        x2 = Variable(img[:,:,mid:]).cuda()

        out_s = resnet_s.forward(x1)
        out_b = resnet_b.forward(x2)

        concat_out = torch.cat([out_s,out_b],dim=1)

        out = classifier.forward(concat_out)
        out = nn.Sigmoid()(out)
        out = out.view(args.batch_size,-1)

        target = Variable(_).float().cuda()
        target = target.view(args.batch_size,-1)

        # for accuracy calc
        for b in range(args.batch_size):

            out_val = out.data.cpu().numpy()[b,0]
            target_val = target.data.cpu().numpy()[b,0]
        
            if target_val == 1:
                dsc_total += 1
                if out_val > thsd:
                    ac += 1
            else:
                if out_val <= thsd:
                    ac_zero += 1
            if out_val  > thsd:
                dsc_total += 1
            sum_out += out_val
          

    print('predict avg = {}, dsc = {}%, accuracy = {}%'.format(sum_out/(args.batch_size*len(val_batch)), (2*ac)/dsc_total*100, (ac+ac_zero)/(args.batch_size*len(val_batch))*100))
    file_acc.write('Epoch {} : predict avg = {}, dsc = {}%, accuracy = {}%\n'.format(ep, sum_out/(args.batch_size*len(val_batch)), (2*ac)/dsc_total*100, (ac+ac_zero)/(args.batch_size*len(val_batch))*100))
    print('Validation done.\n') 
