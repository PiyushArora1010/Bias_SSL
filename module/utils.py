import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
import random
from numpy.random import RandomState

def evaluate_accuracy_LfF_mem(mw_model, test_loader, memory_loader, target_attr_idx, device):
  mw_model.eval()
  mw_correct = 0
  mem_iter = iter(memory_loader)
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)
        try:
            indexm,memory_input,_ = next(mem_iter)
        except:
            mem_iter = iter(memory_loader)
            indexm,memory_input,_ = next(mem_iter)
        memory_input = memory_input.to(device)

        mw_outputs = mw_model(data, memory_input)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def evaluate_accuracy_LfF(mw_model, test_loader, target_attr_idx, device, param = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def evaluate_rotation(model, test_loader, device, param1 = -1, param2 = -1):
  model.eval()
  correct = 0
  with torch.no_grad():
    for _, images, labels in test_loader:
        labels = torch.zeros(len(labels))
        images_90 = TF.rotate(images, 90)
        labels_90 = torch.ones(len(labels))
        images_180 = TF.rotate(images, 180)
        labels_180 = torch.ones(len(labels))*2
        images_270 = TF.rotate(images, 270)
        labels_270 = torch.ones(len(labels))*3
        images = torch.cat((images, images_90, images_180, images_270), dim=0)
        labels = torch.cat((labels, labels_90, labels_180, labels_270), dim=0)
        images = images.to(device)
        labels = labels.to(device)
        del images_90, images_180, images_270, labels_90, labels_180, labels_270

        outputs  = model(images)
        pred = outputs.data.max(1, keepdim=True)[1]

        correct += pred.eq(labels.data.view_as(pred)).sum().item()
  accuracy = 100.*(torch.true_divide(correct,len(test_loader.dataset)*4)).item()
  model.train()
  return accuracy

def evaluate_accuracy_simple(mw_model, test_loader, target_attr_idx, device, param1 = -1):
  mw_model.eval()
  mw_correct = 0
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)

        mw_outputs  = mw_model(data)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy

def concat_dummy(z):
    def hook(model, input, output):
        z.append(output.squeeze())
        return torch.cat((output, torch.zeros_like(output)), dim=1)
    return hook

def evaluate_accuracy_LDD(model_b, model_l, data_loader, target_idx,device, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0

        for index, data, attr in data_loader:
            label = attr[:, target_idx]

            data = data.to(device)
            label = label.to(device)


            with torch.no_grad():

                try:
                    z_l, z_b = [], []
                    hook_fn = model_l.avgpool.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()
                    
                    z_l = z_l[0]
                    hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                except:
                    z_l, z_b = [], []
                    hook_fn = model_l.layer4.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()
                    
                    z_l = z_l[0]
                    hook_fn = model_b.layer4.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]    

                z_origin = torch.cat((z_l, z_b), dim=1)

                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin)

                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = 100*total_correct/float(total_num)
        model_b.train()
        model_l.train()

        return accs.item()

def evaluate_accuracy_LDD_MW(model_b, model_l, data_loader, mem_loader, target_idx, device, model='label'):
        model_b.eval()
        model_l.eval()
        mem_iter_ = None

        total_correct, total_num = 0, 0

        for index, data, attr in data_loader:
            label = attr[:, target_idx]

            data = data.to(device)
            label = label.to(device)
            try:
                indexm, datam, labelm = next(mem_iter_)
            except:
                mem_iter_ = iter(mem_loader)
                indexm, datam, labelm = next(mem_iter_)
            
            datam = datam.to(device)

            with torch.no_grad():

                try:
                    z_l, z_b = [], []
                    hook_fn = model_l.avgpool.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data, data)
                    hook_fn.remove()
                    z_l = z_l[0]

                    hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_lm, z_bm = [], []
                    hook_fn = model_l.avgpool.register_forward_hook(concat_dummy(z_lm))
                    _ = model_l(datam, datam)
                    hook_fn.remove()
                    z_lm = z_lm[0]

                    hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_bm))
                    _ = model_b(datam)
                    hook_fn.remove()
                    z_bm = z_bm[0]

                except:
                    z_l, z_b = [], []
                    hook_fn = model_l.layer4.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()
                    
                    z_l = z_l[0]
                    hook_fn = model_b.layer4.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]    

                    z_lm, z_bm = [], []
                    hook_fn = model_l.layer4.register_forward_hook(concat_dummy(z_lm))
                    _ = model_l(datam, datam)
                    hook_fn.remove()
                    z_lm = z_lm[0]

                    hook_fn = model_b.layer4.register_forward_hook(concat_dummy(z_bm))
                    _ = model_b(datam)
                    hook_fn.remove()
                    z_bm = z_bm[0]

                z_origin = torch.cat((z_l, z_b), dim=1)
                mem_features_ = torch.cat((z_lm, z_bm), dim=1)

                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin, mem_features_)

                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = 100*total_correct/float(total_num)
        model_b.train()
        model_l.train()

        return accs.item()

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def write_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text)
        f.write('\n')

def rotate_tensor(tensor, angle):
    tensor = TF.rotate(tensor, angle)
    return tensor

dic_functions = {
    'MW_LfF MW_LfF_Rotation MW': evaluate_accuracy_LfF_mem,
    'LfF LfF_Rotation': evaluate_accuracy_LfF,
    'Rotation': evaluate_rotation,
    'Simple': evaluate_accuracy_simple,
    'set_seed': set_seed,
    'write_to_file': write_to_file,
    'LDD': evaluate_accuracy_LDD,
    'LDD_MW': evaluate_accuracy_LDD_MW,
    'rotate_tensor': rotate_tensor
}