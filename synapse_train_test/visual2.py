import os
import argparse
import random

import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch import optim
from networks.bra_unet import BRAUnet
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)

    parser.add_argument('--model', default='resnet101', type=str, help='model name')

    parser.add_argument('--weights', default='/home/hutao/BRAU-Netplusplus-master/synapse_train_test/', type=str,
                        help='path of weights file. For resnet101/152, ignore this arg to download from torchvision')

    parser.add_argument('--data_path', default='path_to_imagenet', type=str, help='dataset path')

    parser.add_argument('--save_path', default='temp.npy', type=str, help='path to save the ERF matrix (.npy file)')

    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')

    parser.add_argument('--volume_path',
                        type=str,
                        default='/home/hutao/VesselSeg-Pytorch-master/data1/Synapse',
                        help='root dir for validation volume data')

    parser.add_argument('--dataset', type=str, default='Synpase', help='experiment1 name')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--list_dir', type=str, default='/home/hutao/VesselSeg-Pytorch-master', help='list dir')
    parser.add_argument('--output_dir', type=str, default='./save_baunet', help='output dir')
    parser.add_argument('--img_size', type=int, default=224, help='image size')

    parser.add_argument('Dataset', type=str, default='Synapse', help='Dataset')

    parser.add_argument('z_spacing', type=int, default=1, help='z_spacing')

    args = parser.parse_args()

    return args


def get_input_grad(model, sample):
    args = parse_args()
    h, w = sample["image"].size()[2:]

    image, label, case_name = sample['image'], sample['label'], sample['case_name'][0]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:

        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):

            slice = image[ind, :, :]

            x, y = slice.shape[0], slice.shape[1]

            if x != args.img_size or y != args.img_size:
                slice = zoom(slice, (args.img_size / x, args.img_size / y), order=3)
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float()

            model.eval()

            with torch.no_grad():

                outputs = model(input)

                # outputs = model(sample)

                output_size = outputs.size()

                central_point = torch.nn.functional.relu(outputs[:, : output_size[2] // 2, output_size[3] // 2]).sum()

                grad1 = torch.autograd.grad(central_point, sample)

                grad1 = grad1[0]

                grad1 = torch.nn.functional.relu(grad1)

                aggregated = grad1.sum((0, 1))

                grad_map = aggregated.cpu().numpy()

                prediction[ind] = grad_map
    else:
        prediction = [1]
    grad_map = prediction
    return grad_map


def main(args):
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]

    transform = transforms.Compose(t)
    dataset_config = {

        'Synapse': {
            'Dataset': Synapse_dataset,
            'z_spacing': 1
        }
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    print('reading from datapath', args.data_path)

    random.random(args.seed)
    np.random

    root = os.path.join(args.root, 'val')

    dataset = datasets.ImageFolder(root=root, transform=transform)

    sampler_val = args.Dataset(base_dir=args.volume_path, split='test_vol', img_size=args.img_size,
                               list_dir=args.list_dir)

    data_loader_val = DataLoader(dataset=sampler_val, batch_size=1, shuffle=False, num_workers=1)

    if args.model == 'resnet101':

        model = resnet101(args.weights is None)

    elif args.model == 'resnet152':

        model = resnet152(args.weights is None)

    elif args.model == 'swin_umamba':

        model = BRAUnet(img_size=224, in_chans=3, num_classes=9, n_win=7)
    else:

        raise ValueError('Unsupported model, Please add it here')

    if args.weights is not None:

        print('load weights')

        weights = torch.load(args.weights, map_location='cpu')

        if 'model' in weights:
            weights = weights['model']

        if 'state_dict' in weights:
            weights = weights['state_dict']

        model.load_state_dict(weights)

        print('load')

    model.cuda()

    model.eval()

    optimizer = optim.SGD(model.paramters(), lr=0, weight_decay=0)

    meter = AverageMeter()

    optimizer.zero_grad()

    for i_batch, samples in enumerate(data_loader_val):

        if meter.conut == args.num_images:
            np.save(args.save_path, meter.avg)

            exit()
        samples = samples.cuda(non_blocking=True)

        samples.requires_grad['image'] = True

        samples.requires_grad['label'] = True

        optimizer.zero_grad()

        contribution_scores = get_input_grad(model=model, sample=samples)

        if np.isnan(np.sum(contribution_scores)):
            print('get NAN , next image')

        else:

            print('accumulate')

            meter.update(contribution_scores)


if __name__ == '__main__':
    args = parse_args()

    main(args)


