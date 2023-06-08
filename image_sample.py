"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch as th
import torch.distributed as dist
import torchvision as tv

from guided_diffusion.image_datasets import load_data

# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage, signal
from pooling import MedianPool2d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
SNR_DICT = {100: 0.0,
            30: 0.05,
            25: 0.08,
            20: 0.13,
            15: 0.22,
            10: 0.36,
            5: 0.6,
            1: 0.9}

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    # logger.configure()
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to("cuda")

    print("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)

    print("sampling...")
    all_samples = []
    
    device = "cuda:0"
    for i, (batch, cond) in enumerate(data):
        # print(cond)
        # print(batch)
        # print(batch.size())
        # print(cond["label_ori"].size())
        # print(cond["label"].size())
        # print("Is 188 in label?", 188 in cond["label"])
        # image = ((batch + 1.0) / 2.0).cuda()
        # label = (cond['label_ori'].float() / 255.0).cuda()

        image = ((batch + 1.0) / 2.0).to(device)
        label = (cond['label_ori'].float() / 255.0).to(device)

        sample = image[0].cpu().numpy()
        sample = np.transpose(sample, (1,2,0))
        plot_label = cond['label'][0].cpu().numpy()
        plot_label = plot_label.squeeze(0)
        plot_label2 = cond['label_ori'][0].cpu().numpy()
        plot_label2 = plot_label2

        # plt.subplot(1,3,1)
        # plt.imshow(sample)
        # plt.subplot(1,3,2)
        # plt.imshow(plot_label)
        # plt.subplot(1,3,3)
        # plt.imshow(plot_label2)
        # plt.savefig("./test.png")

        # model_kwargs = preprocess_input(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label, pool=None)
        model_kwargs = preprocess_input_FDS(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label)
        # model_kwargs, cond = preprocess_input(cond, one_hot_label=args.one_hot_label, add_noise=args.add_noise, noise_to=args.noise_to)


        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, image.shape[2], image.shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0
        print("Sample statistics:", th.mean(sample), th.max(sample))

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_samples.extend([sample.cpu().numpy()])

        for j in range(sample.shape[0]):
            tv.utils.save_image(sample[j], "./sample.png")
            # tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            # tv.utils.save_image(sample[j], os.path.join(sample_path + "_SNR" + str(args.snr), cond['path'][j].split('/')[-1].split('.')[0] + '_SNR' + str(args.snr) + '_pool' + str(args.pool) + '.png'))
            # tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0]  + '.png'))
            tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('\\')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(sample[j], os.path.join(sample_path + "_SNR" + str(args.snr), cond['path'][j].split('\\')[-1].split('.')[0] + '_SNR' + str(args.snr) + '_pool' + str(args.pool) + '.png'))
            tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('\\')[-1].split('.')[0]  + '.png'))


        print(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    print("sampling complete")


def preprocess_input(args, data, num_classes, one_hot_label=True):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    if one_hot_label:
        bs, _, h, w = label_map.size()
        input_label = th.FloatTensor(bs, num_classes, h, w).zero_()

        # print("label_map.size()", label_map.size())

        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if 'instance' in data:
            inst_map = data['instance']
            instance_edge_map = get_edges(inst_map)
            input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)
    else:
        label_map = data['label']
        if 'instance' in data:
            # print("Instance in data")
            inst_map = data['instance']
            instance_edge_map = get_edges(inst_map)
            input_semantics = th.cat((label_map, instance_edge_map), dim=1)

    # print("Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))
    # input_semantics = (input_semantics - th.mean(input_semantics)) / th.std(input_semantics)
    # input_semantics = (input_semantics - th.min(input_semantics)) / (th.max(input_semantics - th.min(input_semantics)))
    # print("After norm: Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))

    # SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
    noise = th.randn(input_semantics.shape, device=input_semantics.device)*SNR_DICT[args.snr]
    input_semantics += noise
    print("Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))
    input_semantics = (input_semantics - th.min(input_semantics)) / (th.max(input_semantics) - th.min(input_semantics))
    print("Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))

    if args.pool == "med":
        print("Using Median filter")
        med_filter = MedianPool2d(padding=1, same=True)
        input_semantics_clean = med_filter(input_semantics)
    if args.pool == "mean":
        print("Using Average filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = avg_filter(input_semantics)
    else:
        input_semantics_clean = input_semantics

    # input_semantics_clean = ndimage.median_filter(input_semantics.numpy(), size=20, mode="nearest")
    # input_semantics_clean = np.array([])
    # for map in input_semantics:
    #     clean_map = np.array([])
    #     for channel in map:
    #         print(channel.shape)
    #         clean_channel = signal.medfilt2d(channel.numpy())
    #         clean_map = np.concatenate([clean_map, clean_channel], axis=0)
    #     input_semantics_clean = np.concatenate([input_semantics_clean, clean_map], axis=0)
    # input_semantics_clean = th.tensor(input_semantics_clean)
    # input_semantics = (input_semantics - th.mean(input_semantics)) / th.std(input_semantics)
    # print("After norm: Min, Mean, Max", th.min(input_semantics_clean), th.mean(input_semantics_clean), th.max(input_semantics_clean))
    # input_semantics = (input_semantics - th.min(input_semantics)) / (th.max(input_semantics - th.min(input_semantics)))

    plt.figure(figsize=(30,30))
    for idx, channel in enumerate(input_semantics_clean[0]):
        plt.subplot(6,6,idx+1)
        plt.imshow(channel.numpy(), cmap="gray")
        plt.axis("off")
    plt.savefig("./seg_map.png")

    return {'y': input_semantics_clean}

def preprocess_input_FDS(args, data, num_classes, one_hot_label=True):
    
    pool = "max"
    label_map = data['label'].long()

    # create one-hot label map
    # label_map = label.unsqueeze(0)
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
#     print("label map shape:", label_map.shape)

    input_semantics = input_label.scatter_(1, label_map, 1.0)
    print(input_semantics.shape)
    map_to_be_discarded = []
    map_to_be_preserved = []
    input_semantics = input_semantics.squeeze(0)
    for idx, segmap in enumerate(input_semantics.squeeze(0)):
        if 1 in segmap:
            map_to_be_preserved.append(idx)
        else:
            map_to_be_discarded.append(idx)

    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics.unsqueeze(0), instance_edge_map), dim=1)
        #add instance map to map indexes
        map_to_be_preserved.append(num_classes)
        num_classes += 1

    print(input_semantics.shape, len(map_to_be_preserved))

    # input_semantics = input_semantics[map_to_be_preserved].unsqueeze(0)
    input_semantics = input_semantics[0][map_to_be_preserved]


    # if pool != None:
    #     avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    #     if 'instance' in data:
    #         instance_edge_map = avg_filter(instance_edge_map)
    #         input_semantics = th.cat((input_semantics.unsqueeze(0), instance_edge_map), dim=1)
    noise = th.randn(input_semantics.shape, device=input_semantics.device)*SNR_DICT[args.snr]

    input_semantics += noise

    if pool == "med":
        print("Using Median filter")
        med_filter = MedianPool2d(padding=1, same=True)
        input_semantics_clean = med_filter(input_semantics)
    elif pool == "mean":
        print("Using Average filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # avg_filter2 = th.nn.AvgPool2d(kernel_size=5, stride=1, padding=1)
        input_semantics_clean = avg_filter(input_semantics)
    elif pool == "max":
        print("Using Max filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        max_filter = th.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = max_filter(avg_filter(input_semantics))

    else:
        input_semantics_clean = input_semantics

#     print("After norm: Min, Mean, Max", torch.min(input_semantics_clean), torch.mean(input_semantics_clean), torch.max(input_semantics_clean))
    # print("-->", input_semantics_clean.shape)
    input_semantics_clean = input_semantics_clean.unsqueeze(0)
    
    # Insert non-classes maps
#     print("input_semantics_clean", input_semantics_clean.shape)
    input_semantics = th.empty(size=(input_semantics_clean.shape[0],\
                                        num_classes, input_semantics_clean.shape[2],\
                                        input_semantics_clean.shape[3]), device=input_semantics_clean.device)
    # print("input_semantics", input_semantics.shape)
    # print("Preserved:", map_to_be_preserved, len(map_to_be_preserved))
    # print("Discarded:", map_to_be_discarded, len(map_to_be_discarded))
    # print("input_semantics_clean", input_semantics_clean[0].shape)
    input_semantics[0][map_to_be_preserved] = input_semantics_clean[0]
    input_semantics[0][map_to_be_discarded] = th.zeros((len(map_to_be_discarded), input_semantics_clean.shape[2], input_semantics_clean.shape[3]), device=input_semantics_clean.device)
    
    # plt.figure(figsize=(30,30))
    # for idx, channel in enumerate(input_semantics[0]):
    #     plt.subplot(6,6,idx+1)
    #     plt.imshow(channel.numpy(), cmap="gray")
    #     plt.axis("off")
    # plt.savefig("./seg_map.png")

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        num_classes=35,
        s=1.0,
        snr=100,
        pool="med",
        add_noise=False,
        noise_to="semantics",
        unet_model="unet" #"unet", "spadeboth", "enconly"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
