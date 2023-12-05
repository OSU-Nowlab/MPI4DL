import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from dataloders import WSIDataloader


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size: {size_all_mb:.3f} MB")


def get_gpu_utilization():
    # import pynvml
    # pynvml.nvmlInit()

    # gpu_count = torch.cuda.device_count()
    # for i in range(gpu_count):
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #     info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    #     print(f"GPU {i} Utilization: {info.gpu}%")
    # pynvml.nvmlShutdown()
    pass


def load_custom_dataset(batch_size, image_size):
    # batch_size = 32
    num_workers = 1

    # transform = transforms.Compose(
    #     [ transforms.Resize(image_size), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    torch.manual_seed(0)

    # testset = torchvision.datasets.ImageNet(
    #         root="/home/gulhane.2/GEMS_Inference/datasets/ImageNet/", split='val', transform=transform
    # )
    testset = torchvision.datasets.ImageFolder(
        root="/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/single_gpu/imagenette2-320/val",
        transform=transform,
        target_transform=None,
    )

    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def load_fake_dataset(batch_size, image_size, times):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.FakeData(
        size=10 * batch_size * times,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
        target_transform=None,
        random_offset=0,
    )
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size * times,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader


def load_wsi_dataset():
    batch_size = 1
    num_workers = 1

    # transform = transforms.Compose([
    #     transforms.Resize((tile_size, tile_size)),  # Adjust the size as needed
    #     transforms.ToTensor(),
    # ])
    testset = WSIDataloader.WSIDataloader(
        root="/home/gulhane.2/GEMS_Inference/datasets/test_digital_pathology/CAMELYON16",
        transform=None,
    )

    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return dataloader


def load_dataset(app, batch_size, image_size, times):
    if app == 1:
        return load_custom_dataset(batch_size, image_size)
    if app == 2:
        return load_wsi_dataset()
    if app == 3:
        return load_fake_dataset(batch_size, image_size, times)


def load_torchResNet(device, precision):
    CHECKPOINT_PATH = "/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth"
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet50()
    model.load_state_dict(checkpoint)
    model.eval()
    if precision == "fp_16":
        model.half()
    model.to(device)
    return model


def load_torchResNetCustomClass(device, num_classes):
    CHECKPOINT_PATH = "/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth"
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet50()
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.eval()
    model.to(device)
    return model


# Get the depth of ResNet model based on version and number of ResNet Blocks
# This parameter will used for ResNet model architecture.
def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


def mpi4dlResNetQuantization():
    from torch.ao.quantization import QConfigMapping
    import torch.ao.quantization.quantize_fx as quantize_fx
    import copy
    from custom_models import resnet

    # model_fp = resnet50()

    model_fp = resnet.get_resnet_v2(
        (int(batch_size / parts), 3, image_size, image_size),
        depth=get_depth(2, resnet_n),
        num_classes=num_classes,
    )

    model_to_quantize = copy.deepcopy(model_fp)
    model_to_quantize.eval()
    qconfig_mapping = QConfigMapping().set_global(
        torch.ao.quantization.default_dynamic_qconfig
    )
    input_fp32 = torch.randn(1, 3, 256, 256)
    example_inputs = input_fp32

    model_prepared = quantize_fx.prepare_fx(
        model_to_quantize, qconfig_mapping, example_inputs
    )

    model_quantized = quantize_fx.convert_fx(model_prepared)
    model_quantized.to("cuda")
    input_fp32 = input_fp32.to("cuda")
    res = model_quantized(input_fp32)
    print(res.shape)


def mpi4dlResNet(
    device, batch_size, parts, image_size, resnet_n, num_classes, precision
):
    from custom_models import resnet

    CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee.pth"
    checkpoint = torch.load(CHECKPOINT_PATH)

    model = resnet.get_resnet_v2(
        (int(batch_size / parts), 3, image_size, image_size),
        depth=get_depth(2, resnet_n),
        num_classes=num_classes,
    )

    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if precision == "fp_16":
        model.half()

    model.to(device)
    return model


def load_model(
    device, batch_size, parts, image_size, resnet_n, num_classes, MPI4DL, precision
):
    if MPI4DL:
        return mpi4dlResNet(
            device, batch_size, parts, image_size, resnet_n, num_classes, precision
        )

    # Using PyTorch
    if num_classes == 1000:
        return load_torchResNet(device, precision)
    return load_torchResNetCustomClass(device, num_classes, precision)


# def tile_image(image, tile_size = 256):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(tile_size),
#         transforms.ToTensor()
#     ])
#     total_tiles_h = image.shape[1] // tile_size
#     total_tiles_w = image.shape[2] // tile_size

#     tiles = []
#     for i in range(total_tiles_h):
#         for j in range(total_tiles_w):
#             tile = image[:, i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
#             tile = transform(tile)
#             tiles.append(tile)
#     return torch.stack(tiles)


def evaluate_wsi(dataloader, model, device, tile_size):
    import openslide

    corrects = 0

    perf = []
    count = 0
    print("Staring WSI ")
    get_gpu_utilization()
    # For now, considering batch-size of 1 when using tiling. Need to think of optimization.
    for wsi_imag_path, labels in dataloader:
        slide = openslide.OpenSlide(wsi_imag_path[0])
        # print(" WSI 1: {count} ")
        # get_gpu_utilization()
        width, height = slide.dimensions
        print(f"WSI Image width and height : {width} {height}")
        t = time.time()
        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        start_event.record()
        with torch.no_grad():
            for x_pix in range(0, slide.dimensions[0], tile_size):
                for y_pix in range(0, slide.dimensions[1], tile_size):
                    region = slide.read_region(
                        (x_pix, y_pix), 0, (tile_size, tile_size)
                    )
                    pil_img = region.convert("RGB")  # RGBA to RGB

                    transform = transforms.ToTensor()
                    tensor_image = transform(pil_img)

                    batch_image = tensor_image.unsqueeze(dim=0)

                    batch_image, labels = batch_image.to(device), labels.to(device)

                    outputs = model(batch_image)

                    _, predicted = torch.max(outputs, 1)
                    corrects += (predicted == labels).sum().item()
                    count += 1
                    del batch_image
                    torch.cuda.empty_cache()
        end_event.record()
        torch.cuda.synchronize()
        t = start_event.elapsed_time(end_event) / 1000
        perf.append(1 / t)
        print(f"Time Taken : {t} Throughput : {1 / t}")

        slide.close()
    accuracy = (
        corrects / count
    )  # per slide accuracy  : need to rethink , we might look for per image accuracy
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


# Batchs size = 8: Mean 40.3627477810076 Median 40.46250402955721
# Accuracy : 0.74828025477707
def evaluate_traditional(dataloader, model, device, precision):
    corrects = 0
    t = time.time()
    perf = []
    count = 0
    size_dataset = len(dataloader.dataset)
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            if count > math.floor(size_dataset / (times * batch_size)) - 1:
                print(f"Batch :  {count}")
                break
            count += 1
            inputs, labels = data
            if precision == "fp_16":
                inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(
                    device, dtype=torch.float16
                )
            else:
                inputs, labels = inputs.to(device), labels.to(device)

            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            outputs = model(inputs)

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            perf.append(len(inputs) / t)

            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
    accuracy = corrects / len(dataloader.dataset)
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


def evaluate(device, model, dataloader, WSI_IMAGES, image_size, precision):
    if WSI_IMAGES:
        accuracy = evaluate_wsi(dataloader, model, device, tile_size=image_size)
    else:
        accuracy = evaluate_traditional(dataloader, model, device, precision)
    return accuracy


device = "cuda:0"
APP = 2

## FOR WSI IMAGES
# APP = 2
# MPI4DL = True
# WSI_IMAGES = True

# FOR Fake Dataset
APP = 3
MPI4DL = True
WSI_IMAGES = False  # set image-size = 64, num_classes = 10 for imagenette checkpoint

## FOR ImageNette or ImageNet
# APP = 1
# MPI4DL = True
# WSI_IMAGES = False

batch_size = 2
parts = 1
image_size = 2048
resnet_n = 12
num_classes = 2
precision = "fp_16"  # values [fp_32, fp_16]

times = 1
if WSI_IMAGES:
    assert APP == 2, "Use Pathology Dataset"
model = load_model(
    device,
    batch_size=batch_size,
    parts=parts,
    image_size=image_size,
    resnet_n=resnet_n,
    num_classes=num_classes,
    MPI4DL=MPI4DL,
    precision=precision,
)
print_model_size(model)
dataloader = load_dataset(
    app=APP, batch_size=batch_size, image_size=image_size, times=times
)
accuracy = evaluate(device, model, dataloader, WSI_IMAGES, image_size, precision)
print(f"Accuracy with pretrained model : {accuracy * 100}")
# for image_size in [512, 1024, 2048]:
#     print(f"image_size : {image_size}")
#     model = load_model(device, batch_size = batch_size, parts = parts, image_size = image_size, resnet_n = resnet_n, num_classes = num_classes, MPI4DL = MPI4DL)
#     dataloader = load_dataset(app = APP, batch_size = batch_size, image_size = image_size, times = times)
#     accuracy = evaluate(device, model, dataloader, WSI_IMAGES, image_size)
#     print(f"Accuracy with pretrained model : {accuracy * 100}")

# mpi4dlResNetQuantization()
