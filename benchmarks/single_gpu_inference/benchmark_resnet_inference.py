import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from dataloders import WSIDataloader
from threading import Thread
import subprocess as sp
from torchgems import parser

max_gpu_util = 0
program_running = True
parser_obj = parser.get_parser()
args = parser_obj.parse_args()


def get_gpu_memory():
    while program_running:
        output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(
                sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
            )[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        global max_gpu_util
        max_gpu_util = max(max_gpu_util, memory_use_values[0])


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size: {size_all_mb:.3f} MB")


# def print_model_parameters(model):
#     total_params = 0
#     total_buffers = 0
#     print(model)

#     for name, param in model.named_parameters():
#         param_size = param.numel()
#         total_params += param_size
#         print(f'{name}: {param_size} parameters')

#     for name, buffer in model.named_buffers():
#         buffer_size = buffer.numel()
#         total_buffers += buffer_size
#         print(f'{name}: {buffer_size} buffers')

#     print(f'Total number of parameters: {total_params}')
#     print(f'Total number of buffers: {total_buffers}')


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
    elif precision == "int_8":
        model = torchResNetQuantization()
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


def mpi4dlResNetQuantizationWithTensorRT(
    device, batch_size, parts, image_size, resnet_n, num_classes, precision
):
    import torch_tensorrt
    from custom_models import resnet
    import os

    print(
        f"mpi4dlResNetQuantizationWithTensorRT : {batch_size} {image_size} {num_classes} {precision}"
    )

    # print(f"In mpi4dlResNetQuantizationWithTensorRT")
    # if os.path.exists("res_mpi4dl_quant_model_int4.pth"):
    if False:
        model = torch.jit.load("res_mpi4dl_quant_model.pth")
        model.to(device)
        return model
    else:
        CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee_tRT.pth"
        checkpoint = torch.load(CHECKPOINT_PATH)
        model = resnet.get_resnet_v2(
            (int(batch_size / parts), 3, image_size, image_size),
            depth=get_depth(2, resnet_n),
            num_classes=num_classes,
        )
        # model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        testing_dataloader = load_fake_dataset(batch_size, image_size, times)
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device(device),
        )

        print(f"precision : {precision}")
        trt_mod = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input((batch_size, 3, image_size, image_size))],
            enabled_precisions={torch.int8},
            calibrator=calibrator,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                #  "disable_tf32": False
            },
        )
        path = f"res_mpi4dl_quant_model_img_{image_size}_batch_{batch_size}_classes_{num_classes}.pth"
        torch.jit.save(torch.jit.script(trt_mod), path)
        model_size = os.path.getsize(path) / (1024 * 1024)
        print(f"Model Size: {model_size:.2f} MB")
        os.remove(path)
        state_path = f"res_mpi4dl_quant_model_img_{image_size}_batch_{batch_size}_classes_{num_classes}_state.pth"
        # torch.save(trt_mod.state_dict(),state_path)
        # print(trt_mod.state_dict())
        # trt_mod.to(device)
        return trt_mod


def torchResNetQuantization():
    import torch.ao.quantization.quantize_fx as quantize_fx
    import copy

    from torch.ao.quantization.backend_config import get_tensorrt_backend_config_dict

    trt_qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
        ),
        weight=torch.ao.quantization.default_weight_observer,
    )
    trt_backend_config_dict = get_tensorrt_backend_config_dict()

    CHECKPOINT_PATH = "/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth"
    checkpoint = torch.load(CHECKPOINT_PATH)
    model_fp = resnet50()
    model_fp.load_state_dict(checkpoint)

    model_to_quantize = copy.deepcopy(model_fp)
    model_to_quantize.eval()

    input_fp32 = torch.randn(1, 3, 64, 64)
    model_prepared = quantize_fx.prepare_fx(
        model_to_quantize,
        {"": trt_qconfig},
        input_fp32,
        backend_config=trt_backend_config_dict,
    )

    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized


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
    elif precision == "int_8":
        model = mpi4dlResNetQuantizationWithTensorRT(
            device, batch_size, parts, image_size, resnet_n, num_classes, precision
        )

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
def evaluate_traditional(dataloader, model, device, batch_size, precision):
    print(f"Evaluate Traditional ...")
    corrects = 0
    perf = []
    size_dataset = len(dataloader.dataset)
    t = time.time()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            if batch > math.floor(size_dataset / (times * batch_size)) - 1:
                break

            inputs, labels = data
            if precision == "fp_16":
                inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(
                    device, dtype=torch.float16
                )
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            perf.append(batch_size / t)
            assert batch_size == len(inputs)
            t = time.time()

            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
    accuracy = corrects / len(dataloader.dataset)
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


def evaluate(device, model, dataloader, WSI_IMAGES, batch_size, image_size, precision):
    if WSI_IMAGES:
        accuracy = evaluate_wsi(
            dataloader, model, device, batch_size, tile_size=image_size
        )
    else:
        accuracy = evaluate_traditional(
            dataloader, model, device, batch_size, precision
        )
    return accuracy


device = "cuda:0"
max_memory_before = torch.cuda.max_memory_allocated(device=device)
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

batch_size = 1
parts = 1
image_size = args.image_size
resnet_n = 12
num_classes = 10
# precision = "fp_16"  # values [fp_32, fp_16, int_8]
precision = args.precision

times = 1
if WSI_IMAGES:
    assert APP == 2, "Use Pathology Dataset"
# model = load_model(
#     device,
#     batch_size=batch_size,
#     parts=parts,
#     image_size=image_size,
#     resnet_n=resnet_n,
#     num_classes=num_classes,
#     MPI4DL=MPI4DL,
#     precision=precision,
# )
# print_model_size(model)
# # print_model_parameters(model)
# dataloader = load_dataset(
#     app=APP, batch_size=batch_size, image_size=image_size, times=times
# )
# accuracy = evaluate(device, model, dataloader, WSI_IMAGES, image_size, precision)
# print(f"Accuracy with pretrained model : {accuracy * 100}")

# # Track the maximum memory allocated again
# max_memory_after = torch.cuda.max_memory_allocated(device=device)
# print(f"Max Memory Before Using PyTorch CUDA: {max_memory_before / (1024 ** 2):.2f} MB")
# print(f"Max Memory After Using PyTorch CUDA: {max_memory_after / (1024 ** 2):.2f} MB")

# for image_size in [512, 1024, 2048]:
#     print(f"image_size : {image_size}")
#     model = load_model(device, batch_size = batch_size, parts = parts, image_size = image_size, resnet_n = resnet_n, num_classes = num_classes, MPI4DL = MPI4DL)
#     dataloader = load_dataset(app = APP, batch_size = batch_size, image_size = image_size, times = times)
#     accuracy = evaluate(device, model, dataloader, WSI_IMAGES, image_size)
#     print(f"Accuracy with pretrained model : {accuracy * 100}")

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    print(
        f"*************** Using Config Info Image Size : {image_size} Batch Size {batch_size} Precision : {precision} Num Classes {num_classes}***************"
    )
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
    max_gpu_util = 0
    program_running = True
    t1 = Thread(target=get_gpu_memory, name="t1")
    t1.start()
    max_memory_before = torch.cuda.max_memory_allocated(device=device)
    accuracy = evaluate(
        device, model, dataloader, WSI_IMAGES, batch_size, image_size, precision
    )
    print(
        f"*************** Run Summary for Config Info Image Size : {image_size} Batch Size {batch_size} Precision : {precision} Num Classes {num_classes}***************"
    )
    print(f"Accuracy with pretrained model : {accuracy * 100}")

    program_running = False
    print(f"GPU Utilization nvidia-smi : {max_gpu_util}")
    max_memory_after = torch.cuda.max_memory_allocated(device=device)
    print(
        f"Max Memory Before Using PyTorch CUDA: {max_memory_before / (1024 ** 2):.2f} MB"
    )
    print(
        f"Max Memory After Using PyTorch CUDA: {max_memory_after / (1024 ** 2):.2f} MB"
    )
