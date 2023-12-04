import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# from torchvision.models import resnet50
# from torchvision.models import resnet18
from torchvision.models import resnet101
from dataloders import WSIDataloader
from torchgems.utils import get_gpu_memory_used

# from torchgems.utils import get_gpu_memory

# parser = argparse.ArgumentParser()
# parser.add_argument("--precision", help = "Precision", choices = ["fp_32", "fp_16", "int_8"])
# parser.add_argument("--image-size", help = "Image Size")
# parser.add_argument("--batch-size", help = "Batch Size")
from torchgems import parser

parser_obj = parser.get_parser()
args = parser_obj.parse_args()


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


def load_torchResNet(device, batch_size, image_size, num_classes, precision):
    # CHECKPOINT_PATH = "/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth"
    # checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet101(num_classes=num_classes)
    checkpoint = resnet101(pretrained=True).state_dict()

    # Assuming pretrain = True
    if num_classes != checkpoint["fc.weight"].size(0):
        del checkpoint["fc.weight"], checkpoint["fc.bias"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    if precision == "fp_16":
        model.half()
    elif precision == "bfp_16":
        assert (
            torch.cuda.is_bf16_supported() == True
        ), "Native System doen't support bf16"
        model = model.to(torch.bfloat16)
    elif precision == "int_8":
        # model = torchResNetQuantization(model)
        model = torchResNetQuantizationWithTensorRT(
            model, device, batch_size, image_size, num_classes, precision
        )
    model.to(device)
    return model


def load_torchResNetCustomClass(device, num_classes):
    CHECKPOINT_PATH = "/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth"
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet101()
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
    from models import resnet
    import os

    saved_checkpt = (
        f"res_mpi4dl_quant_model_{image_size}_{num_classes}_{batch_size}.pth"
    )
    # print(f"In mpi4dlResNetQuantizationWithTensorRT")
    # :
    if os.path.exists(saved_checkpt):
        # if False:
        print(f"Using saved int_8 checkpoint at : {saved_checkpt}")
        model = torch.jit.load(saved_checkpt)
        return model
    else:
        model = resnet.get_resnet_v2(
            (int(batch_size / parts), 3, image_size, image_size),
            depth=get_depth(2, resnet_n),
            num_classes=num_classes,
        )
        if APP == 1 and image_size == 64:
            CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee_TensorRT_model_temp.pth"
            checkpoint = torch.load(CHECKPOINT_PATH)
            print("Using imagenetee checkpoint..")
            model.load_state_dict(checkpoint["model_state_dict"])

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
                "disable_tf32": False,
            },
        )
        torch.jit.save(torch.jit.script(trt_mod), saved_checkpt)

        # torch.save(trt_mod.state_dict(),'res_torch_quant_model_stats.pth')
        # trt_mod.to(device)
        return trt_mod


def torchResNetQuantizationWithTensorRT(
    model, device, batch_size, image_size, num_classes, precision
):
    import torch_tensorrt

    import os

    saved_checkpt = (
        f"res_latest_torch_quant_model_{image_size}_{num_classes}_{batch_size}.pth"
    )
    # print(f"In mpi4dlResNetQuantizationWithTensorRT")
    # :
    if os.path.exists(saved_checkpt):
        # if False:
        print(f"Using saved checkpoint at : {saved_checkpt}")
        model = torch.jit.load(saved_checkpt)
        return model
    else:
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
                "disable_tf32": False,
            },
        )
        torch.jit.save(torch.jit.script(trt_mod), saved_checkpt)
        size = os.path.getsize(saved_checkpt)
        # os.remove(saved_checkpt)
        print(f"checkpont file size : {size / 1024 ** 2}")
        # torch.save(trt_mod.state_dict(),'res_torch_quant_model_stats.pth')
        # trt_mod.to(device)
        return trt_mod


def torchResNetQuantization(model_fp):
    import torch.ao.quantization.quantize_fx as quantize_fx
    import copy

    from torch.ao.quantization.backend_config import get_tensorrt_backend_config_dict

    trt_qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
        ),
        weight=torch.ao.quantization.default_weight_observer.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
        ),
    )
    trt_backend_config_dict = get_tensorrt_backend_config_dict()

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
    from models import resnet

    model = resnet.get_resnet_v2(
        (int(batch_size / parts), 3, image_size, image_size),
        depth=get_depth(2, resnet_n),
        num_classes=num_classes,
    )
    if APP == 1 and image_size == 64:
        # CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee.pth"
        # CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee_TensorRT_model.pth"
        CHECKPOINT_PATH = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee_tRT.pth"
        checkpoint = torch.load(CHECKPOINT_PATH)
        print("Using imagenetee checkpoint..")
        print(f"Using epoch : {checkpoint['epoch']} loss : {checkpoint['loss']}")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if precision == "fp_16":
        model.half()
    elif precision == "bfp_16":
        assert (
            torch.cuda.is_bf16_supported() == True
        ), "Native System doen't support bf16"
        model = model.to(torch.bfloat16)
    elif precision == "int_8":
        print("using int_8 mpi4dl quant")
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
    else:
        return load_torchResNet(device, batch_size, image_size, num_classes, precision)

    # Using PyTorch
    # if num_classes == 1000:
    #     return load_torchResNet(device, precision)
    # return load_torchResNetCustomClass(device, num_classes, precision)


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
                break
            count += 1
            inputs, labels = data
            if precision == "fp_16":
                inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(
                    device, dtype=torch.float16
                )
            elif precision == "bfp_16":
                assert (
                    torch.cuda.is_bf16_supported() == True
                ), "Native System doen't support bf16"
                inputs, labels = inputs.to(device, dtype=torch.bfloat16), labels.to(
                    device, dtype=torch.bfloat16
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


def run_inference(image_size):
    print(
        f"\n\n***********image_size : {image_size} Precision : {precision}  batch_size : {batch_size} num_classes : {num_classes} APP : {APP} ***********"
    )
    # get_gpu_memory(rank=0)
    # max_memory_before = torch.cuda.max_memory_allocated(device="cuda")

    if WSI_IMAGES:
        assert APP == 2, "Use Pathology Dataset"
    # before = get_gpu_memory_used(0).split()[0]
    # before_torch = torch.cuda.memory_allocated(device="cuda")
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
    # after_torch = torch.cuda.memory_allocated(device="cuda")
    # after = get_gpu_memory_used(0).split()[0]
    print_model_size(model)
    # print(f"Mem req by model : {float(after) - float(before)} MiB")
    # print(
    #     f"Mem req by model using torch : {(float(after_torch) - float(before_torch)) / 1024**2} MB"
    # )

    if precision != "int_8":
        checkpoint = {
            "model_state_dict": model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # Add any other information you want to save
        }
        import os

        torch.save(checkpoint, "temp.pth")
        size = os.path.getsize("temp.pth")
        os.remove("temp.pth")

        print(f"checkpont file size : {size / 1024 ** 2}")
    dataloader = load_dataset(
        app=APP, batch_size=batch_size, image_size=image_size, times=times
    )
    torch.cuda.empty_cache()
    accuracy = evaluate(device, model, dataloader, WSI_IMAGES, image_size, precision)
    print(f"Accuracy with pretrained model : {accuracy * 100}")
    # max_memory_after = torch.cuda.max_memory_allocated(device="cuda")
    # reserve_mem = torch.cuda.memory_reserved(device="cuda")
    # print(f"Max Memory Before  Using PyTorch CUDA: {max_memory_before / (1024 ** 2):.2f} MB")
    # print(f"Max Memory After Using PyTorch CUDA: {max_memory_after / (1024 ** 2):.2f} MB")
    # print(f"Total Resever by caching : {reserve_mem/ (1024 ** 2):.2f}")


device = "cuda:0"
# APP = 2

## FOR WSI IMAGES
# APP = 2
# MPI4DL = True
# WSI_IMAGES = True

# FOR Fake Dataset
# APP = 3
# MPI4DL = True
# WSI_IMAGES = False  # set image-size = 64, num_classes = 10 for imagenette checkpoint
NVIDIA_SMI_MEM_CAPTURE = False  # Enable if you want to profile memory usage

# FOR ImageNette or ImageNet
APP = 1
MPI4DL = False
WSI_IMAGES = False

batch_size = args.batch_size
parts = 1
image_size = args.image_size
resnet_n = 12
num_classes = 10
precision = args.precision  # values [fp_32, fp_16, bf_16, int_8]

times = 1

######
max_mem_used_nvidia_smi = 0
if NVIDIA_SMI_MEM_CAPTURE:
    EXECUTING = True
    import threading

    def my_function():
        # Your specific function code goes here
        curr = get_gpu_memory_used(0).split()[0]
        global max_mem_used_nvidia_smi
        max_mem_used_nvidia_smi = max(max_mem_used_nvidia_smi, float(curr))

    def thread_function():
        while EXECUTING:
            my_function()

    my_thread = threading.Thread(target=thread_function)
    my_thread.start()
#####

# for pow in range(9, 20):
#     image_size = 2 ** pow
#     torch.cuda.empty_cache()
#     run_inference(image_size)
#     torch.cuda.empty_cache()

# torch.cuda.memory._record_memory_history(max_entries=100000)
# print_gpu_memory_every_5secs()
run_inference(image_size=image_size)

# torch.cuda.memory._dump_snapshot(f"/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/single_gpu_inference/mem_mpi4dl_snapshot_prec_{precision}_img_{image_size}_batch_{batch_size}.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)
if NVIDIA_SMI_MEM_CAPTURE:
    print(f"Max Mem Util using nvidia smi : {max_mem_used_nvidia_smi} MiB")
EXECUTING = False
