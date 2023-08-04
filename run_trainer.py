import os, sys
import subprocess
import urllib.request
import re
import toml
import torch
from time import time

main_config_file = "config.toml"
try:
    main_config = toml.load(main_config_file)
except Exception:
    print(f"Error on parsing main config file. Please check the format. : {main_config_file}")
    raise

os.environ["CUDA_HOME"] = main_config.get("general").get("cuda_home")

print("cuda version =", torch.version.cuda , "and available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("이 시스템에서는 CUDA를 사용할 수 없습니다. 재설치 후 다시 실행하세요.")
    exit(1)

# 미리 설치할 사항
# sed , cuda, cudnn, pytorch 2.x, aria2c, accelerate
# 각종 설정값. 추후 ui가 필요하면 아래 내용을 기준으로 제작.
# triton은 linux lib. 윈도우에서는 경고 무시!

# ### ▶️ Setup
old_model_url = None

if "dependencies_installed" not in globals():
    dependencies_installed = False
if "model_file" not in globals():
    model_file = None
if "custom_dataset" not in globals():
    custom_dataset = None
if "override_dataset_config_file" not in globals():
    override_dataset_config_file = None
if "override_config_file" not in globals():
    override_config_file = None

COLAB = True # False = low vram
COMMIT = "f037b09c2de13df549290b7c8d4d4a22ab165c36"
BETTER_EPOCH_NAMES = True
LOAD_TRUNCATED_IMAGES = True

# # getattr 함수를 사용하여 frozen 속성이 있는지 확인
if getattr(sys, 'frozen', False):
    # frozen 속성이 있으면 실행 파일이므로 sys.executable의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(sys.executable)
else:
    # frozen 속성이 없으면 스크립트가 실행 중이므로 현재 파일(__file__)의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(__file__)

data_path = os.path.join(current_directory, "Loras")
project_name = main_config.get("general").get("lora_name")
folder_structure = data_path + "/lora_training/datasets/" + project_name
optional_custom_training_model_url = main_config.get("general").get("model_url")
custom_model_is_based_on_sd2 = False

continue_from_lora = ""
if optional_custom_training_model_url:
    model_url = optional_custom_training_model_url
else:
    model_url = "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/animefull-final-pruned-fp16.safetensors"

# ### ▶️ Processing
resolution = main_config.get("general").get("resolution") #min:512, max:1024, step:256
flip_aug = True # This option will train your images both normally and flipped, for no extra cost, to learn more from them. Turn it on specially if you have less than 20 images.
caption_extension = ".txt" #param {type:"string"}
shuffle_tags = True # Shuffling improves learning for anime tags. An activation tag goes at the start of every text file and will not be shuffled.
shuffle_caption = shuffle_tags
activation_tags = "1" # dateset 설정 시 activation tags 설정한 개수와 동일하게..
keep_tokens = int(activation_tags)


# ### ▶️ Steps
num_repeats = 10 # Your images will repeat this number of times during training. I recommend that your images multiplied by their repeats is between 200 and 400.
# Choose how long you want to train for. A good starting point is around 10 epochs or around 2000 steps.
# One epoch is a number of steps equal to: your number of images multiplied by their repeats, divided by batch size.
preferred_unit = "Epochs" # ["Epochs", "Steps"]
how_many = 10
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
# Saving more epochs will let you compare your Lora's progress better.
save_every_n_epochs = 10
keep_only_last_n_epochs = 10
if not save_every_n_epochs:
  save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
  keep_only_last_n_epochs = max_train_epochs
# Increasing the batch size makes training faster, but may make learning worse. Recommended 2 or 3.
train_batch_size = 2 # min:1, max:8, step:1
  

# ### ▶️ Learning
# The learning rate is the most important for your results. If you want to train slower with lots of images, or if your dim and alpha are high, move the unet to 2e-4 or lower. <p>
# The text encoder helps your Lora learn concepts slightly better. It is recommended to make it half or a fifth of the unet. If you're training a style you can even set it to 0.
unet_lr = 5e-4
text_encoder_lr = 1e-4 # unet_lr/5
# The scheduler is the algorithm that guides the learning rate. If you're not sure, pick `constant` and ignore the number. I personally recommend `cosine_with_restarts` with 3 restarts.
lr_scheduler = "cosine_with_restarts" # ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
lr_scheduler_number = 3
lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0
# Steps spent "warming up" the learning rate during training for efficiency. I recommend leaving it at 5%.
lr_warmup_ratio = 0.1 # min:0.0, max:0.5, step:0.01
lr_warmup_steps = 0
# New feature that adjusts loss over time, makes learning much more efficient, and training can be done with about half as many epochs. Uses a value of 5.0 as recommended by [the paper](https://arxiv.org/abs/2303.09556).
min_snr_gamma = True
min_snr_gamma_value = 5.0 if min_snr_gamma else None


# ### ▶️ Structure
# LoRA is the classic type, while LoCon is good with styles. Lycoris require [this extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris) for webui to work like normal loras. More info [here](https://github.com/KohakuBlueleaf/Lycoris).
lora_type = "LoRA" #@param ["LoRA", "LoCon Lycoris", "LoHa Lycoris"]

# Below are some recommended values for the following settings:
# | type  | network_dim | network_alpha | conv_dim | conv_alpha |
# | :---: | :---:       | :---:         | :---:    | :---:      |
# | LoRA  | 32          | 16            |          |            |
# | LoCon | 16          | 8             | 8        | 1          |
# | LoHa  | 8           | 4             | 4        | 1          |

# More dim means larger Lora, it can hold more information but more isn't always better. A dim between 8-32 is recommended, and alpha equal to half the dim.
network_dim = 32  # min:1, max:128, step:1
network_alpha = 16 #min:1, max:128, step:1
# The following values don't affect LoRA. They work like dim/alpha but only for the additional learning layers of Lycoris.
conv_dim = 8 #@param {type:"slider", min:1, max:64, step:1}
conv_alpha = 1 #@param {type:"slider", min:1, max:64, step:1}
conv_compression = False #@param {type:"boolean"}

network_module = "lycoris.kohya" if "Lycoris" in lora_type else "networks.lora"
network_args = None if lora_type == "LoRA" else [
  f"conv_dim={conv_dim}",
  f"conv_alpha={conv_alpha}",
]
if "Lycoris" in lora_type:
  network_args.append(f"algo={'loha' if 'LoHa' in network_args else 'lora'}")
  network_args.append(f"disable_conv_cp={str(not conv_compression)}")


# ### ▶️ Experimental
# Save additional data equaling ~1 GB allowing you to resume training later.
save_state = False #param {type:"boolean"}
# Resume training if a save state is found.
resume = False #param {type:"boolean"}


# ### ▶️ Done



# 👩‍💻 Cool code goes here

root_dir = data_path
deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "kohya-trainer")
model_dir = os.path.join(root_dir, "model")

if "/Loras" in folder_structure:
    main_dir      = root_dir
    log_folder    = os.path.join(main_dir, "_logs")
    config_folder = os.path.join(main_dir, project_name)
    images_folder = os.path.join(main_dir, project_name, "dataset")
    output_folder = os.path.join(main_dir, project_name, "output")
else:
    main_dir      = root_dir
    images_folder = os.path.join(main_dir, "datasets", project_name)
    output_folder = os.path.join(main_dir, "output", project_name)
    config_folder = os.path.join(main_dir, "config", project_name)
    log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
accelerate_config_file = os.path.join(repo_dir, "accelerate_config/config.yaml")


def clone_repo():
    os.chdir(root_dir)
    print("move to ", root_dir)
    subprocess.run(['git', 'clone', 'https://github.com/kohya-ss/sd-scripts', repo_dir])
    os.chdir(repo_dir)
    print("move to ", repo_dir)
    subprocess.run(['git', 'reset', '--hard', COMMIT])
    urllib.request.urlretrieve("https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/main/requirements.txt", "requirements.txt") # urllib.request.urlretrieve은 파일을 다운로드 하는데 사용

    # .venv 폴더가 있는지 확인
    venv_folder = os.path.join(current_directory, ".venv")
    if os.path.exists(venv_folder) and os.path.isdir(venv_folder):
        target_folder = "..\\..\\.venv\\Lib\\site-packages\\bitsandbytes\\"
    else:
        target_folder = sys.prefix + "\\Lib\\site-packages\\bitsandbytes\\"

    subprocess.run(["copy", ".\\bitsandbytes_windows\\*.dll", target_folder], shell=True)
    subprocess.run(["copy", ".\\bitsandbytes_windows\\cextension.py", f"{target_folder}cextension.py"], shell=True)
    subprocess.run(["copy", ".\\bitsandbytes_windows\\main.py", f"{target_folder}cuda_setup\\main.py"], shell=True)

def install_dependencies():
    clone_repo()
    subprocess.run(['pip', 'install', '--upgrade', '-r', 'requirements.txt'], capture_output=True)


    if COLAB:
        subprocess.run(["sed","-i", "s/cpu/cuda/", "library/model_util.py"]) # low ram

    if LOAD_TRUNCATED_IMAGES:
        subprocess.run(["sed","-i","s/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g", "library/train_util.py"]) # fix truncated jpegs error

    if BETTER_EPOCH_NAMES:
        subprocess.run(["sed", "-i", "s/{:06d}/{:02d}/g", "library/train_util.py"]) # make epoch names shorter
        subprocess.run(["sed", "-i", "s/model_name + '.'/model_name + '-{:02d}.'.format(num_train_epochs)/g", "train_network.py"]) # name of the last epoch will match the rest

    from accelerate.utils import write_basic_config
    if not os.path.exists(accelerate_config_file):
        write_basic_config(save_location=accelerate_config_file)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"  
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

def validate_dataset():
    global lr_warmup_steps, lr_warmup_ratio, caption_extension
    supported_types = (".png", ".jpg", ".jpeg")

    print("\n💿 Checking dataset...")
    if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
        print("💥 Error: Please choose a valid project name.")
        return

    if custom_dataset:
        try:
            datconf = toml.loads(custom_dataset)
            datasets = {d["image_dir"]: d["num_repeats"] for d in datconf["datasets"][0]["subsets"]}
        except:
            print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
            return
        folders = datasets.keys()
        files = [f for folder in folders for f in os.listdir(folder)]
        images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets[folder]) for folder in folders}
    else:
        folders = [images_folder]
        files = os.listdir(images_folder)
        images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

    for folder in folders:
        if not os.path.exists(folder):
            print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
            return
        for folder, (img, rep) in images_repeats.items():
            if not img:
                print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
                return
            for f in files:
                if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
                    print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
                    return

    if not [txt for txt in files if txt.lower().endswith(".txt")]:
        caption_extension = ""

    if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
        print(f"💥 Error: Invalid path to existing Lora. Example: lora_training/example.safetensors")
        return

    pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
    steps_per_epoch = pre_steps_per_epoch/train_batch_size
    total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
    estimated_epochs = int(total_steps/steps_per_epoch)
    lr_warmup_steps = int(total_steps*lr_warmup_ratio)

    for folder, (img, rep) in images_repeats.items():
        print("📁"+folder.replace("/content/drive/", ""))
        print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
    print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
    if max_train_epochs:
        print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
    else:
        print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

    if total_steps > 10000:
        print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...") 
        return
    return True

def create_config():
    global dataset_config_file, config_file, model_file

    if resume:
        resume_points = [f.path for f in os.scandir(output_folder) if f.is_dir()]
        resume_points.sort()
        last_resume_point = resume_points[-1] if resume_points else None
    else:
        last_resume_point = None

    if override_config_file:
        config_file = override_config_file
        print(f"\n⭕ Using custom config file {config_file}")
    else:
        config_dict = {
            "additional_network_arguments": {
            "unet_lr": unet_lr,
            "text_encoder_lr": text_encoder_lr,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_module": network_module,
            "network_args": network_args,
            "network_train_unet_only": True if text_encoder_lr == 0 else None,
            "network_weights": continue_from_lora if continue_from_lora else None
        },
        "optimizer_arguments": {
            "learning_rate": unet_lr,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
            "lr_warmup_steps": lr_warmup_steps,
            "optimizer_type": "AdamW8bit",
        },
        "training_arguments": {
            "max_train_steps": max_train_steps,
            "max_train_epochs": max_train_epochs,
            "save_every_n_epochs": save_every_n_epochs,
            "save_last_n_epochs": keep_only_last_n_epochs,
            "train_batch_size": train_batch_size,
            "noise_offset": None,
            "clip_skip": 2,
            "min_snr_gamma": min_snr_gamma_value,
            "seed": 42,
            "max_token_length": 225,
            "xformers": True,
            "lowram": COLAB,
            "max_data_loader_n_workers": 8,
            "persistent_data_loader_workers": True,
            "save_precision": "fp16",
            "mixed_precision": "fp16",
            "output_dir": output_folder,
            "logging_dir": log_folder,
            "output_name": project_name,
            "log_prefix": project_name,
            "save_state": save_state,
            "save_last_n_epochs_state": 1 if save_state else None,
            "resume": last_resume_point
        },
        "model_arguments": {
            "pretrained_model_name_or_path": model_file,
            "v2": custom_model_is_based_on_sd2,
            "v_parameterization": True if custom_model_is_based_on_sd2 else None,
        },
        "saving_arguments": {
            "save_model_as": "safetensors",
        },
        "dreambooth_arguments": {
            "prior_loss_weight": 1.0,
        },
        "dataset_arguments": {
            "cache_latents": True,
        },
    }

    for key in config_dict:
        if isinstance(config_dict[key], dict):
            config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
        f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

    if override_dataset_config_file:
        dataset_config_file = override_dataset_config_file
        print(f"⭕ Using custom dataset config file {dataset_config_file}")
    else:
        dataset_config_dict = {
            "general": {
                "resolution": resolution,
                "shuffle_caption": shuffle_caption,
                "keep_tokens": keep_tokens,
                "flip_aug": flip_aug,
                "caption_extension": caption_extension,
                "enable_bucket": True,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
                "min_bucket_reso": 320 if resolution > 640 else 256,
                "max_bucket_reso": 1280 if resolution > 640 else 1024,
            },
            "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
            {
                "subsets": [
                    {
                        "num_repeats": num_repeats,
                        "image_dir": images_folder,
                        "class_tokens": None if caption_extension else project_name
                    }
                ]
            }
        ]
    }

    for key in dataset_config_dict:
        if isinstance(dataset_config_dict[key], dict):
            dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
        f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")

def download_model():
    global old_model_url, model_url, model_file
    real_model_url = model_url.strip()
  
    if real_model_url.lower().endswith((".ckpt", ".safetensors")):
        model_file = f"model{real_model_url[real_model_url.rfind('/'):]}"
    else:
        model_file = "model\downloaded_model.safetensors"
        if os.path.exists(model_file):
            subprocess.run(["del",f"{repo_dir}\{model_file}"], shell=True)
            # subprocess.run(["rm","{model_file}"])

    if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", model_url):
        real_model_url = real_model_url.replace("blob", "resolve")
    elif m := re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)", model_url):
        real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"


    subprocess.run(["aria2c", real_model_url, "--console-log-level=warn", "-c", "-s", "16", "-x", "16", "-k", "10M", "-d", "", "-o", model_file], bufsize=0)


    if model_file.lower().endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors
        try:
            test = load_safetensors(model_file)
            del test
        except Exception as e:
            #if "HeaderTooLarge" in str(e):
            new_model_file = os.path.splitext(model_file)[0]+".ckpt"
            subprocess.run("mv", model_file, new_model_file)
            model_file = new_model_file
            print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

    if model_file.lower().endswith(".ckpt"):
        from torch import load as load_ckpt
        try:
            test = load_ckpt(model_file)
            del test
        except Exception as e:
            return False
        
    return True

def main():
    global dependencies_installed

    for dir in (main_dir, deps_dir, repo_dir, log_folder, images_folder, output_folder, config_folder, model_dir):
        os.makedirs(dir, exist_ok=True)
        print("make dir =", dir)

    if not dependencies_installed:
        print("\n🏭 Installing dependencies...\n")
        t0 = time()
        install_dependencies()
        t1 = time()
        dependencies_installed = True
        print(f"\n✅ Installation finished in {int(t1-t0)} seconds.")
    else:
        print("\n✅ Dependencies already installed.")

    if old_model_url != model_url or not model_file or not os.path.exists(model_file):
        print("\n🔄 Downloading model...")
        if not download_model():
            print("\n💥 Error: The model you selected is invalid or corrupted, or couldn't be downloaded. You can use a civitai or huggingface link, or any direct download link.")
            return
    else:
        print("\n🔄 Model already downloaded.\n")

    if not validate_dataset():
        return
    
    create_config()
  
    print("\n⭐ Starting trainer...\n")
    os.chdir(repo_dir)
  
    subprocess.run(["accelerate", "launch", f"--config_file={accelerate_config_file}", "--num_cpu_threads_per_process=2", "train_network.py", f"--dataset_config={dataset_config_file}", f"--config_file={config_file}"])

    print("### ✅ Done! Go download your Lora(s)!!")

print("accelerate_config_file = ", accelerate_config_file)
main()