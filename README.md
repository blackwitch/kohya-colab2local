# kohya-colab2local
This is a project that modifies "kohya-colab" so that it can be run locally on Windows OS.

See the link below for the original source.

[Guide document](https://civitai.com/models/22530)

[Github link](https://github.com/hollowstrawberry/kohya-colab)

For colleagues who are less comfortable with Python or Colab, we've modified the Kohya-colab code to make it easy to run on a local Windows OS PC. Currently only tested in the SD version.

It is almost similar to Kohya-colab's code and only some code has been modified to run on windows OS by itself. Here we only describe the modified code and a guide to run it locally, please refer to the above link for more details.

## Pre-installed programs
 - python 3.10.x
 - sed - https://gnuwin32.sourceforge.net/packages/sed.htm
 - cuda - https://developer.nvidia.com/cuda-toolkit
 - cudnn - https://developer.nvidia.com/cudnn
 - aria2c - https://aria2.github.io/
 - cloneing  a source : git clone https://github.com/blackwitch/kohya-colab2local.git
 - install packages   : pip install -r requirements.txt

## How to do
- setting config file : 
  - resolution - Image size to be trained
  - cuda_home - path to where CUDA is installed 
  - lora_name - lora file name
  - model_url - Setting the model address to use for training -default value is "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/animefull-final-pruned-fp16.safetensors". set a civitai model like "https://civitai.com/models/7240"
  - activation_tag - prompt texts for your lora. 
  - remove_tags - prompt texts you want to delete from the auto generated text.
- setting dateset    : 
  - python run_dataset.py (Create a basic folder structure and a config file before adding any images that are still to be learned.)
  - put images you want to train in "Loras\datasets\your_lora_name"
  - python run_dataset.py
  - You can modify the prompts by opening the txt file generated in the datasets folder.
- training :
  - python run_training.py
  - (* ignore an warning and error about triton lib. it's a library for linux.)


## Explanation of the modified code

For ease of execution, the parameters for training are all set to default values. Feel free to modify the values in the code if needed.

Below we've summarized the main changes we've made to simplify running in a Windows environment. We've also made it possible to set some settings in config.toml that often need to be modified at runtime. See "Config settings" described above.

It doesn't use IPython, IPython.display, which is needed for colab, so it doesn't import it. Also, the commands executed with !!! have been replaced with subprocess.run for Windows, which I won't duplicate in the description below.

### [run_dateset.py]

 (source) https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Dataset_Maker.ipynb 

- In "STEP 1: Setup", we've added some code to allow you to use your local drive instead of Google Drive.
```
if getattr(sys, 'frozen', False):
    # frozen 속성이 있으면 실행 파일이므로 sys.executable의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(sys.executable)
else:
    # frozen 속성이 없으면 스크립트가 실행 중이므로 현재 파일(__file__)의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(__file__)

data_path = os.path.join(current_directory, "Loras")
```

- "STEP 2: Scrape images from Gelbooru" has been annotated because it is not used. 
- "STEP 3: Curate your images" is also commented out as it uses already curated images for training.
- In "STEP 4: Tag your images", we removed unnecessary path changes and installation code for the local execution environment, and no other major modifications.
- "STEP 5: Curate your tags" is unchanged.

### [run_trainer.py]

 (source) https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Lora_Trainer.ipynb

- Added code to "STEP 1: Setup" to prevent it from running if cuda is not available.
```
os.environ["CUDA_HOME"] = main_config.get("general").get("cuda_home")

print("cuda version =", torch.version.cuda , "and available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("이 시스템에서는 CUDA를 사용할 수 없습니다. 재설치 후 다시 실행하세요.")
    exit(1)
```

I kept the model_url setting simple because this is a tool that will be used with a fixed value. If you use optional_custom_training_model_url, it will be downloaded each time you run it, so if you want to use it repeatedly, leave the optional_custom_training_model_url value blank and set the model_url value yourself in your code.
```
optional_custom_training_model_url = main_config.get("general").get("model_url")
.
.
if optional_custom_training_model_url:
    model_url = optional_custom_training_model_url
else:
    model_url = "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/animefull-final-pruned-fp16.safetensors"
```

- "STEP Processing, Steps, Learning, Structure" does not change the code. If you need to change the option values, please modify the code here yourself. If possible, it's best for users to find the optimal values and not modify them after setting them.
- The path settings, clone_repo, etc. at the bottom of the code have been modified to use local paths. The part to set dll and source code for using bitsandbites in Windows environment is modified as below.
```
# .venv 폴더가 있는지 확인
venv_folder = os.path.join(current_directory, ".venv")
if os.path.exists(venv_folder) and os.path.isdir(venv_folder):
	target_folder = "..\\..\\.venv\\Lib\\site-packages\\bitsandbytes\\"
else:
	target_folder = sys.prefix + "\\Lib\\site-packages\\bitsandbytes\\"

subprocess.run(["copy", ".\\bitsandbytes_windows\\*.dll", target_folder], shell=True)
subprocess.run(["copy", ".\\bitsandbytes_windows\\cextension.py", f"{target_folder}cextension.py"], shell=True)
subprocess.run(["copy", ".\\bitsandbytes_windows\\main.py", f"{target_folder}cuda_setup\\main.py"], shell=True)
```


That's it for the code modifications. There are very few modifications to the core code and most of the modifications are for the WINDOWS OS environment, so if you have seen the existing code, you shouldn't have much trouble understanding it. 

If you have any difficulty using it, please open an issue :)
