import os, sys
import subprocess
import toml

main_config_file = "config.toml"
try:
    main_config = toml.load(main_config_file)
except Exception:
    print(f"Error on parsing main config file. Please check the format. : {main_config_file}")
    raise

COLAB = True # False = low vram

# getattr 함수를 사용하여 frozen 속성이 있는지 확인
if getattr(sys, 'frozen', False):
    # frozen 속성이 있으면 실행 파일이므로 sys.executable의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(sys.executable)
else:
    # frozen 속성이 없으면 스크립트가 실행 중이므로 현재 파일(__file__)의 디렉토리 경로를 가져옴
    current_directory = os.path.dirname(__file__)

data_path = os.path.join(current_directory, "Loras")
project_name = main_config.get("general").get("lora_name")
folder_structure = data_path + "/lora_training/datasets/" + project_name
optional_custom_training_model_url = ""
custom_model_is_based_on_sd2 = False

project_base = project_name if "/" not in project_name else project_name[:project_name.rfind("/")]
project_subfolder = project_name if "/" not in project_name else project_name[project_name.rfind("/")+1:]

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

for dir in [main_dir, deps_dir, images_folder, config_folder]:
    os.makedirs(dir, exist_ok=True)


print(f"✅ Project {project_name} is ready!", images_folder)


###########################################
# 중복 이미지 없애기.. 꼭 필요한 과정은 아님.
#@markdown ### 3️⃣ Curate your images
#@markdown We will find duplicate images with the FiftyOne AI, and mark them with `delete`. <p>
#@markdown Then, an interactive area will appear below this cell that lets you visualize all your images and manually mark with `delete` to the ones you don't like. <p>
#@markdown If the interactive area appears blank for over a minute, try enabling cookies and removing tracking protection for the Google Colab website, as they may break it.
#@markdown Regardless, you can save your changes by sending Enter in the input box above the interactive area.<p>
#@markdown This is how similar 2 images must be to be marked for deletion. I recommend 0.97 to 0.99:
'''
similarity_threshold = 0.985 #@param {type:"number"}

os.chdir(root_dir)
model_name = "clip-vit-base32-torch"
supported_types = (".png", ".jpg", ".jpeg")
print("images_folder = ", images_folder)
img_count = len(os.listdir(images_folder))
batch_size = min(250, img_count)


import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from sklearn.metrics.pairwise import cosine_similarity

non_images = [f for f in os.listdir(images_folder) if not f.lower().endswith(supported_types)]
if non_images:
  print(f"💥 Error: Found non-image file {non_images[0]} - This program doesn't allow it. Sorry! Use the Extras at the bottom to clean the folder.")
elif img_count == 0:
  print(f"💥 Error: No images found in {images_folder}")
else:
  print("\n💿 Analyzing dataset...\n")
  dataset = fo.Dataset.from_dir(images_folder, dataset_type=fo.types.ImageDirectory)
  model = foz.load_zoo_model(model_name)
  embeddings = dataset.compute_embeddings(model, batch_size=batch_size)

  batch_embeddings = np.array_split(embeddings, batch_size)
  similarity_matrices = []
  max_size_x = max(array.shape[0] for array in batch_embeddings)
  max_size_y = max(array.shape[1] for array in batch_embeddings)

  for i, batch_embedding in enumerate(batch_embeddings):
    similarity = cosine_similarity(batch_embedding)
    #Pad 0 for np.concatenate
    padded_array = np.zeros((max_size_x, max_size_y))
    padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
    similarity_matrices.append(padded_array)

  similarity_matrix = np.concatenate(similarity_matrices, axis=0)
  similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]

  similarity_matrix = cosine_similarity(embeddings)
  similarity_matrix -= np.identity(len(similarity_matrix))

  dataset.match(F("max_similarity") > similarity_threshold)
  dataset.tags = ["delete", "has_duplicates"]

  id_map = [s.id for s in dataset.select_fields(["id"])]
  samples_to_remove = set()
  samples_to_keep = set()

  for idx, sample in enumerate(dataset):
    if sample.id not in samples_to_remove:
      # Keep the first instance of two duplicates
      samples_to_keep.add(sample.id)

      dup_idxs = np.where(similarity_matrix[idx] > similarity_threshold)[0]
      for dup in dup_idxs:
          # We kept the first instance so remove all other duplicates
          samples_to_remove.add(id_map[dup])

      if len(dup_idxs) > 0:
          sample.tags.append("has_duplicates")
          sample.save()
    else:
      sample.tags.append("delete")
      sample.save()

  
  sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
  for group in sidebar_groups[1:]:
    group.expanded = False
  dataset.app_config.sidebar_groups = sidebar_groups
  dataset.save()
  session = fo.launch_app(dataset)

  print("❗ Wait a minute for the session to load. If it doesn't, read above.")
  print("❗ When it's ready, you'll see a grid of your images.")
  print("❗ On the left side enable \"sample tags\" to visualize the images marked for deletion.")
  print("❗ You can mark your own images with the \"delete\" label by selecting them and pressing the tag icon at the top.")
  input("⭕ When you're done, enter something here to save your changes: ")

  print("💾 Saving...")

  kys = [s for s in dataset if "delete" in s.tags]
  dataset.remove_samples(kys)
  previous_folder = images_folder[:images_folder.rfind("/")]
  dataset.export(export_dir=os.path.join(images_folder, project_subfolder), dataset_type=fo.types.ImageDirectory)

  temp_suffix = "_temp"
  subprocess.run(["mv", images_folder, images_folder+temp_suffix], shell=True)
  subprocess.run(["mv", images_folder+temp_suffix+project_subfolder, images_folder], shell=True)
  subprocess.run(["rm", "-r", images_folder+temp_suffix], shell=True)
  
  session.refresh()
  fo.close_app()
  
  print(f"\n✅ Removed {len(kys)} images from dataset. You now have {len(os.listdir(images_folder))} images.")
'''
########################################################


# Tag your images - 프롬프트 만들기. 만든 후 수동으로 수정해도 됨.
#@markdown We will be using AI to automatically tag your images, specifically [Waifu Diffusion](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2) in the case of anime and [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) in the case of photos.
#@markdown Giving tags/captions to your images allows for much better training. This process should take a couple minutes. <p>
method = "Anime tags" #@param ["Anime tags", "Photo captions"]
#@markdown **Anime:** The threshold is the minimum level of confidence the tagger must have in order to include a tag. Lower threshold = More tags. Recommended 0.35 to 0.5
tag_threshold = 0.35 #@param {type:"slider", min:0.0, max:1.0, step:0.01}
blacklist_tags = "bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber, official alternate costume, official alternate hairstyle, official alternate hair length, alternate costume, alternate hairstyle, alternate hair length, alternate hair color" #@param {type:"string"}
#@markdown **Photos:** The minimum and maximum length of tokens/words in each caption.
caption_min = 10 #@param {type:"number"}
caption_max = 75 #@param {type:"number"}

os.chdir(root_dir)
kohya = "./kohya-trainer"
if not os.path.exists(kohya):
    subprocess.run(['git', 'clone', 'https://github.com/kohya-ss/sd-scripts', repo_dir])
    os.chdir(kohya)
    subprocess.run(['git', 'reset', '--hard', '5050971ac687dca70ba0486a583d283e8ae324e2'])
    os.chdir(root_dir)

if "tags" in method:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  subprocess.run(["python", f'{kohya}/finetune/tag_images_by_wd14_tagger.py', 
                  images_folder, "--repo_id=SmilingWolf/wd-v1-4-swinv2-tagger-v2", 
                  f"--model_dir={root_dir}", f"--thresh={tag_threshold}", 
                  "--batch_size=8", "--caption_extension=.txt", "--force_download"], shell=True)

  print("removing underscores and blacklist...")
  blacklisted_tags = [t.strip() for t in blacklist_tags.split(",")]
  from collections import Counter
  top_tags = Counter()
  for txt in [f for f in os.listdir(images_folder) if f.lower().endswith(".txt")]:
      with open(os.path.join(images_folder, txt), 'r') as f:
          tags = [t.strip() for t in f.read().split(",")]
          tags = [t.replace("_", " ") if len(t) > 3 else t for t in tags]
          tags = [t for t in tags if t not in blacklisted_tags]
      top_tags.update(tags)
      with open(os.path.join(images_folder, txt), 'w') as f:
          f.write(", ".join(tags))

  print(f"📊 Tagging complete. Here are the top 50 tags in your dataset:")
  print("\n".join(f"{k} ({v})" for k, v in top_tags.most_common(50)))
else: # Photos
  subprocess.run(["python", f'{kohya}/finetune/make_captions.py', 
                  images_folder, "--beam_search", 
                  "--max_data_loader_n_workers=2", f"--min_length={caption_min}", f"--max_length={caption_max}",
                  "--batch_size=8", "--caption_extension=.txt", "--force_download"], shell=True)  

#@markdown ### 5️⃣ Curate your tags
#@markdown Modify your dataset's tags. You can run this cell multiple times with different parameters. <p>

#@markdown Put an activation tag at the start of every text file. This is useful to make learning better and activate your Lora easier. Set `keep_tokens` to 1 when training.<p>
#@markdown Common tags that are removed such as hair color, etc. will be "absorbed" by your activation tag.
global_activation_tag = main_config.get("general").get("activation_tag")
remove_tags = main_config.get("general").get("remove_tags")

#@markdown In this advanced section, you can search text files containing matching tags, and replace them with less/more/different tags. If you select the checkbox below, any extra tags will be put at the start of the file, letting you assign different activation tags to different parts of your dataset. Still, you may want a more advanced tool for this.
search_tags = "" #@param {type:"string"}
replace_with = "" #@param {type:"string"}
search_mode = "OR" #@param ["OR", "AND"]
new_becomes_activation_tag = False #@param {type:"boolean"}
#@markdown These may be useful sometimes. Will remove existing activation tags, be careful.
sort_alphabetically = False #@param {type:"boolean"}
remove_duplicates = False #@param {type:"boolean"}

def split_tags(tagstr):
  return [s.strip() for s in tagstr.split(",") if s.strip()]

activation_tag_list = split_tags(global_activation_tag)
remove_tags_list = split_tags(remove_tags)
search_tags_list = split_tags(search_tags)
replace_with_list = split_tags(replace_with)
replace_new_list = [t for t in replace_with_list if t not in search_tags_list]

replace_with_list = [t for t in replace_with_list if t not in replace_new_list]
replace_new_list.reverse()
activation_tag_list.reverse()

remove_count = 0
replace_count = 0

for txt in [f for f in os.listdir(images_folder) if f.lower().endswith(".txt")]:

  with open(os.path.join(images_folder, txt), 'r') as f:
    tags = [s.strip() for s in f.read().split(",")]

  if remove_duplicates:
    tags = list(set(tags))
  if sort_alphabetically:
    tags.sort()

  for rem in remove_tags_list:
    if rem in tags:
      remove_count += 1
      tags.remove(rem)

  if "AND" in search_mode and all(r in tags for r in search_tags_list) \
      or "OR" in search_mode and any(r in tags for r in search_tags_list):
    replace_count += 1
    for rem in search_tags_list:
      if rem in tags:
        tags.remove(rem)
    for add in replace_with_list:
      if add not in tags:
        tags.append(add)
    for new in replace_new_list:
      if new_becomes_activation_tag:
        if new in tags:
          tags.remove(new)
        tags.insert(0, new)
      else:
        if new not in tags:
          tags.append(new)

  for act in activation_tag_list:
    if act in tags:
      tags.remove(act)
    tags.insert(0, act)

  with open(os.path.join(images_folder, txt), 'w') as f:
    f.write(", ".join(tags))

if global_activation_tag:
  print(f"\n📎 Applied new activation tag(s): {', '.join(activation_tag_list)}")
if remove_tags:
  print(f"\n🚮 Removed {remove_count} tags.")
if search_tags:
  print(f"\n💫 Replaced in {replace_count} files.")
print("\n✅ Done! Check your updated tags in the Extras below.")