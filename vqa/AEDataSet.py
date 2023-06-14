import json
import os
import random

from IPython.display import Image, display
import PIL.Image
import PIL.Image as Image
import io
import torch
import numpy as np
from matplotlib import pyplot as plt

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


URL = "./testfiles/COCO_val2014_000000262148.jpg"

# 这是一个存储检测对象的地址，用于将模型预测的目标对象 ID 转换为对应的类别名称。
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
# 提供属性的 ID 到属性名称的映射（id2attr），用于将模型预测的属性 ID 转换为对应的属性名称。
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

objids = utils.get_data(OBJ_URL)
attrids = utils.get_data(ATTR_URL)
vqa_answers = utils.get_data(VQA_URL)
# %%
# load models and model components
# unc-nlp/frcnn-vg-finetuned 是一个模型的名称，它是基于自然语言处理 (NLP) 和计算机视觉 (CV) 技术的模型。
# 具体来说，它是一个基于 Faster R-CNN (Region-based Convolutional Neural Networks) 和 Visual Genome 数据集进行了微调的模型。
# Faster R-CNN 是一种常用的目标检测算法，可以在图像中准确地识别和定位不同的物体。
# 它基于卷积神经网络 (Convolutional Neural Network, CNN) 的架构，通过在图像中提取特征并进行区域推荐 (Region Proposal) 来实现目标检测。
# Visual Genome 数据集是一个大规模的图像描述和场景理解数据集，其中包含了丰富的图像和相应的语言描述信息。
# 通过在 Visual Genome 数据集上进行微调，unc-nlp/frcnn-vg-finetuned 模型能够更好地理解和处理图像与自然语言之间的关系。
# 总的来说，unc-nlp/frcnn-vg-finetuned 是一个经过微调的模型，结合了目标检测和图像理解技术，能够在图像中识别和理解物体，并与自然语言进行交互和处理。

# 具体而言，该函数的作用是：
#
# 下载指定预训练模型的配置文件。
# 根据配置文件创建一个模型配置对象。
# 返回模型配置对象，以便后续用于构建模型或加载预训练模型的参数。
# 这样，你可以使用这个模型配置对象来创建一个与预训练模型相同配置的新模型，并使用预训练的参数进行初始化，或者用它来加载预训练模型的参数进行微调或推理
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
print("frcnn_cfg:", frcnn_cfg)
# 具体而言，这个函数的作用是：
#
# 从 Hugging Face 模型库中下载指定预训练模型的权重文件。
# 根据预训练模型的配置文件创建一个模型配置对象。
# 使用预训练模型的权重和配置，创建一个通用的 Faster R-CNN 模型对象。
# 这个函数还接受一个可选的 config 参数，用于提供自定义的模型配置对象。通过将现有的模型配置对象 frcnn_cfg 传递给 config 参数，
# 可以使用自定义的配置来创建 Faster R-CNN 模型。这允许你根据需要修改模型的参数和设置。
# 最终，GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
# 将返回一个初始化并加载了预训练权重的 Faster R-CNN 模型。你可以使用这个模型进行目标检测任务，如在图像中检测和定位特定对象。
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
print("frcnn:", frcnn)
# Preprocess(frcnn_cfg) 的作用是：
#
# 基于提供的模型配置对象 frcnn_cfg，创建一个图像预处理对象。
# 图像预处理对象包含了根据模型配置定义的图像预处理操作。
image_preprocess = Preprocess(frcnn_cfg)

# 具体而言，BertTokenizerFast.from_pretrained("bert-base-uncased") 的作用是：
#
# 从 Hugging Face 模型库中下载 "bert-base-uncased" 预训练模型的词汇表（vocabulary）和配置文件。
# 根据预训练模型的词汇表和配置创建一个 BERT 分词器对象。
# 这个分词器对象可以用于对输入文本进行分词，并生成模型可接受的输入表示，如词嵌入或子词嵌入。
#
# 使用 bert_tokenizer.tokenize(text) 可以将文本 text 分词为标记序列，
# 使用 bert_tokenizer.encode(text) 可以将文本编码为模型所需的整数序列。
# 此外，还可以使用其他方法如 bert_tokenizer.decode() 将模型输出的整数序列解码为可读的文本。
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa") 的作用是：
#
# 从 Hugging Face 模型库中下载 "uclanlp/visualbert-vqa" 预训练模型的权重和配置文件。
# 根据预训练模型的配置创建一个 VisualBERT 用于 VQA 任务的模型对象。
# 这个函数返回的模型对象，即 visualbert_vqa，是一个已经初始化并加载了预训练权重的 VisualBERT 模型，用于解决视觉问答任务。
#
# 通过这个模型对象，你可以将图像和问题输入模型，获取模型对于给定图像和问题的答案预测。
visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")


# for visualizing output
def showarray(a, fmt="jpeg"):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)

    # 在PyCharm中，你可以使用matplotlib库来显示从BytesIO对象中读取的图像数据。
    image = np.array(Image.open(io.BytesIO(f.getvalue())))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # 使用Jupyter Notebook的 display() 函数显示从BytesIO对象中读取的图像数据。
    # display(Image(data=f.getvalue()))
    # %%
    # load object, attribute, and answer labels


def evaluation(image, question, answers):
    print("evaluation image:", image)
    flie_path = "E:/MasterOfAdelaideUniversity/7100/Data/VQA2/val2014/COCO_val2014_" + str(image['img_id']).zfill(12) + ".jpg"
    print("flie_path:", flie_path)
    frcnn_visualizer = SingleImageViz(flie_path, id2obj=objids, id2attr=attrids)

    # 图像处理后是一个四维张量，模型分析输入的是四维张量

    output_dict = frcnn(
        image['img'],
        image['sizes'],
        scales_yx=image['scales_yx'],
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    # add boxes and labels to the image
    print("output_dict:", output_dict
          )
    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
    )

    # 显示图片分析结果
    showarray(frcnn_visualizer._get_buffer())

    # test_questions_for_url2 = [
    #     "Where is he looking?",
    #     "What are the people in the background doing?",
    #     "What is he on top of?",
    # ]

    # test_questions_for_url2 = [item['question'] for item in question]
    # test_answers_for_url2 = [item['answers'] for item in answers]

    # print("test_questions_for_url2:", test_questions_for_url2)
    # Very important that the boxes are normalized
    # normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    # %%

    countright = 0
    # for test_question in test_questions_for_url2:
    #     test_question = [test_question]

    inputs = bert_tokenizer(
        # test_question,
        question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    print("inputs:", inputs)
    output_vqa = visualbert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_embeds=features,
        visual_attention_mask=torch.ones(features.shape[:-1]),
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    print("output_vqa:", output_vqa)
    # get prediction
    pred_vqa = output_vqa["logits"].argmax(-1)
    print("pred_vqa:", pred_vqa)
    # print("test_question:", test_question)
    # print("item:", item)
    # print("test_answers_for_url2:", len(test_answers_for_url2))
    # find_r_answer = [item['answers'] for item in item if item['question'] == test_question[0]]
    # print("Question:", test_question)

    # print("find_r_answer:", [item['answer'] for item in sum(find_r_answer, [])])
    print("prediction from VisualBert VQA:", vqa_answers[pred_vqa])
    print('image_id:', image['img_id'], '\nquestion:', question, '\nanswers:', answers)
    # if vqa_answers[pred_vqa] in [item['answer'] for item in sum(find_r_answer, [])]:
    if vqa_answers[pred_vqa] in [answers]:
        countright = 1

    # print("countright/len(test_questions_for_url2):", countright / len(test_questions_for_url2))
    return countright


class CustomDataset(Dataset):
    def __init__(self, img_path_dict, questions_data, answers_data, transform=None):
        # print('questions_data:' , len(questions_data))
        # print('answers_data:', len(answers_data))
        # print('img_path_dict:', img_path_dict)
        self.img_path_dict = img_path_dict
        self.questions_data = questions_data
        self.answers_data = {item['question_id']: item for item in answers_data}  # 将answers_data转为字典
        self.transform = transform

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        # 获取问题数据和对应的答案
        question_data = self.questions_data[idx]
        print('question_data:', question_data)
        question_id = question_data["question_id"]
        question = question_data["question"]

        img_id = str(question_data["image_id"])
        img_path = self.img_path_dict.get(img_id, None)
        if img_path is None:
            raise ValueError(f"Image id {img_id} not found in path dict")

        # # 加载图像
        # img = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     img = self.transform(img)

        # 通过question_id找到对应的答案
        answer_data = self.answers_data.get(question_id, {})
        answers = answer_data.get("answers", [])
        print('answer_data:', answer_data)

        images, sizes, scales_yx = image_preprocess(img_path)
        image_info = {
            'img_id': img_id,
            'img': images,
            'sizes': sizes,
            'scales_yx': scales_yx,
        }

        return {'image_info': image_info, 'question': question, 'answers': answers}


def collate_fn(data):
    # `data`是一个列表，每个元素是一个字典，包含'image', 'question'和'answers'
    return data


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 假设你已经有了图像路径列表
img_dir = 'E:/MasterOfAdelaideUniversity/7100/Data/VQA2/val2014'  # 图像文件夹的路径

# 创建一个字典，键为图像ID，值为图像路径
img_path_dict = {}

for filename in os.listdir(img_dir):
    if filename.endswith('.jpg'):  # 如果文件是一个jpg图像
        img_id = filename.split('_')[-1].split('.')[0].lstrip('0')  # 提取图像ID并删除前导零
        img_path = os.path.join(img_dir, filename)  # 获取图像的完整路径
        img_path_dict[img_id] = img_path
        if img_id == '262148':
            print('img_id:', img_id, 'img_path:', img_path, 'img_path_dict[img_id]:', img_path_dict[img_id])

# print('img_path_dict[262148]:', img_path_dict[262148])

# 加载并解析问题和答案的JSON文件
with open('v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
    questions_data = json.load(f)['questions']

# 读取并解析答案JSON文件
with open('v2_mscoco_val2014_annotations.json', 'r') as f:
    data = json.load(f)

# 提取答案数据
answers_data = data['annotations']

dataset = CustomDataset(img_path_dict, questions_data, answers_data, transform=transform)

from matplotlib import pyplot as plt


from torch.utils.data import DataLoader

# 创建一个DataLoader实例
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


answerset = []


first_batch = next(iter(dataloader))

# `first_batch`现在是一个包含批次数据的字典列表
# 你可以遍历这个列表来处理每个样本
for sample in first_batch:
    image = sample['image_info']
    question = sample['question']
    answers = sample['answers']
    # print(f"image: {image}")
    # print(f"question: {question}")
    # print(f"answers: {answers}")
    result = evaluation(image, question, answers)
    answerset.append(result)
    print(f"answerset: {answerset}")


print(f"Accuracy: {sum(answerset) / len(answerset)}")
