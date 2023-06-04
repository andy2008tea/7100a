import json
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

# URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


URL = "./testfiles/COCO_val2014_000000262148.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
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


def questions():
    # 加载 JSON 文件
    with open('v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
        data = json.load(f)

    questions = data['questions']

    # 创建一个空字典来存储 image_id 和对应的问题
    # image_to_questions = {}

    # # 填充字典
    # for question in questions:
    #     image_id = question['image_id']
    #     question_text = question['question']
    #     question_id = question['question_id']
    #
    #
    #     # 因为一个 image_id 可能对应多个问题，所以我们需要将问题添加到一个列表中
    #     if image_id not in image_to_questions:
    #         image_to_questions[image_id] = [question_text]
    #     else:
    #         image_to_questions[image_id].append(question_text)
    #
    #     # 现在你可以通过 image_id 来查询问题
    # image_id_to_search = 262148  # 替换为你想要查询的 image_id
    # if image_id_to_search in image_to_questions:
    #     print(f'Image ID {image_id_to_search} has the following questions:')
    #     for question in image_to_questions[image_id_to_search]:
    #         print(question)
    # else:
    #     print(f'Image ID {image_id_to_search} is not found in the dataset.')

    image_ids = [question['image_id'] for question in questions]
    unique_image_ids = set(image_ids)

    return questions, unique_image_ids


def annotations():
    # 加载 JSON 文件
    with open('v2_mscoco_val2014_annotations.json', 'r') as f:
        data = json.load(f)

        # data 是一个字典，其中包含多个键，如 'info', 'license', 'data_subtype', 'annotations' 等。
        # 我们主要关心 'annotations' 键，该键对应的值是一个包含所有标注信息的列表。
    annotations = data['annotations']

    # 我们可以打印第一个 annotation 来看看它的结构
    print(annotations[0])

    # 你可以遍历 annotations 列表，来处理每个 annotation
    for annotation in annotations:
        # 每个 annotation 也是一个字典，包含 'question_type', 'multiple_choice_answer', 'answers', 'image_id', 'answer_type', 'question_id' 等键。
        question_type = annotation['question_type']
        multiple_choice_answer = annotation['multiple_choice_answer']
        answers = annotation['answers']
        image_id = annotation['image_id']
        answer_type = annotation['answer_type']
        question_id = annotation['question_id']

        # 'answers' 键对应的值是一个包含10个答案的列表，每个答案是一个字典，包含 'answer_id', 'answer_confidence', 'answer' 等键。
        for answer in answers:
            answer_id = answer['answer_id']
            answer_confidence = answer['answer_confidence']
            answer_text = answer['answer']

    return annotations


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


def evaluation(imageid, qaset):
    # [item['answer'] for item in qa_array]
    # imgname = matching[0]["image_id"]
    flie_path = "E:/MasterOfAdelaideUniversity/7100/Data/VQA2/val2014/COCO_val2014_" + str(imageid).zfill(12) + ".jpg"
    qa_array = []

    # %%

    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # image viz
    # 具体而言，SingleImageViz(URL, id2obj=objids, id2attr=attrids) 的作用是：
    #
    # 提供图像的 URL，以便从网络或本地加载图像。
    # 提供目标对象的 ID 到类别名称的映射（id2obj），用于将模型预测的目标对象 ID 转换为对应的类别名称。
    # 提供属性的 ID 到属性名称的映射（id2attr），用于将模型预测的属性 ID 转换为对应的属性名称。
    # 通过这个可视化对象，你可以调用其方法来将目标检测结果可视化，并在图像上标注出检测到的目标对象以及其属性。
    frcnn_visualizer = SingleImageViz(flie_path, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = image_preprocess(flie_path)

    # 具体而言，这段代码的作用是：
    #
    # 将图像 images 和图像的尺寸 sizes 作为输入传递给 Faster R-CNN 模型 frcnn。
    # 使用 scales_yx 参数来指定图像的缩放比例。
    # 使用 padding="max_detections" 参数来指定在图像中填充目标检测结果的方式。
    # 使用 max_detections=frcnn_cfg.max_detections 参数来指定允许的最大检测数。
    # 使用 return_tensors="pt" 参数来指定返回的输出结果的张量类型为 PyTorch 张量。
    # 执行这段代码后，将得到一个名为 output_dict 的字典，其中包含了模型的输出结果。这些输出结果可能包括检测到的目标对象的边界框坐标、类别标签、置信度得分等信息。
    #
    # 通过访问 output_dict 中的不同字段，你可以获取模型输出的具体信息，进而对检测到的目标对象进行后续处理、可视化或其他任务。
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    # add boxes and labels to the image

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

    test_questions_for_url2 = [item['question'] for item in qaset]
    test_answers_for_url2 = [item['answers'] for item in qaset]

    print("test_questions_for_url2:", test_questions_for_url2)
    # Very important that the boxes are normalized
    # normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    # %%

    countright = 0
    for test_question in test_questions_for_url2:
        test_question = [test_question]

        inputs = bert_tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        output_vqa = visualbert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        # get prediction
        pred_vqa = output_vqa["logits"].argmax(-1)

        # print("test_question:", test_question)
        # print("qaset:", qaset)
        # print("test_answers_for_url2:", len(test_answers_for_url2))
        find_r_answer = [item['answers'] for item in qaset if item['question'] == test_question[0]]
        # print("Question:", test_question)
        # print("prediction from VisualBert VQA:", vqa_answers[pred_vqa])
        # print("find_r_answer:", [item['answer'] for item in sum(find_r_answer, [])])

        if vqa_answers[pred_vqa] in [item['answer']for item in sum(find_r_answer, [])]:
            countright = countright + 1

    print("countright/len(test_questions_for_url2):", countright / len(test_questions_for_url2))
    return countright / len(test_questions_for_url2)


questions, unique_image_ids = questions()
annotations = annotations()
sample_image_ids = random.sample(unique_image_ids, 2)
print("annotations:", annotations[0])
answerset = []
for id in sample_image_ids:
    # matching_questions = [q['question'] for q in questions if q['image_id'] == id]
    matching_questions = [{'question': q['question'], 'question_id': q['question_id'], 'image_id': q['image_id']} for
                          q in questions if q['image_id'] == id]
    # print("matching_questions:", matching_questions)
    imgname = matching_questions[0]["image_id"]
    print("imgname:", imgname)

    # all_questions = [{item['question'], item['question_id']} for item in matching_questions]
    matching_answers = [{'answers': item['answers'], 'question_id': item['question_id']} for item in annotations if
                        item['image_id'] == id]

    # print("matching_answers:", matching_answers)

    # 合并问题和答案
    merged = []
    for mq in matching_questions:
        # print("mq", mq)
        # print("mq['question_id']:", mq['question_id'])
        for ma in matching_answers:
            # print("ma['question_id']:", ma['question_id'])
            if mq['question_id'] == ma['question_id']:
                # print("mq['question_id']:", mq['question_id'])
                # print("ma['question_id']:", ma['question_id'])
                # 将问题和答案合并为一个字典
                merged_item = {**mq, **ma}
                merged.append(merged_item)
    # for mq in matching_questions:

    result = evaluation(id, merged)
    answerset.append(result)
    # for item in merged:
    #     print(f"item: {item}")
    #     print(f"Question ID: {item['question_id']}")
    #     print(f"Image ID: {item['image_id']}")
    #     print(f"Question: {item['question']}")
    #     print(f"Answer: {item['answers']}")
    #     print("-----------------------------")
print(f"Accuracy: {sum(answerset) / len(answerset)}")
