import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    try:
        # Aquí configuramos valores por defecto para --gpu-id y --cfg-path
        parser = argparse.ArgumentParser(description="MiniGPT-4 inference")

        # Parámetros con valores predeterminados
        parser.add_argument("--cfg-path", type=str, default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="text prompt for the model.")

        # Parámetros obligatorios
        parser.add_argument("--image", type=str, required=True, help="path to the image.")

        args = parser.parse_args()
        print(f"[DEBUG] Argumentos procesados: {args}")  # Depuración: mostrar los argumentos procesados
        return args
    except Exception as e:
        print(f"[ERROR] Error en la lectura de argumentos: {e}")
        raise


def setup_seeds(config):
    try:
        print("[DEBUG] Inicializando semillas")
        seed = config.run_cfg.seed + get_rank()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cudnn.benchmark = False
        cudnn.deterministic = True
    except Exception as e:
        print(f"[ERROR] Error al inicializar las semillas: {e}")
        raise


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

try:
    print('[DEBUG] Inicializando el modelo')
    args = parse_args()  # Obtienes los argumentos directamente
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    print(f"[DEBUG] Cargando modelo: {model_config.arch}")
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('[DEBUG] Inicialización del modelo terminada')
except Exception as e:
    print(f"[ERROR] Error durante la inicialización del modelo: {e}")
    raise


# ========================================
#             Inference Function
# ========================================

def infer(image_path, prompt_text):
    """
    Función para hacer la inferencia con la imagen y el mensaje de texto.
    """
    try:
        print(f"[DEBUG] Procesando la imagen: {image_path}")
        # Cargar imagen
        img = vis_processor.process(image_path)

        print(f"[DEBUG] Realizando inferencia con el prompt: {prompt_text}")
        # Ejecutar la inferencia
        llm_message = chat.answer(
            conv=CONV_VISION,
            img_list=[img],
            num_beams=3,
            temperature=0.7,
            max_new_tokens=300,
            max_length=2000
        )[0]

        print(f"[DEBUG] Respuesta generada: {llm_message}")
        return llm_message
    except Exception as e:
        print(f"[ERROR] Error durante la inferencia: {e}")
        raise


# ========================================
#             Main Execution
# ========================================

if __name__ == "__main__":
    try:
        print("[DEBUG] Comenzando ejecución principal")

        # Esto se llama con los argumentos de la línea de comandos
        image_path = args.image
        prompt_text = args.prompt

        print(f"[DEBUG] Imagen cargada: {image_path}")
        print(f"[DEBUG] Prompt recibido: {prompt_text}")

        # Llamar a la función de inferencia
        result = infer(image_path, prompt_text)

        # Imprimir o retornar la respuesta generada
        print(f"Respuesta generada por el modelo: {result}")
        print("[DEBUG] Ejecución terminada")
    except Exception as e:
        print(f"[ERROR] Error en la ejecución principal: {e}")
