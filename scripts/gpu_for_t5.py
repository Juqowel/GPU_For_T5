import torch
import json
import gradio as gr
from modules import scripts
from modules import shared
from modules.ui_components import InputAccordion
from backend import memory_management

T5D_t2i_nbl_key = "customscript/gpu_for_t5.py/txt2img/Enabled/value"
T5D_i2i_nbl_key = "customscript/gpu_for_t5.py/img2img/Enabled/value"
T5D_t2i_D_key = "customscript/gpu_for_t5.py/txt2img/Select device/value"
T5D_i2i_D_key = "customscript/gpu_for_t5.py/img2img/Select device/value"
T5D_t2i_lvr_key = "customscript/gpu_for_t5.py/txt2img/LowVRAM/value"
T5D_i2i_lvr_key = "customscript/gpu_for_t5.py/img2img/LowVRAM/value"

CONFIG = shared.cmd_opts.ui_config_file

with open(CONFIG, 'r', encoding="utf-8") as json_file:
    ui_config = json.load(json_file)

T5D_t2i_nbl = ui_config[T5D_t2i_nbl_key] if T5D_t2i_nbl_key in ui_config else False
T5D_i2i_nbl = ui_config[T5D_i2i_nbl_key] if T5D_i2i_nbl_key in ui_config  else False
T5D_t2i_D = ui_config[T5D_t2i_D_key] if  T5D_t2i_D_key in ui_config else 'cpu'
T5D_i2i_D = ui_config[T5D_i2i_D_key] if T5D_i2i_D_key in ui_config else 'cpu'

class T5onOtherDevice(scripts.Script):

    def title(self):
        return "T5 on Other Device"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        current_gpu = memory_management.get_torch_device()
        device_list = [(torch.cuda.get_device_name(f"cuda:{i}"),f'cuda:{i}') for i in range(0, torch.cuda.device_count())]
        device_list.append(('CPU', 'cpu'))
        is_i2i = is_img2img

        if len(device_list) < 2:
             return []
        
        def nbl_toggle():
            key = T5D_i2i_nbl_key if is_i2i else T5D_t2i_nbl_key
            with open(CONFIG, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
            if key in data:
                data[key] = not data[key]
            else:
                data[key] = True

            with open(CONFIG, 'w', encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"{ f'T5 on Other Device is enabled for' if data[key] else f'T5 on Other Device is disabled for'}{' img2img' if is_i2i else ' txt2img'}")

        def choise_toggle(device):
            key = T5D_i2i_D_key if is_i2i else T5D_t2i_D_key
            with open(CONFIG, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
            data[key] = device

            with open(CONFIG, 'w', encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"{ f'T5 on Other Device - device changed to {device} for'}{' img2img' if is_i2i else ' txt2img'}")

        
        device_list.remove((torch.cuda.get_device_name(current_gpu),f'{current_gpu}'))
        
        with InputAccordion(T5D_i2i_nbl if is_img2img else T5D_t2i_nbl, label="T5 on Other Device", elem_id="T5onOtherDevice_enabled"+f"{is_img2img}") as enabled:
            device = T5D_i2i_D if is_img2img else T5D_t2i_D
            if (device not in list(sum(device_list, ()))):
                device = device_list[0][1]
            choise = gr.Radio(label='Select device', choices=device_list, value=device)

            enabled.change(fn=nbl_toggle)
            choise.change(fn=choise_toggle, inputs=[choise])

            gr.Markdown("""**Note:** T5 GGUF is not supported.""")
        return [choise, enabled]

    def process(self, p, *script_args, **kwargs):
        if not script_args[0]:
            return
        if not script_args[1]:
            p.sd_model.forge_objects.clip.patcher.load_device=torch.device(f'{memory_management.get_torch_device()}')
            return
        p.sd_model.forge_objects.clip.patcher.load_device=torch.device(f'{script_args[0]}')
        return