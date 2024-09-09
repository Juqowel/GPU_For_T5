import torch
import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
from backend import memory_management

lowvram_available = memory_management.lowvram_available

class T5onOtherDevice(scripts.Script):

    def title(self):
        return "T5 on Other Device"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        current_gpu = memory_management.get_torch_device()
        device_list = [(torch.cuda.get_device_name(f"cuda:{i}"),f'cuda:{i}') for i in range(0, torch.cuda.device_count())]
        device_list.append(('CPU', 'cpu'))

        if len(device_list) < 2:
             return []
        
        device_list.remove((torch.cuda.get_device_name(current_gpu),f'{current_gpu}'))
        
        with InputAccordion(False, label="T5 on Other Device", elem_id="T5onOtherDevice_enabled"+f"{is_img2img}") as enabled:
            choise = gr.Radio(label='Select device', choices=device_list, value=device_list[0][1])
            enabled.change(lambda i, c: print(f"{ f'T5 on Other Device is enabled for' if i else f'T5 on Other Device is disabled for'}{' img2img' if is_img2img else ' txt2img'} and {c}"), inputs=[enabled, choise], outputs=[])
            choise.change(lambda i, c: print(f"{f'T5 on Other Device is enabled for' if i else f'T5 on Other Device is disabled for'} {'img2img' if is_img2img else 'txt2img'} and {c}"), inputs=[enabled, choise], outputs=[])
        return [choise, enabled]

    def process(self, p, *script_args, **kwargs):
        if not script_args[0]:
            return
        if not script_args[1]:
            p.sd_model.forge_objects.clip.patcher.load_device=torch.device(f'{memory_management.get_torch_device()}')
            memory_management.lowvram_available = lowvram_available
            return
        memory_management.lowvram_available = False
        p.sd_model.forge_objects.clip.patcher.load_device=torch.device(f'{script_args[0]}')
        return