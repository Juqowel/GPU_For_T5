import torch
import gradio as gr
from modules import scripts
from backend import memory_management

lowvram_available = memory_management.lowvram_available

class T5onOtherDevice(scripts.Script):

    def title(self):
        return "T5 on Other Device"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        gpus = [f'cuda:{i}' for i in range(0, torch.cuda.device_count())]
        if len(gpus) < 2:
             return []
        gpus.remove(f'{memory_management.get_torch_device()}')
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            choise = gr.Radio(label='Select GPU', choices=gpus, value=gpus[0])
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