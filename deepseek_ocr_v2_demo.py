import gradio as gr
import torch
import spaces
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import re
import numpy as np
import base64
from io import StringIO, BytesIO
from transformers import AutoModel, AutoTokenizer
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
"""

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME,
                                  _attn_implementation='flash_attention_2', 
                                  torch_dtype=torch.bfloat16, trust_remote_code=True, 
                                  use_safetensors=True)
model = model.eval().cuda()

MODEL_CONFIGS = {
    "Default": {"base_size": 1024, "image_size": 768, "crop_mode": True},
    "Quality": {"base_size": 1280, "image_size": 960, "crop_mode": True},
    "Fast": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "No Crop": {"base_size": 1024, "image_size": 768, "crop_mode": False},
    "Small": {"base_size": 768, "image_size": 512, "crop_mode": False},
}

TASK_PROMPTS = {
    "Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "OCR Image": {"prompt": "<image>\n<|grounding|>OCR this image.", "has_grounding": True},
    "Parse Figure": {"prompt": "<image>\nParse the figure.", "has_grounding": False},
    "Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "Custom": {"prompt": "", "has_grounding": False}
}

def extract_grounding_references(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)

def get_font(size=15):
    """Attempt to load a font, falling back to default if necessary."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
        "arial.ttf", 
        "Arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = get_font(15)
    crops = []
    
    color_map = {}
    np.random.seed(42)

    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))

        color = color_map[label]
        try:
            coords = eval(ref[2])
        except:
            continue
            
        color_a = color + (60,)
        
        for box in coords:
            x1, y1, x2, y2 = int(box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)
            
            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))
            
            width = 5 if label == 'title' else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)
            
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops

def clean_output(text, include_images=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            text = re.sub(rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)
    
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    return text.strip()

def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(f'**[Figure {i + 1}]**', f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown

@spaces.GPU
def process_image(image, mode, task, custom_prompt):
    if image is None:
        return "Error: Upload an image", "", "", None, []
    if task in ["Custom", "Locate"] and not custom_prompt.strip():
        return "Please enter a prompt", "", "", None, []
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    config = MODEL_CONFIGS[mode]
    
    if task == "Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task == "Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()
    
    stdout = sys.stdout
    sys.stdout = StringIO()
    
    model.infer(
        tokenizer=tokenizer, 
        prompt=prompt, 
        image_file=tmp.name, 
        output_path=out_dir,
        base_size=config["base_size"], 
        image_size=config["image_size"], 
        crop_mode=config["crop_mode"],
        save_results=False
    )
    
    result = '\n'.join([l for l in sys.stdout.getvalue().split('\n') 
                        if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size'])]).strip()
    sys.stdout = stdout
    
    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    
    if not result:
        return "No text detected", "", "", None, []
    
    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)
    
    img_out = None
    crops = []
    
    if has_grounding and '<|ref|>' in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)
    
    markdown = embed_images(markdown, crops)
    
    return cleaned, markdown, result, img_out, crops

def toggle_prompt(task):
    if task == "Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Add <|grounding|> for bounding boxes")
    elif task == "Locate":
        return gr.update(visible=True, label="Text to Locate", placeholder="Enter text to locate")
    return gr.update(visible=False)

def select_boxes(task):
    if task == "Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()

with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-OCR-2", elem_id="main-title")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])
            mode = gr.Dropdown(list(MODEL_CONFIGS.keys()), value="Default", label="Resolution")
            task = gr.Dropdown(list(TASK_PROMPTS.keys()), value="Markdown", label="Task")
            prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
            btn = gr.Button("Perform OCR", variant="primary", size="lg")
            
            examples = gr.Examples(
                examples=["examples/1.jpg", "examples/2.jpg", "examples/3.jpg"],
                inputs=image_input, 
                label="Examples"
            )
        
        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.Tab("Text", id="tab_text"):
                    text_out = gr.Textbox(lines=20, show_label=False)
                with gr.Tab("Markdown Preview", id="tab_markdown"):
                    md_out = gr.Markdown("")
                with gr.Tab("Boxes", id="tab_boxes"):
                    img_out = gr.Image(type="pil", height=500, show_label=False)
                with gr.Tab("Cropped Images", id="tab_crops"):
                    gallery = gr.Gallery(show_label=False, columns=3, height=400)
                with gr.Tab("Raw Text", id="tab_raw"):
                    raw_out = gr.Textbox(lines=20, show_label=False)
    
            with gr.Accordion("Note", open=False):
                gr.Markdown("Inference using Huggingface transformers on NVIDIA GPUs. This app is running with transformers version 4.46.3")
    
    task.change(toggle_prompt, [task], [prompt])
    
    submit_event = btn.click(
        process_image, 
        [image_input, mode, task, prompt],
        [text_out, md_out, raw_out, img_out, gallery]
    )
    submit_event.then(select_boxes, [task], [tabs])

if __name__ == "__main__":
    demo.queue(max_size=50).launch(theme=steel_blue_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)