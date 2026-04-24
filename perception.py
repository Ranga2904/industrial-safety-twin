import numpy as np
import cv2
import os
from openai import OpenAI

# ── Image Generator (unchanged) ─────────────────────────────────────────────

def generate_equipment_state(
    valve_states=[1, 1, 1, 0],
    tools_present=True,
    person_present=False,
    pressure_gauge=None
):
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (550, 350), (80, 80, 80), -1)

    valve_positions = [(150, 150), (350, 150), (150, 250), (350, 250)]
    valve_labels    = ['V1', 'V2', 'V3', 'V4']

    for pos, label, state in zip(valve_positions, valve_labels, valve_states):
        color = (0, 200, 0) if state == 1 else (0, 0, 200)
        cv2.circle(img, pos, 30, color, -1)
        cv2.putText(img, label, (pos[0]-15, pos[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if pressure_gauge is not None:
        cv2.circle(img, (500, 100), 40, (150, 150, 150), -1)
        cv2.putText(img, f"{pressure_gauge}",
                    (475, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

    if tools_present:
        cv2.rectangle(img, (450, 280), (530, 330), (150, 150, 150), -1)

    if person_present:
        cv2.circle(img, (100, 120), 25, (255, 200, 150), -1)

    return img

import base64, cv2, os
from openai import OpenAI

class VLMAnalyzer:
    """
    Sends the actual synthetic image to a vision-language model.
    The model genuinely sees the image — not a text description.
    """
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"]
        )
        self.vlm   = "meta/llama-3.2-11b-vision-instruct"
        self.llm   = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

    def _image_to_b64(self, img_array) -> str:
        _, buf = cv2.imencode('.jpg', img_array)
        return base64.b64encode(buf).decode('utf-8')

    def analyze_scene(self, img_array) -> dict:
        b64 = self._image_to_b64(img_array)

        # Step 1 — VLM observes the actual image
        vision_response = self.client.chat.completions.create(
            model=self.vlm,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing an industrial control panel image.\n"
                            "Identify and report ONLY:\n"
                            "- Each circle labelled V1-V4: is it GREEN (open) or RED/BLUE (closed)?\n"
                            "- Is there a grey rectangle in the lower right (tools present)?\n"
                            "- Is there a flesh-coloured circle in the upper left (worker present)?\n"
                            "- What number appears near the grey circle top-right (pressure gauge)?\n"
                            "Reply in this exact format:\n"
                            "V1: OPEN or CLOSED\n"
                            "V2: OPEN or CLOSED\n"
                            "V3: OPEN or CLOSED\n"
                            "V4: OPEN or CLOSED\n"
                            "TOOLS: YES or NO\n"
                            "WORKER: YES or NO\n"
                            "PRESSURE: [number] or UNKNOWN"
                        )
                    }
                ]
            }],
            max_tokens=256
        )
        raw = vision_response.choices[0].message.content
        print(f"\n[VLM raw observation]\n{raw}\n")
        return self._parse_output(raw)

    def _parse_output(self, text: str) -> dict:
        import re
        result = {
            'valves_open': [], 'valves_closed': [],
            'pressure_psi': 0, 'tools_present': False,
            'worker_present': False, 'raw_response': text
        }
        for line in text.upper().splitlines():
            for v in ['V1','V2','V3','V4']:
                if line.startswith(v + ':'):
                    if 'OPEN' in line or 'GREEN' in line:
                        result['valves_open'].append(v)
                    else:
                        result['valves_closed'].append(v)
            if line.startswith('TOOLS:'):
                result['tools_present'] = 'YES' in line
            if line.startswith('WORKER:'):
                result['worker_present'] = 'YES' in line
            if line.startswith('PRESSURE:'):
                m = re.search(r'\d+', line)
                if m:
                    result['pressure_psi'] = int(m.group())
        return result
