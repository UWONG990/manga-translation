from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import base64
import html
import json


def resolve_font_path(font_path: str | Path | None = None) -> str | None:
    if font_path:
        resolved = Path(font_path)
        if resolved.exists():
            return str(resolved)
    return None


def hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (0, 0, 0, alpha)
    try:
        return (*[int(value[i:i + 2], 16) for i in range(0, 6, 2)], alpha)
    except ValueError:
        return (0, 0, 0, alpha)


def _load_font(size: int, font_path: str | Path | None = None) -> ImageFont.ImageFont:
    resolved = resolve_font_path(font_path)
    if resolved:
        try:
            return ImageFont.truetype(resolved, size=size)
        except OSError:
            pass

    candidate_paths = [
        "C:/Windows/Fonts/anime-ace.regular.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/mingliu.ttc",
        "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "c:/Windows/Fonts/anime-ace.regular.ttf",
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: float) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return [""]

    words = stripped.split()
    if not words:
        return [stripped]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)

    # Fallback for long single words / punctuation-heavy strings.
    if len(lines) == 1 and draw.textbbox((0, 0), lines[0], font=font)[2] > max_width:
        chars = []
        current = ""
        for ch in stripped:
            candidate = current + ch
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                if current:
                    chars.append(current)
                current = ch
        if current:
            chars.append(current)
        return chars

    return lines


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[float, float, float, float],
    font_path: str | Path | None = None,
    max_font_size: int = 34,
) -> ImageFont.ImageFont:
    left, top, right, bottom = box
    max_w = max(12.0, right - left)
    max_h = max(12.0, bottom - top)

    for size in range(max(10, min(34, max_font_size)), 7, -1):
        font = _load_font(size, font_path)
        wrapped = _wrap_text(draw, text, font, max_w * 0.9)
        line_height = draw.textbbox((0, 0), "A", font=font)[3]
        total_h = len(wrapped) * line_height * 1.25
        if total_h <= max_h * 0.9:
            return font
    return _load_font(10, font_path)


def render_translated_page(
    page_image_path: str | Path,
    translated_items: list[dict],
    output_path: str | Path,
    font_path: str | Path | None = None,
    max_font_size: int = 34,
    text_color: str = "#000000",
    bubble_fill: str = "#FFFFFF",
    bubble_outline: str = "#141414",
    bubble_padding: float = 1.0,
    alignment: str = "center",
):
    page_image = Image.open(page_image_path).convert("RGBA")
    canvas = Image.new("RGBA", page_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for item in sorted(translated_items, key=lambda x: x.get("index", 0)):
        bbox = item.get("bbox")
        translated = item.get("translated_text", "")
        if not bbox or len(bbox) < 4 or not translated:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1 or y2 <= y1:
            continue

        pad = max(8.0, min(18.0, (y2 - y1) * 0.12)) * bubble_padding
        bubble_x1 = max(0.0, x1 - pad * 0.6)
        bubble_y1 = max(0.0, y1 - pad * 0.4)
        bubble_x2 = max(0.0, x2 + pad * 0.6)
        bubble_y2 = max(0.0, y2 + pad * 0.4)

        if bubble_x2 <= bubble_x1 or bubble_y2 <= bubble_y1:
            continue

        radius = max(6, min(26, min((bubble_x2 - bubble_x1) / 4, (bubble_y2 - bubble_y1) / 4)))
        draw.rounded_rectangle(
            [bubble_x1, bubble_y1, bubble_x2, bubble_y2],
            radius=int(radius),
            fill=hex_to_rgba(bubble_fill, 230),
            outline=hex_to_rgba(bubble_outline, 220),
            width=2,
        )

        inner_x1 = bubble_x1 + pad * 0.7
        inner_y1 = bubble_y1 + pad * 0.6
        inner_x2 = bubble_x2 - pad * 0.7
        inner_y2 = bubble_y2 - pad * 0.6
        font = _fit_font(
            draw,
            translated,
            (inner_x1, inner_y1, inner_x2, inner_y2),
            font_path,
            max_font_size,
        )
        lines = _wrap_text(draw, translated, font, inner_x2 - inner_x1)
        line_height = draw.textbbox((0, 0), "A", font=font)[3]
        total_h = len(lines) * line_height * 1.25
        start_y = inner_y1 + max(0, (inner_y2 - inner_y1 - total_h) / 2)

        for i, line in enumerate(lines):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            if alignment == "left":
                x = inner_x1
            elif alignment == "right":
                x = inner_x2 - line_w
            else:
                x = inner_x1 + max(0, (inner_x2 - inner_x1 - line_w) / 2)
            y = start_y + i * line_height * 1.25
            draw.text(
                (x, y),
                line,
                fill=hex_to_rgba(text_color),
                font=font,
                stroke_width=1,
                stroke_fill=hex_to_rgba(bubble_fill, 180),
            )

    result = Image.alpha_composite(page_image, canvas).convert("RGB")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return str(output)


def build_editable_project(
    page_image_path: str | Path,
    cleaned_image_path: str | Path,
    translated_items: list[dict],
    font_path: str | Path | None = None,
) -> dict:
    with Image.open(page_image_path) as page:
        width, height = page.size

    bubbles = []
    for item in sorted(translated_items, key=lambda value: value.get("index", 0)):
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        text = str(item.get("translated_text", "")).strip()
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        bubble_width = max(12.0, x2 - x1)
        bubble_height = max(12.0, y2 - y1)
        estimated_chars = max(8, int(bubble_width / 8))
        wrapped_chars = max(8, min(estimated_chars, 28))
        words = text.split()
        line_count = 1
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= wrapped_chars:
                current = candidate
            else:
                line_count += 1
                current = word
        initial_font_size = max(
            10,
            min(
                24,
                int(bubble_height / max(1.25 * line_count, 1.0)),
                int(bubble_width / 8),
            ),
        )
        bubbles.append({
            "id": f"bubble-{item.get('index', len(bubbles))}",
            "text": text,
            "source_text": item.get("source_text", ""),
            "bbox": [float(value) for value in bbox[:4]],
            "font": str(font_path) if font_path else "Automatic",
            "font_size": initial_font_size,
            "auto_fit": True,
            "text_color": "#000000",
            "bubble_fill": "#FFFFFF",
            "bubble_outline": "#141414",
            "bubble_padding": 1.0,
            "alignment": "center",
            "visible": True,
        })

    return {
        "version": 1,
        "canvas": {
            "width": width,
            "height": height,
            "source_image": str(Path(page_image_path).resolve()),
            "cleaned_image": str(Path(cleaned_image_path).resolve()),
        },
        "bubbles": bubbles,
    }


def save_editable_project(project: dict, output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def _data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def export_editor_html(project: dict, output_path: str | Path) -> str:
    canvas = project["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    data_uri = _data_uri(canvas["cleaned_image"])
    project_json = json.dumps(project, ensure_ascii=False)

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Manga Typesetter</title>
<style>
body {{ margin:0; background:#f5f5f5; color:#202124; font-family:Segoe UI,Arial,sans-serif; overflow-x:hidden; }}
body.dark {{ background:#151515; color:#ffffff; }}
#toolbar {{ position:sticky; top:0; z-index:3; background:#ffffff; color:#202124; border-bottom:1px solid #d9dde3; padding:8px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
body.dark #toolbar {{ background:#202020; color:#ffffff; border-bottom-color:#3a3a3a; }}
#toolbar label {{ display:flex; align-items:center; gap:4px; }}
#viewport {{ width:100%; height:calc(100vh - 58px); overflow:auto; background:#eceff3; }}
body.dark #viewport {{ background:#252525; }}
#stageShell {{ position:relative; margin:12px auto; }}
#editor {{ position:absolute; left:0; top:0; width:{width}px; height:{height}px; transform-origin:0 0; background:url("{data_uri}") center/contain no-repeat; background-size:{width}px {height}px; --handle-scale:1; }}
.bubble {{ position:absolute; border:2px solid #141414; border-radius:12px; background:rgba(255,255,255,.92); box-sizing:border-box; cursor:move; touch-action:none; }}
.bubble.selected {{ outline:3px solid #4da3ff; }}
.bubble textarea {{ width:100%; height:100%; resize:none; border:0; outline:0; background:transparent; text-align:center; box-sizing:border-box; padding:8px; overflow:hidden; white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal; line-height:1.2; }}
.handle {{ position:absolute; background:#4da3ff; border:1px solid #fff; border-radius:3px; display:none; z-index:5; box-sizing:border-box; transform:scale(var(--handle-scale,1)); }}
.bubble.selected .handle {{ display:block; }}
.h-nw {{ width:12px; height:12px; left:-6px; top:-6px; cursor:nwse-resize; border-radius:50%; }}
.h-n {{ width:18px; height:8px; left:50%; top:-4px; margin-left:-9px; cursor:ns-resize; }}
.h-ne {{ width:12px; height:12px; right:-6px; top:-6px; cursor:nesw-resize; border-radius:50%; }}
.h-e {{ width:8px; height:18px; right:-4px; top:50%; margin-top:-9px; cursor:ew-resize; }}
.h-se {{ width:12px; height:12px; right:-6px; bottom:-6px; cursor:nwse-resize; border-radius:50%; }}
.h-s {{ width:18px; height:8px; left:50%; bottom:-4px; margin-left:-9px; cursor:ns-resize; }}
.h-sw {{ width:12px; height:12px; left:-6px; bottom:-6px; cursor:nesw-resize; border-radius:50%; }}
.h-w {{ width:8px; height:18px; left:-4px; top:50%; margin-top:-9px; cursor:ew-resize; }}
</style></head><body>
<div id="toolbar">
  <label>Font <select id="fontChoice"><option>Automatic</option><option>Arial</option><option>Comic Sans MS</option><option>Calibri</option><option>Consolas</option><option>Segoe UI</option><option>Anime Ace</option></select></label>
  <label>Size <input id="fontSize" type="number" min="8" max="96" value="24"></label>
  <label>Text <input id="textColor" type="color" value="#000000"></label>
  <label>Transparent text <input id="transparentText" type="checkbox"></label>
  <label>Fill <input id="fillColor" type="color" value="#ffffff"></label>
  <label>Transparent fill <input id="transparentFill" type="checkbox"></label>
  <label>Outline <input id="outlineColor" type="color" value="#141414"></label>
  <label>Transparent outline <input id="transparentOutline" type="checkbox"></label>
  <label>Alignment <select id="alignment"><option>left</option><option>center</option><option>right</option></select></label>
  <label>Dark mode <input id="themeToggle" type="checkbox"></label>
  <button id="enableAutoFit">Auto fit text</button>
  <button id="saveJson">Save project JSON</button>
  <button id="exportPng">Export final PNG</button>
</div>
<div id="viewport"><div id="stageShell"><div id="editor"></div></div></div>
<script>
const project = {project_json};
const editor = document.getElementById("editor");
let selected = null;

const themeToggle = document.getElementById("themeToggle");
const savedTheme = localStorage.getItem("mangaTypesetterTheme");
themeToggle.checked = savedTheme === "dark";
document.body.classList.toggle("dark", themeToggle.checked);
themeToggle.addEventListener("change", () => {{
  document.body.classList.toggle("dark", themeToggle.checked);
  localStorage.setItem("mangaTypesetterTheme", themeToggle.checked ? "dark" : "light");
}});

function wrapAndFit(el, bubble) {{
  if (bubble.auto_fit === false) {{
    el.querySelector("textarea").style.fontSize = bubble.font_size + "px";
    return;
  }}
  const textarea = el.querySelector("textarea");
  textarea.style.width = "100%";
  textarea.style.height = "100%";
  textarea.style.whiteSpace = "pre-wrap";
  textarea.style.overflow = "hidden";
  const innerWidth = Math.max(20, el.clientWidth - 16);
  const innerHeight = Math.max(20, el.clientHeight - 16);
  const fits = (size) => {{
    textarea.style.fontSize = size + "px";
    return textarea.scrollHeight <= innerHeight && textarea.scrollWidth <= innerWidth;
  }};
  let best = 24;
  if (fits(24)) {{
    let lo = 24, hi = 300;
    while (lo <= hi) {{
      const mid = (lo + hi) >> 1;
      if (fits(mid)) {{ best = mid; lo = mid + 1; }}
      else {{ hi = mid - 1; }}
    }}
  }}
  const size = best;
  bubble.font_size = size;
  textarea.style.fontSize = size + "px";
}}

const stageShell = document.getElementById("stageShell");
const viewport = document.getElementById("viewport");
let stageScale = 1;
function fitStage() {{
  const availableWidth = Math.max(320, viewport.clientWidth - 24);
  const availableHeight = Math.max(240, viewport.clientHeight - 24);
  const scale = Math.min(1, availableWidth / {width}, availableHeight / {height});
  stageScale = scale;
  editor.style.transform = "scale(" + scale + ")";
  editor.style.setProperty("--handle-scale", (1 / scale));
  stageShell.style.width = ({width} * scale) + "px";
  stageShell.style.height = ({height} * scale) + "px";
}}
window.addEventListener("resize", fitStage);

function selectBubble(el) {{
  if (selected) selected.classList.remove("selected");
  selected = el;
  selected.classList.add("selected");
  const bubble = project.bubbles.find(b => b.id === selected.dataset.id);
  document.getElementById("fontSize").value = bubble.font_size;
  document.getElementById("textColor").value = bubble.text_color;
  document.getElementById("fillColor").value = bubble.bubble_fill;
  document.getElementById("outlineColor").value = bubble.bubble_outline;
  document.getElementById("alignment").value = bubble.alignment;
  document.getElementById("transparentText").checked = bubble.text_color === "transparent";
  document.getElementById("transparentFill").checked = bubble.bubble_fill === "transparent";
  document.getElementById("transparentOutline").checked = bubble.bubble_outline === "transparent";
}}

function startResize(e, el, bubble, dir) {{
  e.preventDefault();
  e.stopPropagation();
  selectBubble(el);
  const startX=e.clientX, startY=e.clientY;
  const left0=el.offsetLeft, top0=el.offsetTop;
  const width0=el.offsetWidth, height0=el.offsetHeight;
  const move=(ev) => {{
    const dx=(ev.clientX-startX)/stageScale;
    const dy=(ev.clientY-startY)/stageScale;
    let newLeft=left0, newTop=top0, newW=width0, newH=height0;
    if (dir.includes("e")) newW=width0+dx;
    if (dir.includes("s")) newH=height0+dy;
    if (dir.includes("w")) {{ newW=width0-dx; newLeft=left0+dx; }}
    if (dir.includes("n")) {{ newH=height0-dy; newTop=top0+dy; }}
    if (newLeft<0) {{ newW+=newLeft; newLeft=0; }}
    if (newTop<0) {{ newH+=newTop; newTop=0; }}
    if (newLeft+newW>{width}) newW={width}-newLeft;
    if (newTop+newH>{height}) newH={height}-newTop;
    newW=Math.max(30,newW);
    newH=Math.max(20,newH);
    el.style.left=newLeft+"px";
    el.style.top=newTop+"px";
    el.style.width=newW+"px";
    el.style.height=newH+"px";
  }};
  const up=()=>{{
    document.removeEventListener("pointermove",move);
    document.removeEventListener("pointerup",up);
    sync();
  }};
  document.addEventListener("pointermove",move);
  document.addEventListener("pointerup",up);
}}

function renderBubble(bubble) {{
  const [x1,y1,x2,y2] = bubble.bbox;
  const el = document.createElement("div");
  el.className = "bubble";
  el.dataset.id = bubble.id;
  el.style.left = x1 + "px";
  el.style.top = y1 + "px";
  el.style.width = (x2-x1) + "px";
  el.style.height = (y2-y1) + "px";
  el.style.background = bubble.bubble_fill;
  el.style.borderColor = bubble.bubble_outline;
  el.style.color = bubble.text_color;
  el.innerHTML = '<textarea></textarea>';
  const textarea = el.querySelector("textarea");
  textarea.value = bubble.text;
  textarea.style.fontSize = bubble.font_size + "px";
  textarea.style.textAlign = bubble.alignment;
  ["nw","n","ne","e","se","s","sw","w"].forEach((dir) => {{
    const handle = document.createElement("div");
    handle.className = "handle h-" + dir;
    handle.dataset.dir = dir;
    handle.addEventListener("pointerdown", (ev) => startResize(ev, el, bubble, dir));
    el.appendChild(handle);
  }});
  requestAnimationFrame(() => wrapAndFit(el, bubble));
  el.addEventListener("pointerdown", e => {{
    if (e.target.classList.contains("handle")) return;
    selectBubble(el);
    const startX=e.clientX, startY=e.clientY, left=el.offsetLeft, top=el.offsetTop;
    function move(ev) {{
      el.style.left = Math.max(0, Math.min({width}-el.offsetWidth, left+(ev.clientX-startX)/stageScale))+"px";
      el.style.top = Math.max(0, Math.min({height}-el.offsetHeight, top+(ev.clientY-startY)/stageScale))+"px";
    }}
    function up() {{
      document.removeEventListener("pointermove",move);
      document.removeEventListener("pointerup",up);
      sync();
      if (bubble.auto_fit) requestAnimationFrame(() => wrapAndFit(el, bubble));
    }}
    document.addEventListener("pointermove",move); document.addEventListener("pointerup",up);
  }});
  textarea.addEventListener("input", sync);
  textarea.addEventListener("input", () => {{
    if (bubble.auto_fit) {{
      wrapAndFit(el, bubble);
    }}
  }});
  editor.appendChild(el);
}}
function sync() {{
  if (!selected) return;
  const bubble=project.bubbles.find(b=>b.id===selected.dataset.id);
  bubble.text=selected.querySelector("textarea").value;
  const explicitFontChange = document.activeElement === document.getElementById("fontSize");
  if (explicitFontChange) {{
    bubble.font_size=parseInt(document.getElementById("fontSize").value)||24;
    bubble.auto_fit=false;
  }}
  bubble.text_color=document.getElementById("transparentText").checked
    ? "transparent"
    : document.getElementById("textColor").value;
  bubble.bubble_fill=document.getElementById("transparentFill").checked
    ? "transparent"
    : document.getElementById("fillColor").value;
  bubble.bubble_outline=document.getElementById("transparentOutline").checked
    ? "transparent"
    : document.getElementById("outlineColor").value;
  bubble.alignment=document.getElementById("alignment").value;
  const textarea=selected.querySelector("textarea");
  textarea.style.fontSize=bubble.font_size+"px";
  textarea.style.textAlign=bubble.alignment;
  selected.style.color=bubble.text_color==="transparent"?"transparent":bubble.text_color;
  selected.style.background=bubble.bubble_fill==="transparent"?"transparent":bubble.bubble_fill;
  selected.style.borderColor=bubble.bubble_outline==="transparent"
    ? "transparent"
    : bubble.bubble_outline;
  const font=document.getElementById("fontChoice").value;
  bubble.font=font;
  textarea.style.fontFamily=font==="Automatic"?"inherit":font;
  bubble.bbox=[
    selected.offsetLeft, selected.offsetTop,
    selected.offsetLeft+selected.offsetWidth, selected.offsetTop+selected.offsetHeight
  ];
  requestAnimationFrame(() => wrapAndFit(selected, bubble));
}}
["fontSize","textColor","fillColor","outlineColor","alignment","fontChoice","transparentText","transparentFill","transparentOutline"].forEach(id =>
  document.getElementById(id).addEventListener("change", sync));
document.getElementById("fontSize").addEventListener("input", () => {{
  if (!selected) return;
  const bubble=project.bubbles.find(b=>b.id===selected.dataset.id);
  bubble.font_size=parseInt(document.getElementById("fontSize").value)||34;
  bubble.auto_fit=false;
  selected.querySelector("textarea").style.fontSize=bubble.font_size+"px";
}});
document.getElementById("enableAutoFit").addEventListener("click", () => {{
  if (!selected) return;
  const bubble=project.bubbles.find(b=>b.id===selected.dataset.id);
  bubble.auto_fit=true;
  bubble.font_size=24;
  document.getElementById("fontSize").value=24;
  wrapAndFit(selected,bubble);
}});
project.bubbles.forEach(renderBubble);
fitStage();
document.getElementById("saveJson").addEventListener("click", () => {{
  sync();
  const blob=new Blob([JSON.stringify(project,null,2)],{{type:"application/json"}});
  const link=document.createElement("a"); link.href=URL.createObjectURL(blob); link.download="manga_typesetter_project.json"; link.click();
}});
document.getElementById("exportPng").addEventListener("click", () => {{
  sync();
  const exportCanvas=document.createElement("canvas");
  exportCanvas.width={width};
  exportCanvas.height={height};
  const ctx=exportCanvas.getContext("2d");
  const base=new Image();
  base.onload=()=>{{
    ctx.drawImage(base,0,0);
    project.bubbles.forEach(bubble=>{{
      if (!bubble.visible) return;
      const [x1,y1,x2,y2]=bubble.bbox;
      if (bubble.bubble_fill !== "transparent") {{
        ctx.fillStyle=bubble.bubble_fill;
        ctx.beginPath();
        ctx.roundRect(x1,y1,x2-x1,y2-y1,12);
        ctx.fill();
      }}
      ctx.strokeStyle=bubble.bubble_outline;
      ctx.lineWidth=2;
      if (bubble.bubble_outline !== "transparent") {{
        ctx.beginPath();
        ctx.roundRect(x1,y1,x2-x1,y2-y1,12);
        ctx.stroke();
      }}
      if (bubble.text_color === "transparent") return;
      ctx.fillStyle=bubble.text_color;
      const px=bubble.font_size||24;
      const family=(bubble.font&&bubble.font!="Automatic")?bubble.font:"sans-serif";
      ctx.font=px+"px "+family;
      ctx.textAlign=bubble.alignment;
      ctx.textBaseline="middle";
      const anchor=bubble.alignment==="left"?x1+10:bubble.alignment==="right"?x2-10:(x1+x2)/2;
      const lines=bubble.text.split(/\\n|\\s{2,}/);
      const lineHeight=bubble.font_size*1.25;
      lines.forEach((line,index)=>ctx.fillText(line,anchor,y1+(y2-y1)/2+(index-(lines.length-1)/2)*lineHeight));
    }});
    exportCanvas.toBlob(blob=>{{
      const link=document.createElement("a"); link.href=URL.createObjectURL(blob); link.download="manga_translated.png"; link.click();
    }},"image/png");
  }};
  base.src={json.dumps(data_uri)};
}});
</script></body></html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return str(output)
