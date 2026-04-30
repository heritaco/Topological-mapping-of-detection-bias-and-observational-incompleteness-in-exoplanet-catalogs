from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / 'tex' / 'final'
FIG = TEX / 'figures'
SRC = TEX / 'source_reports'
FIG.mkdir(parents=True, exist_ok=True)

SOURCES = {
    'imputation': SRC / '01_imputation_pipeline_explanation.pdf',
    'mapper': SRC / '02_mapper_report.pdf',
    'toi': SRC / '03_topological_incompleteness_index.pdf',
    'case': SRC / '04_toi_ati_case_anatomy.pdf',
    'future': SRC / '05_toi_ati_future_validation.pdf',
    'local': SRC / '06_local_shadow_case_studies.pdf',
}

# Crop boxes are fractions of rendered page: (x0,y0,x1,y1)
# They crop the original generated figures from prior report PDFs.
CROPS = [
    ('fig02_missingness_before_after.png', 'imputation', 11, (0.10, 0.13, 0.88, 0.45)),
    ('fig03_mapper_coverage.png', 'imputation', 13, (0.13, 0.12, 0.78, 0.45)),
    ('fig04_mapper_candidate_population.png', 'mapper', 21, (0.07, 0.07, 0.94, 0.43)),
    ('fig05_mapper_imputation_fraction.png', 'mapper', 22, (0.08, 0.10, 0.93, 0.47)),
    ('fig06_final_region_synthesis_counts.png', 'mapper', 34, (0.14, 0.05, 0.82, 0.51)),
    ('fig07_selected_graphs_by_evidence.png', 'mapper', 35, (0.06, 0.06, 0.94, 0.42)),
    ('fig08_orbital_mapper_evidence_class.png', 'mapper', 36, (0.07, 0.09, 0.93, 0.50)),
    ('fig09_orbital_mapper_method.png', 'mapper', 37, (0.07, 0.05, 0.93, 0.43)),
    ('fig10_orbital_mapper_imputation.png', 'mapper', 37, (0.07, 0.48, 0.93, 0.82)),
    ('fig11_top_regions_toi.png', 'toi', 2, (0.08, 0.06, 0.93, 0.58)),
    ('fig12_top_anchors_ati.png', 'toi', 3, (0.08, 0.08, 0.93, 0.57)),
    ('fig13_toi_vs_shadow.png', 'toi', 4, (0.09, 0.06, 0.90, 0.64)),
    ('fig14_deficit_profiles_by_radius.png', 'future', 4, (0.12, 0.22, 0.92, 0.58)),
    ('fig15_ati_original_vs_conservative.png', 'future', 5, (0.14, 0.06, 0.88, 0.48)),
    ('fig16_observational_priority_ranking.png', 'future', 8, (0.09, 0.05, 0.95, 0.50)),
    ('fig17_hip90988_ego_network.png', 'local', 3, (0.08, 0.12, 0.92, 0.69)),
    ('fig18_hip90988_r3_projections.png', 'local', 4, (0.06, 0.15, 0.95, 0.42)),
    ('fig19_hip90988_method_composition.png', 'local', 5, (0.10, 0.16, 0.90, 0.44)),
    ('fig20_hip90988_r3_audit.png', 'local', 6, (0.16, 0.12, 0.84, 0.40)),
    ('fig21_hip90988_local_deficit.png', 'local', 7, (0.15, 0.12, 0.85, 0.44)),
    ('fig22_hip90988_rv_proxy.png', 'local', 8, (0.14, 0.09, 0.85, 0.38)),
]


def render_crop(pdf_path: Path, page_num: int, frac_box, out_path: Path, scale: float = 3.0):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    w, h = im.size
    x0, y0, x1, y1 = frac_box
    box = (int(x0*w), int(y0*h), int(x1*w), int(y1*h))
    crop = im.crop(box)
    # Add a thin white border to prevent visual crowding in LaTeX.
    bordered = Image.new('RGB', (crop.width + 24, crop.height + 24), 'white')
    bordered.paste(crop, (12, 12))
    bordered.save(out_path, optimize=True, quality=95)


def flowchart(out_path: Path):
    W, H = 2400, 520
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 44)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 34)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    except Exception:
        font_title = font = font_small = None
    colors = ['#E8F1FA', '#EAF6F0', '#FFF4E5', '#F4ECFA', '#EAF6F0', '#FFF4E5']
    labels = [
        ('Sesgo de\ndescubrimiento', 'metadata observacional'),
        ('Sombra\nobservacional', 'frontera local'),
        ('TOI(v)', 'prioridad regional'),
        ('ATI(p*)', 'ancla local'),
        ('ATI\nconservador', 'estabilidad multiescala'),
        ('Prioridad\nobservacional', 'ranking accionable'),
    ]
    margin = 80
    gap = 36
    box_w = int((W - 2*margin - gap*(len(labels)-1)) / len(labels))
    box_h = 215
    y = 160
    d.text((margin, 40), 'Cadena final de valor del proyecto', fill='#1B263B', font=font_title)
    for i,(lab,sub) in enumerate(labels):
        x = margin + i*(box_w+gap)
        d.rounded_rectangle([x,y,x+box_w,y+box_h], radius=28, fill=colors[i], outline='#263238', width=3)
        # center text lines
        lines = lab.split('\n')
        yy = y+38
        for line in lines:
            tw = d.textlength(line, font=font)
            d.text((x+(box_w-tw)/2, yy), line, fill='#111111', font=font)
            yy += 44
        tw = d.textlength(sub, font=font_small)
        d.text((x+(box_w-tw)/2, y+box_h-52), sub, fill='#37474F', font=font_small)
        if i < len(labels)-1:
            ax0 = x+box_w+8
            ax1 = x+box_w+gap-8
            ay = y+box_h//2
            d.line([ax0,ay,ax1,ay], fill='#263238', width=5)
            d.polygon([(ax1,ay),(ax1-18,ay-12),(ax1-18,ay+12)], fill='#263238')
    d.text((margin, 420), 'Lectura ejecutiva: convertir sesgos conocidos en señales ordenadas para decidir dónde mirar con mayor información esperada.', fill='#1B263B', font=font_small)
    img.save(out_path, optimize=True, quality=95)


def case_cards(out_path: Path):
    W, H = 2400, 1100
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    try:
        title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 48)
        head = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
        body = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    except Exception:
        title = head = body = small = None
    d.text((80, 50), 'Casos finales para exposición ejecutiva', fill='#1B263B', font=title)
    cards = [
        ('cube12_cluster0', 'Región top por TOI', 'TOI ≈ 0.077', 'Mejor región para explicar prioridad regional: sombra observacional, baja imputación, continuidad física y soporte de red.'),
        ('HIP 90988 b', 'Ancla conservadora principal', 'ATI_cons ≈ 0.006', 'Caso más defendible para la narrativa final: déficit pequeño, pero estable bajo radios locales.'),
        ('HD 42012 b', 'Transición Mapper', 'cube19_cluster5 / cube26_cluster6', 'Explica solapamiento de cubiertas: una misma ancla vive en más de un vecindario topológico.'),
        ('HD 4313 b', 'Candidato secundario estable', 'cube17_cluster10', 'Caso alternativo con déficit positivo multirradio y lectura local consistente.'),
    ]
    colors = ['#E8F1FA','#EAF6F0','#FFF4E5','#F4ECFA']
    x0s = [80, 1230, 80, 1230]
    y0s = [160, 160, 620, 620]
    for i,(name,role,metric,text) in enumerate(cards):
        x,y=x0s[i], y0s[i]
        d.rounded_rectangle([x,y,x+1070,y+360], radius=32, fill=colors[i], outline='#263238', width=3)
        d.text((x+36,y+36), name, fill='#111111', font=head)
        d.text((x+36,y+88), role, fill='#263238', font=body)
        d.rounded_rectangle([x+36,y+140,x+590,y+195], radius=20, fill='white', outline='#90A4AE', width=2)
        d.text((x+60,y+153), metric, fill='#1B263B', font=small)
        # wrap text
        words = text.split()
        line=''; yy=y+225
        for w in words:
            trial=(line+' '+w).strip()
            if d.textlength(trial, font=small) > 960:
                d.text((x+36,yy), line, fill='#263238', font=small)
                yy += 34
                line=w
            else:
                line=trial
        if line:
            d.text((x+36,yy), line, fill='#263238', font=small)
    d.text((80, 1030), 'Mensaje clave: el ranking final traduce estructura topológica en prioridades de inspección; favorece estabilidad, trazabilidad y utilidad observacional.', fill='#1B263B', font=small)
    img.save(out_path, optimize=True, quality=95)


def main():
    flowchart(FIG / 'fig01_pipeline_chain.png')
    case_cards(FIG / 'fig23_final_case_cards.png')
    for out, src, page, box in CROPS:
        render_crop(SOURCES[src], page, box, FIG / out)
        print('wrote', out)

if __name__ == '__main__':
    main()
