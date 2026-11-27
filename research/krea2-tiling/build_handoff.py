"""Build the handoff from the archived captions, grades, prompts and image comparisons.

Usage: python build_handoff.py --revision GIT_COMMIT [--output handoff.pdf]
Requires ReportLab and Pillow. The repository includes all figure inputs.
"""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
import reportlab

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/tea-time-labs/sweet-tea-nodes"
ST = "8c97fe110adb6e3a1c92976c2fd7f3d82c2e0935"
COMFY = "7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b"
W, H, M = 460, 790, 29
INK, MUTED, TEAL, RULE, TINT = [colors.HexColor(s) for s in
                                 ("#192c30", "#415959", "#005e60", "#c7d6d0", "#eaf2ef")]
FONT_DIR = Path(reportlab.__file__).parent / "fonts"
for face, filename in (("Body", "Vera.ttf"), ("Heavy", "VeraBd.ttf")):
    pdfmetrics.registerFont(TTFont(face, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Heavy", italic="Body", boldItalic="Heavy")


class Report:
    def __init__(self, output, rev):
        self.c = canvas.Canvas(str(output), (W, H), pageCompression=1)
        self.rev = rev
        self.c.setTitle("Krea 2 Turbo | From 1 MP to 9 MP")
        self.c.setAuthor("Tea Time Labs · Upscaling research")
        self.n = 0
        self.y = 0
        self.sources = []
        self.pages = []

    def text(self, message, *, size=11.3, leading=15.6, face="Body", color=INK, after=7):
        style = ParagraphStyle("text", fontName=face, fontSize=size, leading=leading, textColor=color)
        block = Paragraph(message, style)
        _, height = block.wrap(W-2*M, H)
        self.y -= height
        block.drawOn(self.c, M, self.y)
        self.y -= after

    def title(self, section, heading, subtitle):
        if self.n:
            self.end()
        self.n += 1
        self.c.bookmarkPage("p"+str(self.n))
        self.c.addOutlineEntry(heading, "p"+str(self.n), 0)
        self.y = H-23
        self.text(section.upper(), size=8.2, leading=10, color=TEAL, face="Heavy", after=7)
        self.text(heading, size=22, leading=25, face="Heavy", after=6)
        self.text(subtitle, size=12, leading=16, color=MUTED, after=12)
        self.pages.append(heading)

    def sub(self, s):
        self.text(s, size=12.2, leading=15, face="Heavy", color=TEAL, after=4)

    def note(self, s):
        self.text(s, size=10.0, leading=13.6, color=MUTED, after=8)

    def grid(self, items, height):
        gap = 9
        tile = (W-2*M-gap*(len(items)-1))/len(items)
        bottom = self.y - height
        maxcap = 0
        for i, (file, label) in enumerate(items):
            p = ROOT/"evidence/figures"/(file+".jpg")
            with Image.open(p) as im:
                sw, sh = im.size
            scale = min(tile/sw, height/sh)
            w, h = sw*scale, sh*scale
            x = M+i*(tile+gap)
            self.c.setFillColor(TINT)
            self.c.rect(x, bottom, tile, height, fill=1, stroke=0)
            self.c.drawImage(str(p), x+(tile-w)/2, bottom+(height-h)/2, w, h)
            caption = Paragraph(escape(label), ParagraphStyle("caption",fontName="Heavy",
                                fontSize=8.6,leading=11,textColor=INK))
            _, capheight = caption.wrap(tile, 100)
            caption.drawOn(self.c, x, bottom-4-capheight)
            maxcap = max(maxcap, capheight)
        self.y = bottom-4-maxcap-9

    def table(self, rows, widths=None, size=9.2, compact=False):
        widths = widths or [1/len(rows[0])]*len(rows[0])
        cells = []
        for i, row in enumerate(rows):
            style = ParagraphStyle("cell",fontName="Heavy" if i==0 else "Body",
                                   textColor=TEAL if i==0 else INK,fontSize=size,leading=size*1.22)
            cells.append([Paragraph(escape(str(v)),style) for v in row])
        t = Table(cells,colWidths=[(W-2*M)*v for v in widths])
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,TINT]),
                               ("LINEBELOW",(0,0),(-1,0),.6,RULE),
                               ("LEFTPADDING",(0,0),(-1,-1),4),
                               ("RIGHTPADDING",(0,0),(-1,-1),3),
                               ("TOPPADDING",(0,0),(-1,-1),2 if compact else 4),
                               ("BOTTOMPADDING",(0,0),(-1,-1),2 if compact else 4)]))
        _, ht=t.wrap(W-2*M,H)
        self.y-=ht
        t.drawOn(self.c,M,self.y)
        self.y-=9

    def refs(self,*pairs):
        self.sources = list(pairs)

    def addr(self, path):
        if path.startswith("https:"):
            return path
        kind = "tree" if path.endswith("/") or not path else "blob"
        return f"{BASE}/{kind}/{self.rev}/research/krea2-tiling/{path}"

    def end(self):
        if self.y < 42:
            raise ValueError(f"Page {self.n} overflows at {self.y:.1f} points")
        c=self.c
        c.setStrokeColor(RULE); c.line(M,38,W-M,38)
        x=M
        for label, path in self.sources:
            c.setFont("Body",8); c.setFillColor(TEAL)
            c.drawString(x,25,label)
            right=x+pdfmetrics.stringWidth(label,"Body",8)
            c.linkURL(self.addr(path),(x,22,right,35),relative=0,thickness=0)
            x=right+12
        if x>W-49:
            raise ValueError(f"Footer links overflow on page {self.n}")
        c.setFont("Heavy",8); c.setFillColor(MUTED);c.drawRightString(W-M,24,str(self.n))
        c.showPage()
        self.sources=[]

    def finish(self):
        self.end()
        self.c.save()


def read(file):
    return json.loads((ROOT/file).read_text())


def write_report(r):
    t, p, s, g, note, tab, refs = r.title, r.text, r.sub, r.grid, r.note, r.table, r.refs
    t("Tea Time Labs • Research handoff • 27 November 2026","From 1 MP to 9 MP",
      "Krea 2 Turbo · Generative upscaling across nine overlapping tiles")
    p("The original request was to make a picture worth opening at 100%. Enlarge a roughly one-megapixel source three times in each direction. At the new size, an apple crate should have grain, a jacket should have weave, and faint structures in the distance can emerge. The same three people should still be lifting the same crate.")
    g([("scene-13","Orchard source · three people and one shared crate")],148)
    g([("orchard-src","Source detail"),("orchard-up","Same area, 3× Vision upscale")],81)
    p("The acceptance test came from the user: “we want increased detail not just magnification of what exists”. A plausible new branch in an orchard can serve the picture. A new eye, a reversed foot or a second child changes what the picture depicts.")
    p("<b>Where the work ended.</b> Large padding ran on Colab. Some prompts gained the intended detail. On one traced face, two tiles placed one eye at different heights on the first denoising step. Manually resolving the disagreement early preserved one eye. The automatic rule we tried failed on both the portrait and the fox family.")
    refs(("Code and evidence",""),("Full-size orchard","evidence/matrix/full-size/13-orchard-harvest.jpg"))

    t("Design brief","What the upscaler owes the artist",
      "The image, the creative prompt and the tile boundary each answer a different question.")
    s("Preserve the event")
    p("Use everyday high-value subjects people actually enlarge: two hands passing flowers, dancers supporting each other, a family of foxes, or readers under a stair. Preserve who touches what, how many subjects exist, which way a foot points, what passes in front, and where straight lines continue. Check the actual source image even when the creative prompt describes a different count.")
    s("Let the generative model earn the extra pixels")
    p("Keep the scene legible at high denoise while allowing new texture and coherent background structure. The user explicitly liked the possibility of fine distant forms at 512 px padding. Empty water should read as water. A wingtip may benefit from the overall brief to distinguish it from an umbrella canopy. That identification gives the tile no reason to draw the rest of the subject.")
    s("Give each tile the right evidence")
    p("The user's rule was “tiles can only have prompt text associated with what's INSIDE the tile but can have the total prompt global context”. Use the original prompt for interpretation. Keep the text sent to the denoiser grounded in the local view. Project the padded processing area back to the clean source for Vision. A caption about the core and visual context from the padding can coexist.")
    s("Requested product behavior")
    p("Start in Vision with the workflow’s Qwen3-VL encoder. Let the user add Canny or Depth. Keep Tile control out of scope until usable weights are available. Test many distinct prompts, especially interactions, and examine actual 3× renders on a GPU. The user accepted 256 px as practical and rejected the sequential waterfront result.")
    note("Reading order: pp. 3–6 cover the design and scene tests. Pages 7–11 cover local text. Pages 12–16 trace the eye, ownership tests and the open design question. Replay and exact prompts follow.")
    refs(("Original waterfront prompt","inputs/padding/waterfront/original-prompt.txt"),("16 scene briefs","experiments/test-harness/scenes.json"))

    t("Geometry and conditioning","What a tile sees and writes",
      "One noisy canvas supplies the working pixels. A separate clean source crop supplies Vision.")
    p("<b>1. Lay out the target.</b> For 1024 × 1024 → 3072 × 3072, nine cores are 1024 px square. At 256 px padding per side, the middle tile reads 1536 × 1536 output pixels. Adjacent tiles share a 512 px strip, equivalent to 64 latent pixels. At 512 padding, the middle window grows to 2048 square. Canvas edges clip the outer windows.")
    p("<b>2. Match the source slices.</b> Project each padded window to source coordinates. Floor the first edge and ceil the last so edge pixels survive. The Krea Vision encoder receives that clean padded crop, area-resized if its long side exceeds 448 px. For the matrix captioner, Qwen also saw the original brief, a marked overview, clean padded crop and clean core. The drawings on the overview were for caption generation only.")
    p("<b>3. Sample one step.</b> All nine calls read slices of the same current noisy latent. Each call receives its own visual tokens and text. The tiler adds the nine model predictions in canvas coordinates, multiplies by tapered cosine weights, and normalizes the sum. The sampler updates the shared canvas once. The next step reads the result of that update.")
    p("<b>Do neighboring tiles see each other?</b> Yes, through the updated shared latent at the next step. During a step, tile 4 cannot see tile 1’s newly predicted eye before making its own proposal. Finished RGB tiles never enter this loop. There is no rule that counts eyes or selects a single writer for each feature.")
    p("<b>Get the output dimensions right.</b> The matrix runner enlarged to the exact requested dimensions, padded at the right/bottom to a multiple of 32, then trimmed. Florist went from 720 × 1280 to 2160 × 3840 on a 2176-wide working canvas. The same arithmetic gives a 4080-wide result from a 1360-wide input.")
    note("The pinned production Vision branch takes vision_prompt for positive text. The experiment supplied the original brief explicitly. Tile-local positions were used in the main matrix. A training-range explanation for canvas positions remains speculative.")
    refs(("Pinned tiler",f"{BASE}/blob/{ST}/usdu_pose_tiled/krea2_vision.py"),("Experiment geometry","experiments/matrix-as-executed/krea16_test_node.py"))

    t("Early tests • Waterfront","How much padding? Which order?",
      "One woman, one burgundy umbrella, open water on the left and a diagonal railing.")
    g([("water-src","Source"),("water-shared","Same global text on all nine"),("water-ground","Local captions, 256 px")],105)
    p("At 256 px and tile-local positions, repeating the scene prompt still produced extra yellow coats and umbrellas. Grounded descriptions removed the conspicuous repeats in this scene. Switching coordinates alone left them visible. The 96 px shared-text/canvas arm changed multiple settings and cannot isolate the padding effect.")
    g([("water-384","384 px padding"),("water-512","512 px padding"),("water-seq","Sequential, 256 px")],102)
    tab([["Padding / side","Largest window","Render time","Peak allocated"],
         ["256 px","1536 × 1536","344 s","14,003 MiB"],["384 px","1792 × 1792","433 s","14,457 MiB"],["512 px","2048 × 2048","544 s","14,979 MiB"]],[.24,.30,.21,.25])
    p("The sequential test gave later tiles access to finished neighbor pixels. Each tile ran all eight steps, then committed its core. The run started with tiles 6, 5, 3. Hard rectangular changes and broken umbrella sections remained. This was a fair test of the proposed advantage on one scene. The older linear redraw used a different frontier mask.")
    note("An earlier proposal would have anchored coarse source frequencies on every step. The user had already seen composition hold at 0.8 in Turbo. The recorded starting sigma of about 0.927 describes this schedule. It cannot measure how much semantic composition survives. The proposed low-frequency anchor and 0.5 reference are outside this comparison.")
    refs(("Waterfront measurements","inputs/padding/waterfront/report.json"),("Sequential experiment","experiments/padding/test_sequential.py"))

    t("Colab matrix • 16 prompts","Where the images held up",
      "Each scene used the same source and upscale seed across Euler/simple, Euler/SGM Uniform and DPM++ 2M/simple.")
    g([("scene-04","Partner rescue"),("scene-13","Orchard lift"),("scene-15","Tailoring")],88)
    tab([["Source scene","What inspection found across the three outputs"],
         ["Florist handoff","Two people still pass flowers. All three reverse the florist’s front sneaker."],
         ["Jazz trio","Three musicians and instruments remain distinct. The drummer’s gaze changes."],
         ["Library","Two readers and the stair survive. A new diagonal handrail appears at the landing."],
         ["Kayak rescue","Both boats and the shared grip hold. Ripples, life jackets and treeline gain detail."],
         ["Pottery class","The bowl and people hold. Two cyan outlines appear where captions mention a locator."],
         ["Restaurant pass","Three staff and two visible plates remain. A cyan outline appears behind the pass."],
         ["Tango poster","Joined hands survive. Broad floor graphics become parquet. A cyan outline appears."],
         ["Night market","The central bag handoff holds. Wet paving gains detail. People deeper in the lane change."]],[.25,.75],size=9.45)
    p("The three florist results were three sampling configurations, all using Vision. The backwards foot did not survive three separate tests of Canny, Depth and Vision. A later caption-only intervention changed that particular foot at the same seed (p. 11).")
    note("All 48 matrix upscales ran on an NVIDIA L4. Eight follow-ups in the original matrix review tested beta schedules, manual caption edits or structural guides on selected scenes. Later caption and portrait studies have their own denominators.")
    refs(("48-run illustrated review","evidence/reports/matrix.pdf"),("Full-resolution Euler gallery","evidence/matrix/full-size/"))

    t("Colab matrix • 16 prompts","What guides corrected",
      "Depth and Canny supply different evidence. Their value depended on the scene and the feature.")
    tab([["Source scene","What inspection found in the Vision matrix"],
         ["Ballet rehearsal","Both dancers remain. Two samplers add a stray shoe. SGM adds a bracket/line."],
         ["Wildlife family","All three add a kit face. SGM also adds an isolated muzzle."],
         ["Coffee product","Stream and crema improve. Extra crockery appears. Cup handle and pot lid change."],
         ["Urban basketball","SGM produces a second ball. Euler/simple and DPM++ keep one."],
         ["Orchard harvest","Three people, one crate and the shared lift hold while apples, foliage and slats gain detail."],
         ["Veterinary exam","Two adults and dog remain. A second round silver fitting appears on the scrubs."],
         ["Tailoring studio","Two people and one tape survive. Weave and checks improve. Tape shifts slightly."],
         ["Museum diorama","A second foreground child appears. Glass corner and rover detail survive."]],[.25,.75],size=9.15,compact=True)
    g([("10-src","Source · two kits"),("10-canny","Vision + Canny"),("10-depth","Vision + Depth")],64)
    p("The optional guides used the same padded boxes, captions and seed. On the fox, both restored two kits. Canny stayed closer to the curled right kit. Depth turned it toward the camera. At the library, both removed the extra rail. Canny followed the balusters more closely. Depth added richer exterior detail.")
    p("Canny supplies a 2D edge map. Depth supplies relative surface distance and can help separate planes or silhouettes. Neither tells the sampler which way a shoe ought to point or how many eyes belong to a face. In the earlier portrait guide test, Canny produced one eye with a distorted brow. Depth retained two.")
    note("An earlier battery compared Vision, Vision + Depth and Vision + Canny on station, portrait, café, hikers, illustration and wedding prompts. In the 16-scene matrix the matched guide follow-ups cover library and fox. The guide strength in those follow-ups was 0.7.")
    refs(("Per-scene observations","evidence/matrix/review-observations.json"),("Guide implementation","experiments/test-harness/krea16_control_guides.py"))

    t("Caption audit • 144 cores","How accurate was each tile?",
      "Nine unpadded source cores per scene. Tiles 1–3 are the top row, 4–6 the middle and 7–9 the bottom.")
    with (ROOT/"evidence/matrix/tile-audit.tsv").open() as file:
        grades = list(csv.DictReader(file, delimiter="\t"))
    rows=[["Scene",*map(str,range(1,10))]]
    for case in sorted({x["scene"] for x in grades}):
        rows.append([case[3:].replace("-"," ").title()]+[x["grade"][0] for x in grades if x["scene"]==case])
    tab(rows,[.37]+[.07]*9,size=8.5,compact=True)
    p("<b>A · Accurate, 64/144.</b> The main visible content is correctly described. Small omissions are allowed. <b>P · Partial, 65/144.</b> A material omission, wrong relation or locator reference remains. <b>W · Wrong, 15/144.</b> The main description misses or contradicts the crop.")
    p("The audit flagged 37 descriptions for material outside the core and 11 for mentioning the cyan locator. Flags overlap. Florist tile 7 earned A for the main visible content, yet its caption did not guarantee the denoiser would preserve the shoe’s direction. Caption grades and visual success measure different things.")
    note("The linked original audit shows the exact crop, wording and reviewer’s reason for every cell. The TSV is a searchable transcription with the original PDF page number. These are visual judgments on synthetic scenes, without independent inter-rater calibration.")
    refs(("144 illustrated judgments","evidence/reports/tile-audit.pdf"),("Machine-readable audit","evidence/matrix/tile-audit.tsv"))

    t("Caption experiments","What changing the method bought",
      "The workflow’s Qwen3-VL-4B generated the captions. The shots and scene brief were already available locally.")
    tab([["Method","Evidence supplied to Qwen"],
         ["Original matrix","Marked whole scene, original creative prompt, clean padded neighborhood, clean core. Describe only the core."],
         ["Clean core","Only the clean core and a request for literal fragments, contacts and directions."],
         ["Context then check","First use brief and padded neighborhood to propose nouns. Then keep only proposals visible in the clean core."],
         ["Two halves","Read each half along the core’s longer axis. Check both field notes against the whole clean core. This variant omitted the original brief."],
         ["Check both proposals","Generate original and two-half descriptions. Inspect the clean core again, resolving disagreements. Proposal order alternated across scenes."]],[.28,.72],size=9.8)
    tab([["Graded tile set","Original A","Two-half A","Both proposals A"],
         ["18 targeted hard tiles","1/18","10/18","—"],["30 held-out tiles, 10 scenes","11/30","15/30","—"],
         ["16 fresh tiles, 1 per familiar scene","5/16","6/16","8/16"]],[.43,.18,.18,.21],size=9.5)
    p("On the hard 18, clean-core alone reached 7 accurate, and context-then-check reached 8. The large early jump followed deliberate selection of difficult cases. On the 30 held-out tiles, two-half descriptions recovered 14 preselected landmarks versus 6 originally, while nine captions gained an A and five lost it.")
    note("“Fresh” means unused tile coordinates from the same 16 scenes. One tile per scene gives breadth, with little power to rank methods within a scene. Splitting changed both the views and the wording/brief, so the independent effect of a magnified half remains unknown.")
    refs(("Methods and all splits","evidence/reports/caption-methods.pdf"),("Prompt implementation","experiments/caption-methods/caption_methods.py"))

    t("Caption experiments","Why 5 → 6 → 8 on the fresh 16",
      "Counting paired wins and losses explains the small lift better than the three totals.")
    g([("jazz-core","Jazz tile 2 · inspect the two lower edges")],115)
    p("The original missed the hand at lower left and curved brass at lower right. Two halves found both, then called a line a metal rod on the saxophone. The new error kept the grade at partial. Checking both descriptions against the whole core retained the two fragments and dropped the rod. That earned an A.")
    tab([["Fresh-set outcome","Accurate","Partial","Wrong","Landmark"],
         ["Original","5","10","1","1"],["Two halves","6","10","0","3"],["Both proposals rechecked","8","8","0","2"]],[.4,.15,.15,.14,.16])
    p("Restaurant, ballet and museum gained an A with halves. Coffee and veterinary lost one: the former confused the handle and nearby metal, the latter misread an ear and overlapping fur. Tango improved from wrong to partial. Three gains minus two losses gives the one net gain. The proposal check retained the six A grades and added jazz and tailoring.")
    p("A landmark score asks if a specific preselected detail was found. An A asks whether the whole description is grounded. The 8/16 method found fewer of those exact landmarks than the 6/16 method. A second look can correct a noun or drop a valid fragment. Only a matched render says whether that wording changes the picture.")
    refs(("Exact 48 captions and grades","evidence/caption-methods/fresh_tiles-scores.tsv"),("Two-proposal check","experiments/caption-methods/duel.py"))

    t("The user’s boundary proposal","Show Qwen more, mark what counts",
      "Four presentations held the target pixels and original brief fixed. Each asked for only what lay inside.")
    g([("cue-padded_box","Padded + border"),("cue-full_box","Full + border"),("cue-full_spotlight","Full + dim surround")],105)
    tab([["16 fresh tiles","A / P / W","Landmark","Outside"],
         ["Padded view + exterior border","6 / 9 / 1","2","8"],["Full scene + exterior border","4 / 10 / 2","0","5"],
         ["Full scene + numeric coordinates","1 / 12 / 3","0","11"],["Full scene + dim surround","4 / 12 / 0","0","8"]],[.48,.22,.15,.15])
    p("The border lay outside the target. The spotlight dimmed the surrounding pixels to 30%. The full view covered about 5.2 times the area of the padded view on this comparison. More context sometimes clarified identity, while small edge fragments became easier to miss. On four extra stress tiles, padded-border scored 1/4 accurate, the full-scene variants 0/4, and the earlier two-half method 2/4.")
    g([("10-src","Source · two kits"),("10-box2","New text on tiles 7, 8"),("10-box9","New text on all nine")],60)
    p("At fixed seed 40310871, full-scene border descriptions on fox tiles 7 and 8 yielded three visible kit faces. Replacing all nine produced five apparent young faces and overlapping adult heads. A border around the target is a sensible input format. These runs show why it needs a test of the text and a test of the rendered result.")
    refs(("Boundary grades and matched renders","evidence/reports/boundary-study.pdf"),("Exact cue generation","experiments/boundary-study/boundary.py"))

    t("Controlled text edits","When changing words changes pixels",
      "Identical sources, seeds, padded Vision inputs and sampler. Only the specified tile text changed.")
    g([("01-src","Source sneaker"),("01-orig","Original captions"),("01-fixed","Only tile 7 changed"),("01-all","All nine changed")],96)
    p("The florist’s front toe points right in the source and left in all three original 3× results. Replacing the caption on tile 7 restored the direction and the rear shoe at this seed. The replacement named the shoes, ledge and pavement without naming a toe direction. Changing all nine still fixed the toe, but the bike wheel shifted onto it.")
    g([("10-orig","Original · extra kit"),("10-fixed","Two-half text · two kits")],83)
    g([("10-boot","Edge probe · black boot"),("10-guard","Sentence removed · extra face returns")],83)
    p("The fox improved after replacing just two captions with the two-half method. Earlier manual edits to three captions had produced four kit faces. A later edge probe introduced a human-foot sentence and a black boot appeared. Removing the sentence removed the boot, while a third face returned. In pottery, deleting the cyan-rectangle wording on two tiles removed both drawn rectangles.")
    note("These changes establish that local text can cause or repair particular artifacts. The portrait on the next page exposes another route to duplication: the two captions in its failing run never contained the disputed noun.")
    refs(("Controlled edits and image findings","evidence/caption-methods/render-findings.json"),("Full florist tile audit","evidence/caption-methods/full-florist-audit.tsv"))

    t("Portrait • The two tiles","Did the upper-left tile say “eye”?",
      "No. Neither of the two disputed captions used the word. Both tiles received source pixels containing that eye.")
    g([("portrait-map","Whole source · grid added for review"),("pad-1","Tile 1 · padded source"),("pad-4","Tile 4 · padded source")],116)
    p("The source contains two women. The duplicate is on the nearer, dark-haired woman’s profile. The cyan horizontal line marks the core boundary at output row 1024, about source row 341.3. Tile 1’s core ends near the brow/upper eyelid crease. The iris falls in tile 4’s core. The padded source boxes are [0, 0, 427, 427] and [0, 256, 427, 768] in left, top, right, bottom order. The cyan lines on these illustrations were added after the run.")
    portrait=read("inputs/padding/portrait/report.json")
    for ix in (0,3):
        row=portrait["captions"]["256"][ix]
        s("Exact failed-run caption · tile "+str(row["tile"]))
        p("“"+escape(row["caption"])+"”")
    g([("eye-src","Source eye"),("eye-blend","Original padded captions"),("eye-corecap","Core-targeted captions")],81)
    note("The separate core-caption run mentioned eyes in both descriptions and rendered one near-side eye. A rule that suppresses the word “eye” in one caption would miss the mechanism in this case. Numbering is 1-based and row-major throughout this report.")
    refs(("All exact portrait captions","inputs/padding/portrait/report.json"),("Isolated tile probe","experiments/padding/test_overlap_owner.py"))

    t("Portrait • Matched predictions","Two placements on the first step",
      "Same 1024-square source and same full-canvas noise, seed 40309561. Crops below use matching canvas coordinates.")
    g([("eye-iso1","Tile 1 alone · final"),("eye-iso4","Tile 4 alone · final")],108)
    p("When sampled to completion independently, each padded tile produced a single prominent near-side eye. Tile 1 put it higher. That experiment shows two local solutions were available. It does not show what either tile proposed during the real shared-canvas run. The trace below measures precisely that.")
    g([("step-late-1-1","Step 1 · tile 1"),("step-late-1-4","Step 1 · tile 4"),("step-late-1-merged","Step 1 · shared estimate")],92)
    p("On the actual common input at step 1, tile 1 already proposed the high eye and tile 4 the lower eye. Both appeared in the merged clean estimate. The final pupils were about 157 output pixels apart, near rows 905 and 1062. The ordinary taper gave tile 1 about 83% weight at the high site and tile 4 about 61% at the lower site. Both sites lie in the overlap, rows 768–1280.")
    p("That answers the user’s objection about an eye already being there. Each tile proposed one plausible eye locally. At inference the sampler did not evaluate an anatomical loss on the full face. Two locally dominant placements could both survive the sum, and both tiles then saw the evolving result.")
    note("The trace converts each tile’s actual flow prediction into a clean latent estimate using the pinned ComfyUI conversion. The merged estimate matched the native x0 callback with max absolute error 0 on every recorded step. Preview windows were decoded separately; exact crops near their borders can differ from a full-canvas decode.")
    refs(("All step PNGs and logs","evidence/portrait-coordination/"),("Trace implementation","experiments/portrait-coordination/trace_portrait.py"))

    t("Portrait • Timing intervention","Which tile gets the first say?",
      "A positive control gave tile 4 sole influence around the disputed eye for four of eight steps.")
    g([("eye-src","Source"),("eye-early","Tile 4 owns steps 1–4"),("eye-late","Tile 4 owns steps 5–8")],97)
    p("The intervention covered both proposed eye sites across the boundary, in the manually marked source box [0, 269, 265, 424], feathered at its edges. All nine tiles still read the usual padded shared latent. Outside that box, the usual blend continued. The original images, noise, captions, positions and sampler stayed fixed.")
    g([("step-early-1-merged","Early owner · step 1 merge"),("step-late-1-merged","Late owner · step 1 merge")],80)
    g([("step-early-3-1","Early owner · upper tile at step 3"),("step-late-5-4","Late owner · middle tile at step 5")],80)
    p("The individual step-1 tile previews were pixel-identical across the two timing runs. Giving tile 4 the early write changed the merged input to subsequent steps. By step 3 the upper tile moved its eye lower. Normal blending resumed at step 5 and the single eye survived, with a small bright bead near the outer lid.")
    p("With the late intervention, the usual first-step merge already carried two placements. By step 5 the middle tile itself proposed both eyes. Giving it the entire write then preserved the duplicate. These two windows show sufficiency of the first four on this seed. The shortest effective window is unmeasured.")
    refs(("Complete timing comparison","evidence/reports/portrait-trace.pdf"),("Exact step records","evidence/portrait-coordination/owner_last_four/steps.json"))

    t("Portrait and fox • Failed generalization","Why “only its tile can draw it” failed",
      "Owning a rectangular core and owning a feature that crosses that core boundary are different operations.")
    g([("eye-blend","Ordinary blend"),("eye-corewrite","Only cores may write"),("eye-manual","Hand-selected eye region")],91)
    p("<b>Read padding, write only the core.</b> This test enforced a single writer per pixel for all eight steps. The duplicate remained. The upper tile placed its high pupil inside its own writable core. The lower tile placed its pupil in its core. Each respected the rule. The hand-marked region succeeded because it covered both candidate locations and assigned the same decision to both.")
    g([("eye-auto","Automatic match · eye still doubled"),("10-auto","Automatic match · extra kit face")],78)
    p("<b>What the automatic pilot actually did.</b> At step 1 it compared each tile’s estimated clean latent with the source latent. It smoothed the difference with a 7 × 7 average, squared it, and averaged over each shared rectangle. A tile more than 10% ahead of the runner-up got the early write there with a four-latent-pixel feather. Close scores retained the ordinary blend.")
    p("At the eye, upper error was 0.329815 and middle error 0.311534. The middle was about 5.5% lower. The rule kept blending, and the eye stayed doubled. On foxes an extra face moved above the right kit. The 10% cutoff was an investigator choice. No successful automatic selector emerged. The orchard and museum qualification runs were discontinued.")
    note("The pilot is archived as a withdrawn research patch. A whole-overlap average can favor matching broad tones even when the feature of interest occupies very few pixels. That is a possible reason for poor discrimination. The measurements establish the failure but do not identify a universal replacement.")
    refs(("Pilot measurements and images","evidence/reports/ownership-pilot.pdf"),("Withdrawn code and patch","experiments/source-overlap/"))

    t("Handoff decision","What the next agent needs to establish",
      "The open question is how to agree on the location of an existing feature early while leaving room to invent fine detail.")
    s("What survives the evidence")
    p("Keep Vision’s clean padded crop aligned with the working tile. Preserve the distinction between the original brief as context and local text as a drawing instruction. Retain the shared-step baseline, 256 px reference and optional guides for matched comparisons. The local-caption and first-step tracing results address distinct mechanisms. Some scenes show both.")
    s("A simple hypothesis worth testing")
    p("Give all tiles one source-aligned early estimate of where resolvable structures are, then let their padded windows refine that shared estimate at high resolution. An implementation could test a single early whole-scene pass at a tractable resolution. This is an untested design hypothesis. It earns its keep only if it restores location and count while still generating useful high-resolution detail.")
    s("A comparison that can distinguish explanations")
    p("At fixed source, global noise and captions, compare ordinary blend, core-write-only, manual early control, and the candidate. Start without a guide. Translate the tile grid relative to the same source, retaining the same canvas-noise coordinates. Track whether the erroneous pupil follows the grid. Record the actual per-tile step-1, step-3 and step-5 predictions and the merged estimates.")
    p("Lock the candidate before trying fresh seeds and unseen prompts. Include the difficult portrait, foxes, florist and museum, plus cases already working such as the orchard lift and kayak grip. Add Canny and Depth in separate arms only after the base comparison. Keep the same image/text conditioning and report each loss as well as each repair.")
    s("What to look at on a phone and at 100%")
    p("Pair the source and output at the same coordinates. Record count, pupil center or toe direction, contact at handoffs, occlusions and seams. Inspect texture crops and whole-frame background continuity. Review without labels when feasible. Pixel error alone rewards simple enlargement and can conceal a small extra face. Existing runs used single seeds per scene and visual judgments.")
    refs(("Source images and run settings","inputs/"),("Original sampler implementation",f"{BASE}/blob/{ST}/usdu_pose_tiled/krea2_vision.py"))

    t("Reproduction • Colab","Replay the portrait first",
      "The code branch includes the original input PNGs, generated text, source hashes, exact tiles, model pins and timing traces.")
    s("Runtime and fixed conditions")
    p("Use a Colab GPU runtime with an NVIDIA L4 for the matched portrait. Archived trace: Torch 2.14.1+cu130, eight Euler/simple steps, CFG 1, denoise 0.8, 3 × 3 cores, 256 output-pixel padding, tile-local positions and seed 40309561. The source was encoded after Lanczos enlargement. The matched runs preserve the original nine padded captions.")
    p("The setup fetches Krea 2 Turbo FP8, Qwen3-VL-4B FP8 and Qwen Image VAE from the same pinned model revision. Captioning uses the existing encoder. Optional Canny and Depth tests additionally load the specific guide weights in setup_controls.sh.")
    s("Commands for a fresh runtime")
    note("Run the shell block in the repository README. It clones the research branch, stages both historical /content layouts and installs the pinned model code and weights. The two portrait entry points are:")
    p("<font name='Courier'>cd /content/st-krea-3x-padding<br/>python -u test_overlap_owner.py<br/>python -u trace_portrait.py</font>",size=10.4)
    p("The first produces isolated tiles 1 and 4 plus the all-eight-step manual control. The second writes both full-size timing renders, all 48 per-step PNGs and two eight-row logs under results/portrait. Use the linked README for the 48-run matrix, saved-caption comparisons, boundary experiment and isolated replay of the failed patch.")
    s("Pins and evidence")
    note("Sweet Tea Nodes · "+ST+"<br/>ComfyUI · "+COMFY+"<br/>Comfy-Org/Krea-2 · 588feaf955b08b9ac4723eeb751752ab045dba71")
    note("This document uses retained Colab outputs. Replay layout and records were verified offline during packaging. No new GPU rerun was used to claim agreement across different Colab software stacks.")
    refs(("Replay instructions","README.md"),("Pinned setup","experiments/test-harness/setup_runtime.py"),("Six original reports","evidence/reports/"))

    t("Design history","Other explanations on the table",
      "These were part of the original discussion. Their evidence is narrower than the measured first-step disagreement.")
    s("Position and tile size")
    p("The original proposal suspected that moving each tile’s image-token positions to full-canvas coordinates could change how outer tiles read their text and visual conditioning. The waterfront includes both local and canvas positions at 256 px with shared or grounded text. Repetition with shared text persisted even using local positions. The training resolution range was unknown. The portrait trace used local positions, so canvas offsets are unnecessary for that specific failure.")
    p("The user's counterexample also matters: Krea can often handle a roughly two-megapixel view. A 1536-square middle window is 2.36 MP. The portrait’s upper and middle left windows were 1280-square and 1280 × 1536. Each isolated view produced one prominent eye. The actual shared step showed how two placements reached the canvas. Other scenes still need single-window and shifted-grid controls.")
    s("Padding, resize and decode")
    p("The initial 96 px illustration would give 1216-square interior tiles on a 3072-square target and 192 px of overlap. The later trials raised that to 256, 384 and 512. Broad ghosting at a join, ringing on an existing edge, and tone lines on a separate VAE grid were distinct proposed diagnostics. Whole-tile, alternative resize and untiled-decode controls would address those respectively. They were never established as the reason for the duplicate eye.")
    p("The original geometry discussion also considered an eight-tile layout for a 1360 × 768 source using Auto tile count. The controlled matrix used nine cores throughout. The square-grid and pad-then-trim code are archived so an agent can test a different layout without confusing it with padding or stretching.")
    s("Early structure without freezing detail")
    p("ILVR-style low-frequency replacement, a DemoFusion-style global pass and shifting an unblended grid between steps were early suggestions. These alternatives did not produce the results in this report. The early-owner intervention tests a timing question inside one region. It supplies a useful positive control for any future layout-preserving approach.")
    refs(("Waterfront experiment arms","inputs/padding/waterfront/report.json"),("Production grid selection",f"{BASE}/blob/{ST}/usdu_pose_tiled/krea2_control.py"))

    t("Image comparisons • Guides and text","Read the guide results side by side",
      "Each row shows the same source coordinates across its interventions. These are crops of saved renders.")
    g([("eye-src","Profile · source"),("eye-blend","Vision"),("eye-canny","Vision + Canny"),("eye-depth","Vision + Depth")],89)
    p("The original eye is low in the frame. Canny keeps one main eye but gives the brow a new shape and leaves a bright dot. Depth draws two eye rows. Both retain recognizable facial volume. Volume alone was insufficient to establish a single eye placement on this seed.")
    g([("03-src","Library · source"),("03-orig","Vision"),("03-canny","Vision + Canny"),("03-depth","Vision + Depth")],76)
    p("Follow the diagonal posts to the lower landing. Extra metal lines in the baseline change the construction of the stair. Both guided runs recover the main rail arrangement here. They add different baluster and window details.")
    g([("05-src","Pottery · source"),("05-orig","Original local text"),("05-fixed","Two caption edits")],126)
    p("The shelf behind the potter acquires cyan outlines and a square object. Those marks were absent from the visual input used by the denoiser. At the same seed, removing the locator wording in two local captions removes the outlines. This gives the next agent a case with a demonstrably different cause from the first-step eye disagreement.")
    refs(("Guide records","evidence/matrix/guide-examples/"),("Caption edit records","evidence/matrix/caption-example/"),("Aligned crop provenance","evidence/figures/manifest.json"))

    scenes=read("experiments/test-harness/scenes.json")
    for i in range(0,16,4):
        t("Appendix • Original creative prompts","The prompt set, "+str(i+1)+"–"+str(i+4),
          "Verbatim from the 16-scene test harness. The source image is the reference when it differs from the prompt.")
        for scene in scenes[i:i+4]:
            s(scene["id"][3:].replace("-"," ").title()+f'  ·  {scene["width"]} × {scene["height"]}')
            p(escape(scene["prompt"]),size=10.2,leading=14.4,after=12)
        refs(("Exact prompt and seed files","inputs/matrix/"),("Scene definitions and focus areas","experiments/test-harness/scenes.json"))

    t("Archive map","Find the evidence behind a claim",
      "The links below point to the code snapshot used to assemble this report. The illustrated comparisons and captions are also readable offline in the repository download.")
    for title,file,desc in [
        ("48 sampler comparisons and eight follow-ups","matrix.pdf","Per-scene sources and results, sampler choices, guide trials and caption edits."),
        ("144 original tile-caption grades","tile-audit.pdf","Tap any grid cell for the source core, full local caption and individual judgment."),
        ("Alternative caption methods","caption-methods.pdf","Development, held-out and fresh comparisons, edge probes and controlled image edits."),
        ("Boundary prompting","boundary-study.pdf","The four marked-view methods, blind grades and fox renders."),
        ("Portrait timing trace","portrait-trace.pdf","Aligned source, tile proposals, merged predictions and timing windows."),
        ("Automatic ownership pilot","ownership-pilot.pdf","The rule, output comparisons, measured decisions and reason it was withdrawn.")]:
        link=r.addr("evidence/reports/"+file)
        p(f"<a href='{link}' color='#005e60'><b>{title}</b></a><br/>{desc}",size=10.5,leading=14)
    s("What is preserved")
    p("The repository holds research runners, untouched source PNGs, exact prompts, tile captions and grades. It also holds lossless timing windows, native-dimension JPEGs for all 16 Euler/simple outputs, figure provenance, seeds and model revisions. The page images are registered review crops. Use the saved PNG sources and replay for pixel-level output comparisons.")
    p("The research branch contains the experimental Vision + guide node and the failed owner as research code. The requested product default and a generalized feature-coordination fix remain open. The evidence is ready to evaluate independently.")
    refs(("Code snapshot",""),("Figure provenance","evidence/figures/manifest.json"),("Replay","README.md"))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision",required=True)
    ap.add_argument("--output",type=Path,default=ROOT/"Krea2-upscaling-handoff.pdf")
    args=ap.parse_args()
    report=Report(args.output,args.revision)
    write_report(report)
    report.finish()
    print(f"Wrote {len(report.pages)} pages to {args.output}")

