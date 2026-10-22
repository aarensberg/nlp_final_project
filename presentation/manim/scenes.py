"""Animations Manim de la soutenance.

Chaque chiffre affiché vient de presentation/key_numbers.md (généré depuis reports/)
ou de presentation/demo_outputs/ (sorties réelles enregistrées). Aucun calcul de modèle ici.

Rendu (voir presentation/manim/README.md) :
    uv run --python 3.12 --with manim manim -r 1920,1080 --fps 30 scenes.py <Scene>
"""

from manim import *

BG = "#0f172a"
PANEL = "#1e293b"
EDGE = "#475569"
FG = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#60a5fa"     # combinaison
ORANGE = "#fb923c"   # Legal-BERT
GREEN = "#34d399"    # TF-IDF / correct
AMBER = "#fbbf24"    # abstention / juriste
RED = "#f87171"      # erreur / limite
FONT = "Helvetica Neue"

config.background_color = BG


def T(s, size=24, color=FG, weight=NORMAL):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def M(s, size=24, color=FG):
    return MarkupText(s, font=FONT, font_size=size, color=color)


def fr(x, d=2):
    """Nombre au format français."""
    return f"{x:.{d}f}".replace(".", ",")


def title(s):
    return T(s, 36, weight=BOLD).to_edge(UP, buff=0.45)


def footer(s, color=MUTED):
    return T(s, 21, color).to_edge(DOWN, buff=0.35)


def box(head, sub=None, w=3.0, h=1.0, color=EDGE, size=22, sub_size=17, head_color=FG):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color,
                         stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
    parts = [T(head, size, head_color, weight=BOLD)]
    if sub:
        parts.append(VGroup(*[T(l, sub_size, MUTED) for l in sub.split("\n")]).arrange(DOWN, buff=0.06))
    g = VGroup(*parts).arrange(DOWN, buff=0.1)
    if g.width > w - 0.25:
        g.scale_to_fit_width(w - 0.25)
    if g.height > h - 0.15:
        g.scale_to_fit_height(h - 0.15)
    g.move_to(r)
    return VGroup(r, g)


def arrow(a, b, color=MUTED, buff=0.12):
    return Arrow(a, b, buff=buff, color=color, stroke_width=3,
                 max_tip_length_to_length_ratio=0.25, tip_length=0.18)


# ---------------------------------------------------------------------------
class S02BusinessNeed(Scene):
    def construct(self):
        ttl = title("Due diligence : chaque clause est lue, typée, comparée")
        cards = VGroup()
        for i in range(4):
            c = RoundedRectangle(corner_radius=0.08, width=2.1, height=2.7, stroke_color=EDGE,
                                 fill_color=PANEL, fill_opacity=1, stroke_width=2)
            lines = VGroup(*[
                Line(ORIGIN, RIGHT * (1.5 - 0.35 * ((j * 3 + i) % 3)), stroke_color=MUTED, stroke_width=2)
                for j in range(7)
            ]).arrange(DOWN, buff=0.24, aligned_edge=LEFT).move_to(c)
            cards.add(VGroup(c, lines).shift(i * 0.15 * (UP + RIGHT)))
        cards.move_to(LEFT * 5.0 + DOWN * 0.2)
        lbl = T("des centaines de clauses\npar contrat", 18, MUTED).next_to(cards, DOWN, buff=0.3)

        core = box("Cœur NLP", "prototype de recherche", w=2.8, h=1.3, color=BLUE, size=28)
        core.move_to(LEFT * 1.4 + DOWN * 0.2)
        outs = VGroup(
            box("1 · Classer", "un type de clause parmi 100", w=4.4, h=1.2, color=BLUE),
            box("2 · Retrouver", "des précédents dans la banque de clauses", w=4.4, h=1.2, color=GREEN),
            box("3 · Signaler", "« à revoir par un juriste » quand il ne sait pas", w=4.4, h=1.2, color=AMBER),
        ).arrange(DOWN, buff=0.35).move_to(RIGHT * 4.25 + DOWN * 0.2)
        a1 = arrow(cards.get_right(), core.get_left())
        arrs = VGroup(*[arrow(core.get_right(), o.get_left()) for o in outs])
        foot = footer("On n'automatise pas le juriste : on trie, on retrouve, on signale.", FG)

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.2), FadeIn(lbl))
        self.play(GrowArrow(a1), FadeIn(core, scale=0.9))
        for a, o in zip(arrs, outs):
            self.play(GrowArrow(a), FadeIn(o, shift=LEFT * 0.3), run_time=0.8)
            self.wait(0.4)
        self.play(Indicate(outs[2], color=AMBER, scale_factor=1.05))
        self.play(FadeIn(foot, shift=UP * 0.2))
        self.wait(2.5)


# ---------------------------------------------------------------------------
class S03DataSplit(Scene):
    def construct(self):
        ttl = title("LEDGAR : 80 000 clauses de contrats déposés à la SEC")

        y = 0.8
        specs = [
            ("Train", "60 000 clauses", "2016 – 2017", -6.3, -0.3, MUTED, "apprentissage des modèles"),
            ("Validation", "10 000 clauses", "2018", 0.0, 3.0, AMBER, "tous les réglages :\ntempératures, poids, seuils"),
            ("Test", "10 000 clauses", "2019", 3.3, 6.3, GREEN, "mesure uniquement,\njamais de réglage"),
        ]
        blocks, years, roles = VGroup(), VGroup(), VGroup()
        for name, n, yr, x0, x1, col, role in specs:
            r = RoundedRectangle(corner_radius=0.1, width=x1 - x0, height=1.5, stroke_color=col,
                                 stroke_width=3, fill_color=PANEL, fill_opacity=1).move_to([(x0 + x1) / 2, y, 0])
            t = VGroup(T(name, 28, weight=BOLD), T(n, 21, MUTED)).arrange(DOWN, buff=0.12).move_to(r)
            blocks.add(VGroup(r, t))
            years.add(T(yr, 22, col).next_to(r, UP, buff=0.18))
            roles.add(T(role, 19, MUTED).next_to(r, DOWN, buff=0.35))
        timeline = Arrow([-6.6, y - 1.05, 0], [6.6, y - 1.05, 0], buff=0, color=EDGE, stroke_width=2,
                         tip_length=0.15)
        tl = VGroup()

        chips = VGroup(*[
            box(s, None, w=3.0, h=0.7, color=EDGE, size=20)
            for s in ["100 types de clauses", "1 étiquette par clause", "anglais juridique", "droit américain (SEC)"]
        ]).arrange(RIGHT, buff=0.25).move_to(DOWN * 2.45)
        foot = footer("Découpage chronologique : le test mesure déjà un peu de dérive (Jensen-Shannon train/test 0,078)")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(Create(timeline), FadeIn(tl))
        for b, yr, ro in zip(blocks, years, roles):
            self.play(FadeIn(b, shift=RIGHT * 0.3), FadeIn(yr), run_time=0.7)
            self.play(FadeIn(ro), run_time=0.5)
        self.play(Indicate(blocks[2], color=GREEN, scale_factor=1.04))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.2))
        self.play(FadeIn(foot))
        self.wait(2.5)


# ---------------------------------------------------------------------------
class S04DataLimits(Scene):
    def construct(self):
        ttl = title("Trois limites mesurées avant de modéliser")

        # 1. déséquilibre
        h1 = T("1 · Déséquilibre (train)", 26, weight=BOLD).move_to([-3.6, 2.3, 0])
        x0, W = -6.5, 5.6
        gl = Rectangle(width=W, height=0.55, fill_color=BLUE, fill_opacity=0.9, stroke_width=0)
        gl.move_to([x0 + W / 2, 1.2, 0])
        bk = Rectangle(width=max(W * 23 / 3167, 0.05), height=0.55, fill_color=RED, fill_opacity=1,
                       stroke_width=0)
        bk.move_to([x0 + bk.width / 2, 0.05, 0])
        gl_l = T("Governing Laws · 3 167", 21).next_to(gl, UP, buff=0.1).align_to(gl, LEFT)
        bk_l = T("Books · 23", 21).next_to(bk, RIGHT, buff=0.2)
        ratio = T("rapport 138", 26, RED, weight=BOLD).move_to([-3.6, -0.9, 0])
        conseq = T("→ on suit le macro-F1 (chaque type pèse autant)", 20, MUTED).next_to(ratio, DOWN, buff=0.2)

        # 2 & 3. périmètre et types absents
        h2 = T("2 · Périmètre", 26, weight=BOLD).move_to([3.6, 2.3, 0])
        scope = VGroup(
            T("anglais · droit américain", 21),
            T("juridiction hors États-Unis : 253 clauses (train)", 19, MUTED),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(h2, DOWN, buff=0.25)
        h3 = T("3 · Types absents des 100 labels", 26, weight=BOLD).next_to(scope, DOWN, buff=0.45)
        items = ["RGPD / protection des données", "force majeure", "non-concurrence",
                 "limitation de responsabilité", "exclusivité · audit · non-sollicitation"]
        lst = VGroup(*[
            VGroup(T("×", 26, RED, weight=BOLD), T(s, 21)).arrange(RIGHT, buff=0.2) for s in items
        ]).arrange(DOWN, buff=0.13, aligned_edge=LEFT).next_to(h3, DOWN, buff=0.25)
        for g in (scope, lst):
            g.align_to(h3, LEFT)
        h2.align_to(h3, LEFT)
        note = T("le système ne peut pas les reconnaître", 20, RED).next_to(lst, DOWN, buff=0.25).align_to(h3, LEFT)

        foot = footer("Aussi : 282 clauses du test identiques à une clause du train → scores publiés avec et sans elles")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(FadeIn(h1))
        self.play(GrowFromEdge(gl, LEFT), FadeIn(gl_l), run_time=1.2)
        self.play(GrowFromEdge(bk, LEFT), FadeIn(bk_l), run_time=0.8)
        self.play(FadeIn(ratio, scale=1.2), FadeIn(conseq))
        self.wait(0.5)
        self.play(FadeIn(h2), FadeIn(scope, shift=UP * 0.1))
        self.play(FadeIn(h3))
        self.play(LaggedStart(*[FadeIn(i, shift=LEFT * 0.2) for i in lst], lag_ratio=0.25))
        self.play(FadeIn(note))
        self.play(FadeIn(foot))
        self.wait(2.5)


# ---------------------------------------------------------------------------
class S05Architecture(Scene):
    def construct(self):
        ttl = title("Architecture : deux chaînes, une interface commune")

        lab1 = T("CLASSIFICATION", 18, BLUE, weight=BOLD).move_to([-5.75, 2.85, 0])
        clause = box("clause", None, w=1.4, h=0.8, size=22).move_to([-6.2, 1.3, 0])
        bert = box("Legal-BERT fine-tuné", "256 tokens · 2 époques", w=2.6, h=0.95, color=ORANGE).move_to([-3.7, 2.05, 0])
        tfidf = box("TF-IDF + rég. log.", "unigrammes + bigrammes", w=2.6, h=0.95, color=GREEN).move_to([-3.7, 0.55, 0])
        cb = box("softmax(z / 1,04)", "calibration", w=2.1, h=0.95, color=ORANGE, size=20).move_to([-0.6, 2.05, 0])
        ct = box("softmax(z / 0,91)", "calibration", w=2.1, h=0.95, color=GREEN, size=20).move_to([-0.6, 0.55, 0])
        sig = VGroup(Circle(radius=0.38, color=BLUE, fill_color=PANEL, fill_opacity=1, stroke_width=3),
                     T("Σ", 30, BLUE, weight=BOLD)).move_to([1.35, 1.3, 0])
        dec = box("max ≥ 0,709 ?", "seuil (validation)", w=2.3, h=0.95, color=BLUE, size=21).move_to([3.2, 1.3, 0])
        ok = box("type prédit", "+ confiance", w=2.0, h=0.85, color=GREEN, size=21).move_to([5.85, 2.05, 0])
        no = box("à revoir", "par un juriste", w=2.0, h=0.85, color=AMBER, size=21).move_to([5.85, 0.55, 0])

        a = [
            arrow(clause.get_right(), bert.get_left()), arrow(clause.get_right(), tfidf.get_left()),
            arrow(bert.get_right(), cb.get_left()), arrow(tfidf.get_right(), ct.get_left()),
            arrow(cb.get_right(), sig.get_top() + LEFT * 0.1, color=ORANGE),
            arrow(ct.get_right(), sig.get_bottom() + LEFT * 0.1, color=GREEN),
            arrow(sig.get_right(), dec.get_left()),
            arrow(dec.get_right(), ok.get_left(), color=GREEN), arrow(dec.get_right(), no.get_left(), color=AMBER),
        ]
        w_b = T("× 0,2", 20, ORANGE, weight=BOLD).move_to([1.0, 2.2, 0])
        w_t = T("× 0,8", 20, GREEN, weight=BOLD).move_to([1.0, 0.4, 0])

        lab2 = T("RECHERCHE", 18, GREEN, weight=BOLD).move_to([-6.0, -0.8, 0])
        q = box("question", "ou clause", w=1.4, h=0.85, size=21).move_to([-6.2, -2.05, 0])
        flt = box("filtre « N jours »", "optionnel · mention seulement", w=2.6, h=0.95, color=AMBER).move_to([-3.7, -2.05, 0])
        dn = box("dense bge-small", "cosinus", w=2.1, h=0.85, color=BLUE, size=20).move_to([-0.6, -1.45, 0])
        bm = box("BM25", "score > 0", w=2.1, h=0.85, color=GREEN, size=20).move_to([-0.6, -2.65, 0])
        rrf = box("RRF", "k = 60", w=1.6, h=0.95, color=FG).move_to([1.85, -2.05, 0])
        top = box("top-k", "≤ k résultats\n+ avertissement", w=2.4, h=0.95, color=EDGE).move_to([5.3, -2.05, 0])
        b = [
            arrow(q.get_right(), flt.get_left()),
            arrow(flt.get_right(), dn.get_left()), arrow(flt.get_right(), bm.get_left()),
            arrow(dn.get_right(), rrf.get_left()), arrow(bm.get_right(), rrf.get_left()),
            arrow(rrf.get_right(), top.get_left()),
        ]
        foot = footer("Températures, poids et seuils : choisis sur la validation uniquement. Le test ne sert qu'à mesurer.", FG)

        self.play(FadeIn(ttl, shift=DOWN * 0.2), FadeIn(lab1))
        self.play(FadeIn(clause))
        self.play(GrowArrow(a[0]), GrowArrow(a[1]), FadeIn(bert), FadeIn(tfidf))
        self.play(GrowArrow(a[2]), GrowArrow(a[3]), FadeIn(cb), FadeIn(ct))
        self.play(GrowArrow(a[4]), GrowArrow(a[5]), FadeIn(sig), FadeIn(w_b), FadeIn(w_t))
        self.play(GrowArrow(a[6]), FadeIn(dec))
        self.play(GrowArrow(a[7]), GrowArrow(a[8]), FadeIn(ok), FadeIn(no))
        self.wait(0.8)
        self.play(FadeIn(lab2), FadeIn(q))
        self.play(GrowArrow(b[0]), FadeIn(flt))
        self.play(GrowArrow(b[1]), GrowArrow(b[2]), FadeIn(dn), FadeIn(bm))
        self.play(GrowArrow(b[3]), GrowArrow(b[4]), FadeIn(rrf))
        self.play(GrowArrow(b[5]), FadeIn(top))
        self.play(FadeIn(foot))
        self.wait(3)


# ---------------------------------------------------------------------------
class S08Results(Scene):
    """Barres micro / macro-F1 test avec IC 95 % (bootstrap par clause)."""

    DATA = {
        "micro": [("TF-IDF", 87.70, 87.02, 88.37, GREEN), ("Legal-BERT", 86.35, 85.66, 87.02, ORANGE),
                  ("Combinaison", 88.43, 87.78, 89.06, BLUE)],
        "macro": [("TF-IDF", 82.19, 80.52, 83.00, GREEN), ("Legal-BERT", 76.27, 75.18, 77.03, ORANGE),
                  ("Combinaison", 82.84, 81.17, 83.68, BLUE)],
    }

    def construct(self):
        ttl = title("Test LEDGAR (10 000 clauses) : trois variantes")
        lo, hi, y0, H = 70, 92, -2.45, 4.6

        def Y(v):
            return y0 + (v - lo) / (hi - lo) * H

        panels = []
        for key, cx, name in [("micro", -4.4, "micro-F1"), ("macro", 0.4, "macro-F1")]:
            grid = VGroup()
            for g in [70, 75, 80, 85, 90]:
                grid.add(DashedLine([cx - 2.1, Y(g), 0], [cx + 2.1, Y(g), 0], color=EDGE, stroke_width=1,
                                    dash_length=0.08))
                grid.add(T(str(g), 16, MUTED).move_to([cx - 2.4, Y(g), 0]))
            head = T(name + " (test)", 26, weight=BOLD).move_to([cx, Y(hi) + 0.3, 0])
            bars, whisk, vals, names = VGroup(), VGroup(), VGroup(), VGroup()
            for i, (lab, v, l, h, col) in enumerate(self.DATA[key]):
                x = cx - 1.4 + i * 1.4
                r = Rectangle(width=0.95, height=Y(v) - y0, fill_color=col, fill_opacity=0.85, stroke_width=0)
                r.move_to([x, (Y(v) + y0) / 2, 0])
                bars.add(r)
                wk = VGroup(Line([x, Y(l), 0], [x, Y(h), 0], color=FG, stroke_width=2.5),
                            Line([x - 0.14, Y(l), 0], [x + 0.14, Y(l), 0], color=FG, stroke_width=2.5),
                            Line([x - 0.14, Y(h), 0], [x + 0.14, Y(h), 0], color=FG, stroke_width=2.5))
                whisk.add(wk)
                vals.add(T(fr(v), 22, FG, weight=BOLD).move_to([x, Y(h) + 0.25, 0]))
                names.add(T(lab, 16, col).move_to([x, y0 - 0.3, 0]))
            panels.append((grid, head, bars, whisk, vals, names))

        foot = footer("Axe tronqué à 70 · traits = IC 95 % bootstrap par clause (probablement un peu trop étroits)")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        for grid, head, bars, whisk, vals, names in panels:
            self.play(FadeIn(grid), FadeIn(head), FadeIn(names), run_time=0.6)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.25), run_time=1.4)
            self.play(Create(whisk), FadeIn(vals), run_time=0.8)
        self.play(FadeIn(foot))
        self.wait(0.8)

        # colonne d'annotations à droite
        bars_macro = panels[1][2]
        ref = DashedLine([0.4 - 1.4, Y(82.19), 0], [0.4 + 0.5, Y(82.19), 0], color=RED, stroke_width=2)
        gap = Arrow([0.4 + 0.5, Y(82.19), 0], [0.4 + 0.5, Y(76.27), 0], buff=0, color=RED, stroke_width=3,
                    tip_length=0.15)
        a1 = VGroup(
            T("Legal-BERT − TF-IDF", 22, RED, weight=BOLD),
            T("macro −5,92 pt [−6,88 ; −4,50]", 19, FG),
            T("micro −1,35 pt [−1,93 ; −0,77]", 19, FG),
            T("BERT seul décroche sur les types rares", 18, MUTED),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([5.0, 1.0, 0]).to_edge(RIGHT, buff=0.25)
        a2 = VGroup(
            T("Combinaison − TF-IDF", 22, BLUE, weight=BOLD),
            T("micro +0,73 pt [+0,51 ; +0,98]", 19, FG),
            T("macro +0,65 pt [+0,31 ; +1,00]", 19, FG),
            T("gain réel mais modeste", 18, MUTED),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([5.05, -1.25, 0])
        a2.align_to(a1, LEFT)
        self.play(Indicate(bars_macro[1], color=RED, scale_factor=1.03), Create(ref), GrowArrow(gap))
        self.play(FadeIn(a1, shift=LEFT * 0.2))
        self.wait(1.5)
        self.play(Indicate(panels[0][2][2], color=BLUE, scale_factor=1.03), Indicate(bars_macro[2], color=BLUE, scale_factor=1.03))
        self.play(FadeIn(a2, shift=LEFT * 0.2))
        self.wait(3)


# ---------------------------------------------------------------------------
class S09Ensemble(Scene):
    def construct(self):
        ttl = title("Combinaison calibrée, puis décision par seuil")
        formula = M(
            f'p = <span fgcolor="{ORANGE}">0,2</span> · softmax(z<span fgcolor="{ORANGE}"> BERT</span> / 1,04)'
            f'  +  <span fgcolor="{GREEN}">0,8</span> · softmax(z<span fgcolor="{GREEN}"> TF-IDF</span> / 0,91)', 30
        ).move_to(UP * 2.35)
        sub = T("poids : grille 0 → 1 (pas 0,05), meilleur macro-F1 de validation · températures apprises sur la validation",
                18, MUTED).next_to(formula, DOWN, buff=0.25)

        x0, W = 0.0, 6.0
        thr_x = x0 + 0.709 * W
        LEFT_X = -6.8

        def row(y, clause, label, conf, verdict, vcol, extra=None):
            c = T(clause, 19, FG).move_to([0, y + 0.6, 0]).align_to([LEFT_X, 0, 0], LEFT)
            frame = Rectangle(width=W, height=0.5, stroke_color=EDGE, stroke_width=1.5).move_to([x0 + W / 2, y - 0.1, 0])
            bar = Rectangle(width=max(conf * W, 0.04), height=0.5, fill_color=BLUE, fill_opacity=0.9,
                            stroke_width=0).move_to([x0 + conf * W / 2, y - 0.1, 0])
            lab = T(label, 22, FG, weight=BOLD).move_to([0, y - 0.1, 0]).align_to([LEFT_X, 0, 0], LEFT)
            val = T(fr(conf, 3), 21, BLUE, weight=BOLD).next_to(bar, RIGHT, buff=0.12)
            v = T(verdict, 20, vcol, weight=BOLD).next_to(lab, DOWN, buff=0.15).align_to(lab, LEFT)
            ex = T(extra, 17, MUTED).next_to(v, DOWN, buff=0.1).align_to(lab, LEFT) if extra else VGroup()
            return c, frame, bar, lab, val, v, ex

        ra = row(0.65, "« This Agreement shall be governed by … the laws of the State of New York … »",
                 "Governing Laws", 0.974, "≥ seuil → type renvoyé", GREEN,
                 "2ᵉ candidat : Applicable Laws 0,023 (quasi-synonyme)")
        rb = row(-1.65, "« Le présent contrat est régi par le droit français. »  (hors périmètre)",
                 "Amendments", 0.033, "< seuil → ABSTAIN : à un juriste", AMBER)

        thr = DashedLine([thr_x, 1.05, 0], [thr_x, -2.15, 0], color=AMBER, stroke_width=3, dash_length=0.1)
        thr_l = T("seuil 0,709", 19, AMBER, weight=BOLD).next_to(thr, DOWN, buff=0.08)
        foot = footer("Sorties réelles enregistrées (demo_outputs) · confiance = le TYPE est correct, pas la validité juridique")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(Write(formula), run_time=2)
        self.play(FadeIn(sub))
        self.wait(0.8)
        self.play(Create(thr), FadeIn(thr_l))
        for r in (ra, rb):
            c, frame, bar, lab, val, v, ex = r
            self.play(FadeIn(c), FadeIn(frame), FadeIn(lab), run_time=0.6)
            self.play(GrowFromEdge(bar, LEFT), run_time=1.0)
            self.play(FadeIn(val), FadeIn(v, shift=RIGHT * 0.2), FadeIn(ex))
            self.wait(0.8)
        self.play(FadeIn(foot))
        self.wait(3)


# ---------------------------------------------------------------------------
class S10Abstention(Scene):
    def construct(self):
        ttl = title("Abstention : sur 100 clauses du test (combinaison)")
        sq = VGroup(*[
            Square(side_length=0.44, stroke_width=0, fill_color=EDGE, fill_opacity=1) for _ in range(100)
        ]).arrange_in_grid(rows=10, cols=10, buff=0.08).move_to([-3.4, -0.05, 0])
        note = T("1 carré ≈ 1 % des clauses (arrondi)", 17, MUTED).next_to(sq, DOWN, buff=0.12)

        handled = sq[:84]
        abst = sq[84:]
        errors = [handled[i] for i in (7, 23, 41, 58, 76)]   # ≈ 4,6 % : positions illustratives

        info = VGroup(
            T("seuil 0,709 fixé sur la validation", 22, FG),
            T("(cible : 95 % de classifications correctes)", 18, MUTED),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        l1 = VGroup(Square(0.3, fill_color=BLUE, fill_opacity=1, stroke_width=0),
                    T("traitées : 84,2 % des clauses", 23, BLUE, weight=BOLD)).arrange(RIGHT, buff=0.2)
        l1b = T("dont 94,5 % correctement classées", 20, FG)
        l2 = VGroup(Square(0.3, fill_color=AMBER, fill_opacity=1, stroke_width=0),
                    T("renvoyées au juriste : 15,8 %", 23, AMBER, weight=BOLD)).arrange(RIGHT, buff=0.2)
        l3 = VGroup(Square(0.3, fill_color=BLUE, fill_opacity=1, stroke_color=RED, stroke_width=4),
                    T("erreurs malgré tout : ≈ 4,6 %", 23, RED, weight=BOLD)).arrange(RIGHT, buff=0.2)
        l3b = T("(= 84,2 % × 5,5 %)", 18, MUTED)
        warn = T("95 % est une cible de validation, pas une garantie.", 21, AMBER)
        right = VGroup(info, l1, l1b, l2, l3, l3b, warn).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([3.3, -0.2, 0])
        l1b.shift(RIGHT * 0.5)
        l3b.shift(RIGHT * 0.5)
        foot = footer("ECE test 0,017 · couverture validation 84,1 % → test 84,2 %")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(s) for s in sq], lag_ratio=0.01), FadeIn(note), run_time=1.5)
        self.play(FadeIn(info))
        self.play(LaggedStart(*[s.animate.set_fill(BLUE) for s in handled], lag_ratio=0.01), FadeIn(l1), run_time=1.6)
        self.play(FadeIn(l1b))
        self.play(LaggedStart(*[s.animate.set_fill(AMBER) for s in abst], lag_ratio=0.05), FadeIn(l2), run_time=1.2)
        self.wait(0.6)
        self.play(*[s.animate.set_stroke(RED, width=5) for s in errors], FadeIn(l3), FadeIn(l3b))
        self.wait(0.8)
        self.play(FadeIn(warn, shift=UP * 0.1))
        self.play(FadeIn(foot))
        self.wait(3)


# ---------------------------------------------------------------------------
class S11Retrieval(Scene):
    def construct(self):
        ttl = title("Recherche hybride : dense + BM25, fusion RRF")
        q = box("requête", "question\nou clause", w=1.9, h=1.1).move_to([-5.95, 1.5, 0])
        flt = box("filtre « N jours »", "garde les clauses qui\nmentionnent la durée", w=2.6, h=1.25, color=AMBER).move_to([-2.95, 1.5, 0])
        dn = box("dense", "bge-small · cosinus", w=2.3, h=0.95, color=BLUE).move_to([0.15, 2.25, 0])
        bm = box("BM25", "mots exacts · score > 0", w=2.3, h=0.95, color=GREEN).move_to([0.15, 0.75, 0])
        rrf = box("RRF", "Σ 1 / (60 + rang)", w=2.1, h=1.1, color=FG).move_to([2.95, 1.5, 0])
        top = box("top-k", "jamais complété\npar des non-classées", w=2.2, h=1.25).move_to([5.85, 1.5, 0])
        arr = [arrow(q.get_right(), flt.get_left()), arrow(flt.get_right(), dn.get_left()),
               arrow(flt.get_right(), bm.get_left()), arrow(dn.get_right(), rrf.get_left()),
               arrow(bm.get_right(), rrf.get_left()), arrow(rrf.get_right(), top.get_left())]
        bank = T("banque : 59 080 clauses du train, sans doublons exacts", 19, MUTED).move_to([0, -0.15, 0])

        # tableau clause -> clause
        head = T("Clause → clauses similaires · 981 requêtes du test · pertinent = même label LEDGAR", 20, FG, weight=BOLD)
        rows = [("", "P@1", "P@10", "MRR@10"), ("Dense", "76,6 %", "64,8 %", "0,814"),
                ("BM25", "77,0 %", "64,6 %", "0,817"), ("Hybride", "77,4 %", "66,4 %", "0,824")]
        table = VGroup()
        head.move_to([0, -0.85, 0])
        for i, r in enumerate(rows):
            col = BLUE if r[0] == "Hybride" else (MUTED if i == 0 else FG)
            wt = BOLD if r[0] == "Hybride" or i == 0 else NORMAL
            line = VGroup(*[T(c, 21, col, weight=wt) for c in r])
            for j, m in enumerate(line):
                m.move_to([-3.0 + j * 2.0, -1.45 - i * 0.48, 0])
            table.add(line)
        foot = footer("L'hybride bat chaque moteur seul, de peu · les 40 questions manuelles sont déjà vues : non-régression")

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(FadeIn(q))
        self.play(GrowArrow(arr[0]), FadeIn(flt))
        self.play(GrowArrow(arr[1]), GrowArrow(arr[2]), FadeIn(dn), FadeIn(bm))
        self.play(GrowArrow(arr[3]), GrowArrow(arr[4]), FadeIn(rrf))
        self.play(GrowArrow(arr[5]), FadeIn(top))
        self.play(FadeIn(bank))
        self.wait(0.8)
        self.play(FadeIn(head))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in table], lag_ratio=0.25))
        self.play(Indicate(table[3], color=BLUE, scale_factor=1.04))
        self.play(FadeIn(foot))
        self.wait(3)


# ---------------------------------------------------------------------------
class S13ControlledFailure(Scene):
    def construct(self):
        ttl = title("Échec contrôlé : la question « 30-day notice »")

        def chips(labels, good):
            g = VGroup()
            for i, (lab, ok) in enumerate(zip(labels, good)):
                col = GREEN if ok else RED
                g.add(box(f"{i + 1}. {lab}", None, w=2.5, h=0.62, color=col, size=19, head_color=col))
            return g.arrange(RIGHT, buff=0.15)

        qa = T("« either party may terminate on thirty days written notice »", 22, FG)
        ca = chips(["Terminations"] * 5, [True] * 5)
        ga = VGroup(T("Formulation proche d'une clause", 20, GREEN, weight=BOLD), qa, ca).arrange(DOWN, buff=0.2)
        ga.move_to(UP * 1.75)

        qb = T("« show me termination clauses with a 30-day notice period »", 22, FG)
        cb = chips(["Notices", "Notices", "Terms", "Terminations", "Disability"], [False, False, False, True, False])
        gb = VGroup(T("Formulation de juriste, même intention", 20, RED, weight=BOLD), qb, cb).arrange(DOWN, buff=0.2)
        gb.move_to(DOWN * 0.9)
        score = T("P@1 = 0 · 1 seul résultat pertinent parmi les 10 premiers", 21, RED).next_to(gb, DOWN, buff=0.3)

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        self.play(FadeIn(ga[0]), FadeIn(qa))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in ca], lag_ratio=0.15))
        self.wait(0.8)
        self.play(FadeIn(gb[0]), FadeIn(qb))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cb], lag_ratio=0.15))
        self.play(FadeIn(score))
        self.wait(1.2)

        # zoom sur le 1er résultat
        self.play(FadeOut(ga), FadeOut(score), FadeOut(qb), FadeOut(gb[0]), cb.animate.scale(0.85).move_to(UP * 2.4))
        first = cb[0]
        lines = [
            "In the event Participant believes that a “Good Reason” event has been triggered,",
            "Participant must give the Company written notice within 30 days",
            "of the first occurrence of such triggering event and a proposed termination",
            "date which shall be not sooner than 60 days nor later than 90 days after …",
        ]
        txt = VGroup(*[T(s, 22, FG) for s in lines]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        hdr = T("Résultat n° 1 (label Notices) — extrait de la clause renvoyée", 19, MUTED)
        card = VGroup(hdr, txt).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        frame = SurroundingRectangle(card, color=RED, buff=0.3, corner_radius=0.1, stroke_width=2.5)
        grp = VGroup(frame, card).move_to(UP * 0.2)
        hl = txt[1][-12:]   # « within30days » (les espaces ne sont pas des glyphes)
        hbox = SurroundingRectangle(hl, color=AMBER, buff=0.07, stroke_width=3)

        expl = VGroup(
            T("30 jours = délai pour notifier une « Good Reason »", 23, AMBER, weight=BOLD),
            T("≠ préavis de résiliation", 23, RED, weight=BOLD),
        ).arrange(DOWN, buff=0.12).next_to(grp, DOWN, buff=0.35)
        foot = footer("Le filtre vérifie qu'une durée est MENTIONNÉE, pas son rôle (ni jours ouvrés / calendaires)", FG)

        self.play(TransformFromCopy(first, frame), FadeIn(card))
        self.play(hl.animate.set_color(AMBER), Create(hbox))
        self.play(FadeIn(expl, shift=UP * 0.15))
        self.play(FadeIn(foot))
        self.wait(3.5)


# ---------------------------------------------------------------------------
class S14Audit(Scene):
    def construct(self):
        ttl = title("Audit : où le système est fragile (combinaison)")

        # ---- 1. groupes
        h1 = T("1 · Groupes (proxys par mots-clés)", 26, weight=BOLD).move_to(UP * 2.55)
        x0, W = -2.2, 7.0

        def X(v):  # échelle 60 % → 100 %
            return x0 + (v - 60) / 40 * W

        def hbar(y, v, col, txt):
            r = Rectangle(width=X(v) - x0, height=0.42, fill_color=col, fill_opacity=0.9, stroke_width=0)
            r.move_to([(X(v) + x0) / 2, y, 0])
            return VGroup(r, T(txt, 19, FG).next_to(r, RIGHT, buff=0.15))

        g_lab = T("Bail (n = 266)", 23, FG, weight=BOLD).move_to([-4.6, 1.35, 0])
        b1 = hbar(1.65, 75.2, RED, "exactitude 75,2 % [69,7 ; 80,0]")
        b2 = hbar(1.1, 84.3, EDGE, "attendu d'après le mix de types : 84,3 %")
        b3 = hbar(0.55, 87.6, BLUE, "correctes parmi les traitées : 87,6 %")
        t_lab = T("Classes de queue (n = 1 283)", 21, FG, weight=BOLD).move_to([-4.6, -0.3, 0])
        b4 = hbar(-0.3, 84.3, BLUE, "correctes parmi les traitées : 84,3 %")
        axis = VGroup(*[T(f"{v} %", 15, MUTED).move_to([X(v), -0.85, 0]) for v in (60, 70, 80, 90, 100)])
        m1 = T("L'abstention ne suffit pas sur le bail et les classes rares.", 21, AMBER).move_to(DOWN * 1.5)
        m2 = T("Groupes « she » (n = 37) et hors États-Unis (n = 36) : trop petits pour conclure.", 19, MUTED).next_to(m1, DOWN, buff=0.18)
        part1 = VGroup(h1, g_lab, b1, b2, b3, t_lab, b4, axis, m1, m2)

        self.play(FadeIn(ttl, shift=DOWN * 0.2), FadeIn(h1))
        self.play(FadeIn(g_lab))
        for b in (b1, b2, b3):
            self.play(GrowFromEdge(b[0], LEFT), FadeIn(b[1]), run_time=0.7)
        self.play(FadeIn(t_lab), GrowFromEdge(b4[0], LEFT), FadeIn(b4[1]), FadeIn(axis), run_time=0.8)
        self.play(FadeIn(m1))
        self.play(FadeIn(m2))
        self.wait(2.5)
        self.play(FadeOut(part1))

        # ---- 2. contrefactuels
        h2 = T("2 · Contrefactuels : le type prédit change-t-il ?", 26, weight=BOLD).move_to(UP * 2.55)
        cf = [("bruit OCR (3 % de lettres)", 5.4, 992), ("État US → juridiction européenne", 3.6, 698),
              ("he ↔ she", 2.3, 487), ("orthographe britannique", 2.2, 1275),
              ("noms de parties européens", 0.8, 3199), ("$ → EUR", 0.0, 365)]
        xs, Wc = -0.6, 5.2
        rows = VGroup()
        for i, (lab, v, n) in enumerate(cf):
            y = 1.7 - i * 0.62
            l = T(lab, 21, FG).move_to([-3.9, y, 0]).align_to([-0.85, 0, 0], RIGHT)
            r = Rectangle(width=max(v / 6.0 * Wc, 0.03), height=0.42, fill_color=AMBER if v >= 3 else BLUE,
                          fill_opacity=0.9, stroke_width=0)
            r.move_to([xs + r.width / 2, y, 0])
            n_txt = f"{n:,}".replace(",", " ")
            vt = T(f"{fr(v, 1)} %  (n = {n_txt})", 19, FG).next_to(r, RIGHT, buff=0.15)
            rows.add(VGroup(l, r, vt))
        w2 = T("Stabilité du type ≠ équivalence juridique : changer la juridiction change le droit applicable.",
               19, MUTED).move_to(DOWN * 2.15)
        ex = T("ex. Governing Laws → Waiver Of Jury Trials quand « Delaware » devient une juridiction européenne",
               18, MUTED).next_to(w2, DOWN, buff=0.15)

        self.play(FadeIn(h2))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in rows], lag_ratio=0.15),
                  run_time=1.8)
        self.play(FadeIn(w2), FadeIn(ex))
        self.wait(3)
        self.play(FadeOut(VGroup(h2, rows, w2, ex)))

        # ---- 3. hors périmètre
        h3 = T("3 · Hors périmètre : 8 entrées sans label valide", 26, weight=BOLD).move_to(UP * 2.55)
        inputs = T("RGPD · force majeure · non-concurrence · français · allemand · …", 21, MUTED).next_to(h3, DOWN, buff=0.3)
        cards = VGroup(
            box("Combinaison", "s'abstient sur 8 / 8", w=3.6, h=1.6, color=BLUE, size=26, sub_size=21),
            box("TF-IDF", "s'abstient sur 8 / 8", w=3.6, h=1.6, color=GREEN, size=26, sub_size=21),
            box("Legal-BERT seul", "clause RGPD → « General »\nconfiance 0,90 : erreur confiante", w=4.0, h=1.6,
                color=RED, size=26, sub_size=20),
        ).arrange(RIGHT, buff=0.4).move_to(UP * 0.1)
        w3 = T("Un seuil réduit le risque hors périmètre ; il ne garantit pas de le détecter.", 23, AMBER, weight=BOLD).move_to(DOWN * 1.8)
        w4 = T("8 entrées : un petit échantillon, pas une preuve.", 19, MUTED).next_to(w3, DOWN, buff=0.2)

        self.play(FadeIn(h3), FadeIn(inputs))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.3))
        self.play(Indicate(cards[2], color=RED, scale_factor=1.04))
        self.play(FadeIn(w3), FadeIn(w4))
        self.wait(3.5)


# ---------------------------------------------------------------------------
class S15Reproducibility(Scene):
    def construct(self):
        ttl = title("Reproductibilité : chaque chiffre est traçable")
        chain = VGroup(
            box("uv.lock", "dépendances\nverrouillées", w=2.3, h=1.55, color=EDGE),
            box("révisions figées", "LEDGAR · Legal-BERT\n· bge-small", w=2.3, h=1.55, color=EDGE),
            box("empreintes", "artefact modifié →\nrefusé ou reconstruit", w=2.3, h=1.55, color=BLUE),
            box("run_manifest.json", "commit · versions\n· empreintes", w=2.3, h=1.55, color=BLUE),
            box("92 tests ✓", "15 / 15 vérifications\ncroisées", w=2.3, h=1.55, color=GREEN),
        ).arrange(RIGHT, buff=0.55).move_to(UP * 1.1)
        arrs = VGroup(*[arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.05) for i in range(4)])

        ok = VGroup(
            T("Démontré sur cette machine (Apple M3)", 23, GREEN, weight=BOLD),
            T("copie propre + uv sync --frozen → 92 tests réussis", 20, FG),
            T("logits recalculés identiques bit à bit", 20, FG),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        ko = VGroup(
            T("Non démontré", 23, RED, weight=BOLD),
            T("reproductibilité sur une autre machine", 20, FG),
            T("variance entre graines (une seule graine)", 20, FG),
            T("Python 3.14 en version bêta", 20, FG),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        bottom = VGroup(ok, ko).arrange(RIGHT, buff=1.4, aligned_edge=UP).move_to(DOWN * 1.75)

        self.play(FadeIn(ttl, shift=DOWN * 0.2))
        for i, c in enumerate(chain):
            anims = [FadeIn(c, shift=RIGHT * 0.2)]
            if i:
                anims.append(GrowArrow(arrs[i - 1]))
            self.play(*anims, run_time=0.6)
        self.wait(0.6)
        self.play(FadeIn(ok, shift=UP * 0.15))
        self.play(FadeIn(ko, shift=UP * 0.15))
        self.wait(3.5)
