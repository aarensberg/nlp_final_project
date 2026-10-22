# Animations Manim de la soutenance

`scenes.py` contient 11 scènes (une par diapositive animée de [../slides/index.html](../slides/index.html)).
Tous les chiffres viennent de [../key_numbers.md](../key_numbers.md) ou des sorties enregistrées de [../demo_outputs/](../demo_outputs/).

Les vidéos rendues sont déjà dans `../slides/media/` (MP4 1080p 30 i/s) avec une image fixe finale par scène dans `../slides/media/posters/` (touche **S** de la présentation).

## Re-rendre (facultatif)

Manim est installé dans un environnement isolé : le projet NLP n'est pas modifié. Prérequis macOS : `brew install cairo pkgconf`.

```bash
cd presentation/manim
uv run --no-project --python 3.12 --with manim manim -r 1920,1080 --fps 30 --media_dir /tmp/manim_media -a scenes.py
cp /tmp/manim_media/videos/scenes/1080p30/*.mp4 ../slides/media/
```

Pas de LaTeX : le texte utilise `Text` / `MarkupText` (police Helvetica Neue).
