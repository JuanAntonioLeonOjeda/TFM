---
title: LumosNet
emoji: 🦴
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: Osteoporosis lumbar en radiografías (TFM)
---

# LumosNet

Demo del TFM *Detección temprana de osteoporosis lumbar mediante aprendizaje profundo en radiografías
convencionales* (Juan Antonio León Ojeda, Máster en Inteligencia Artificial).

Una ResNet50 multitarea clasifica una radiografía lumbar en Sano, Osteopenia u Osteoporosis, estima la
densidad mineral ósea (BMD) y muestra con Grad-CAM en qué zonas se fijó. La web incluye una explicación del
proyecto por capítulos.

- Código: https://github.com/JuanAntonioLeonOjeda/TFM
- **Privacidad:** las radiografías que se suben se procesan solo en memoria. No se guardan, no se
  registran y no se usan para entrenar el modelo.
- Herramienta de investigación: no es un producto sanitario ni sustituye a una densitometría (DXA).

## Datos

El modelo se entrenó con [LUMOS](https://doi.org/10.5281/zenodo.18173664) y las radiografías de ejemplo
proceden de su conjunto de test. Se muestran preprocesadas: redimensionadas a 224×224 y con CLAHE.

> Shi, K., Shen, Q., Ye, Z., Jiang, L., Bu, J. y Wang, H. (2025). LUMOS: A Lumbar Multimodal Osteoporosis
> Screening Dataset with X-ray and CT images. *Proceedings of the 33rd ACM International Conference on
> Multimedia*. https://doi.org/10.1145/3746027.3758282

LUMOS se distribuye con licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Por
eso esta demo y el modelo son solo para uso no comercial.
