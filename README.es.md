<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md"><b>Español</b></a>
</p>

Plan de adquisición de imágenes de SPOT-5 con incertidumbre de nubes y de fallos, con SCIP. Un tablero Streamlit está publicado en [aidecision.streamlit.app](https://aidecision.streamlit.app/).

```bash
pip install pyscipopt streamlit pandas matplotlib
streamlit run streamlit.py
python spotProbaPartial.py spotProba5 --criterion choquet --alpha 0.5
python spotProbaPartial.py --all
```


La documentación en inglés que ya estaba en este repositorio sigue en el archivo en inglés.
