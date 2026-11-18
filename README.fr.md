<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md"><b>Français</b></a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

Plan d’acquisition d’images SPOT-5 sous incertitude de nuages et de pannes, avec SCIP. Un tableau Streamlit est publié sur [aidecision.streamlit.app](https://aidecision.streamlit.app/).

```bash
pip install pyscipopt streamlit pandas matplotlib
streamlit run streamlit.py
python spotProbaPartial.py spotProba5 --criterion choquet --alpha 0.5
python spotProbaPartial.py --all
```


La documentation anglaise déjà présente dans ce dépôt reste dans le fichier anglais.
