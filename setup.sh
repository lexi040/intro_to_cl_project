#run to download the spacy languages corpuses 
#run using bash setup.sh


pip install -r requirements.txt
for m in en_core_web_sm fr_core_news_sm de_core_news_sm fi_core_news_sm; do
    python -m spacy download $m
done