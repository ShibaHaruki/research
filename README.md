cd OneDrive - 学校法人立命館\ドキュメント\研究コード
pip freeze > requirements.txt
git add .
git commit -m "Add requirements"
git push

ssh coder.sotuken.main

cd ~
git clone git@github.com:ShibaHaruki/research.git

cd ~/research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pipreqs
pipreqs . --encoding=latin-1 --force

cd shiba_LSM

